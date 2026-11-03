#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 02 — Overtaking Event Detection

Input: outcome_01_standardized.csv from Step 01
Output: overtaking_events_final.csv

Main operations:
- Filter valid distance_left range: 30–300 cm
- Remove standing > 0.95 plus +/- 2 seconds buffer inside each trip
- Use 2-second temporal windows to detect candidates
- Candidate rule: dmin <= 250 cm and 30 <= var <= 120 cm
- Remove parked-car-like groups
- Consolidate nearby candidates into events
- Aggregate event-level attributes over the full event window
- Keep medium/high confidence events
- Export box_code as the box identifier
"""

import argparse
import math
from typing import List, Dict

import pandas as pd


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return 0.0
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def add_distance_along_trip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative distance along each trip using lat/lon.
    This is used only inside Step 03 for the parked-car spatial extent filter.
    It can reuse distance_along_trip_m from Step 01 when available.
    """
    df = df.copy()
    df["distance_along_trip_m"] = 0.0

    for trip_id, g in df.groupby("trip_id"):
        g = g.sort_values("timestamp_dt")
        cumulative = 0.0
        prev_lat = prev_lon = None

        for idx, row in g.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")

            if prev_lat is not None and prev_lon is not None:
                cumulative += haversine_m(prev_lat, prev_lon, lat, lon)

            df.loc[idx, "distance_along_trip_m"] = cumulative
            prev_lat, prev_lon = lat, lon

    return df


def classify_confidence(value: float) -> str:
    if pd.isna(value):
        return "low"
    if value >= 0.7:
        return "high"
    if value >= 0.2:
        return "medium"
    return "low"


def remove_standing_buffer(df: pd.DataFrame, standing_threshold: float, buffer_s: int) -> pd.DataFrame:
    df = df.copy()
    df["remove_standing_buffer"] = False

    for trip_id, g in df.groupby("trip_id"):
        standing_times = g.loc[g["standing"] > standing_threshold, "timestamp_dt"]
        if standing_times.empty:
            continue
        idx = g.index
        mask = pd.Series(False, index=idx)
        for t in standing_times:
            mask |= (g["timestamp_dt"] >= t - pd.to_timedelta(buffer_s, unit="s")) & \
                    (g["timestamp_dt"] <= t + pd.to_timedelta(buffer_s, unit="s"))
        df.loc[idx, "remove_standing_buffer"] = mask.values

    return df[~df["remove_standing_buffer"]].copy()


def detect_candidates(df: pd.DataFrame, window_s: int) -> pd.DataFrame:
    candidates = []

    for trip_id, g in df.groupby("trip_id"):
        g = g.sort_values("timestamp_dt").copy()
        for idx, row in g.iterrows():
            start = row["timestamp_dt"]
            end = start + pd.to_timedelta(window_s, unit="s")
            w = g[(g["timestamp_dt"] >= start) & (g["timestamp_dt"] <= end)]
            if len(w) < 2:
                continue

            dmin = w["distance_left"].min()
            dmax = w["distance_left"].max()
            var = dmax - dmin

            if dmin <= 250 and 30 <= var <= 120:
                critical_idx = w["distance_left"].idxmin()
                critical = g.loc[critical_idx].copy()
                critical["window_start"] = start
                critical["window_end"] = end
                critical["window_dmin"] = dmin
                critical["window_dmax"] = dmax
                critical["window_var"] = var
                candidates.append(critical)

    if not candidates:
        return pd.DataFrame(columns=list(df.columns) + ["window_start", "window_end", "window_dmin", "window_dmax", "window_var"])

    cand = pd.DataFrame(candidates).drop_duplicates(subset=["trip_id", "timestamp_dt", "window_dmin", "window_var"])
    return cand.sort_values(["trip_id", "timestamp_dt"]).reset_index(drop=True)


def remove_parked_car_patterns(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty:
        return cand

    cand = cand.copy().sort_values(["trip_id", "timestamp_dt"])
    cand["candidate_group"] = 0

    group_counter = 0
    for trip_id, g in cand.groupby("trip_id"):
        prev_time = None
        prev_dist = None
        for idx, row in g.iterrows():
            if prev_time is None:
                group_counter += 1
            else:
                dt = (row["timestamp_dt"] - prev_time).total_seconds()
                dd = abs(row.get("distance_along_trip_m", 0) - prev_dist)
                if dt > 3 or dd > 20:
                    group_counter += 1
            cand.loc[idx, "candidate_group"] = group_counter
            prev_time = row["timestamp_dt"]
            prev_dist = row.get("distance_along_trip_m", 0)

    remove_groups = set()
    for gid, g in cand.groupby("candidate_group"):
        n_events = len(g)
        duration = (g["timestamp_dt"].max() - g["timestamp_dt"].min()).total_seconds()
        spatial_extent = g["distance_along_trip_m"].max() - g["distance_along_trip_m"].min()

        criteria = [
            n_events >= 3,
            spatial_extent >= 15,
            duration >= 5,
        ]
        if sum(criteria) >= 2:
            remove_groups.add(gid)

    return cand[~cand["candidate_group"].isin(remove_groups)].copy()


def consolidate_events(cand: pd.DataFrame, source_df: pd.DataFrame, tolerance_s: int) -> pd.DataFrame:
    if cand.empty:
        return pd.DataFrame()

    cand = cand.copy().sort_values(["trip_id", "timestamp_dt"])
    event_rows: List[Dict] = []
    event_id_counter = 0

    for trip_id, g in cand.groupby("trip_id"):
        current = []
        prev_time = None

        for _, row in g.iterrows():
            if prev_time is None or (row["timestamp_dt"] - prev_time).total_seconds() <= tolerance_s:
                current.append(row)
            else:
                event_id_counter += 1
                event_rows.append(build_event(event_id_counter, current, source_df))
                current = [row]
            prev_time = row["timestamp_dt"]

        if current:
            event_id_counter += 1
            event_rows.append(build_event(event_id_counter, current, source_df))

    events = pd.DataFrame(event_rows)
    if events.empty:
        return events

    events["confidence"] = events["overtaking_pct_max"].apply(classify_confidence)
    events = events[events["confidence"].isin(["high", "medium"])].copy()
    return events.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)


def build_event(event_id: int, candidate_rows: List[pd.Series], source_df: pd.DataFrame) -> Dict:
    cdf = pd.DataFrame(candidate_rows)
    trip_id = cdf.iloc[0]["trip_id"]

    event_start = cdf["window_start"].min()
    event_end = cdf["window_end"].max()

    # Full event window aggregation from the cleaned source data.
    sdf = source_df[
        (source_df["trip_id"] == trip_id) &
        (source_df["timestamp_dt"] >= event_start) &
        (source_df["timestamp_dt"] <= event_end)
    ].copy()

    if sdf.empty:
        sdf = cdf.copy()

    critical_idx = sdf["distance_left"].idxmin()
    critical = sdf.loc[critical_idx]

    event_dmin = sdf["distance_left"].min()
    event_dmax = sdf["distance_left"].max()
    event_var = event_dmax - event_dmin

    return {
        "event_id": event_id,
        "timestamp": critical.get("timestamp_qgis", critical["timestamp_dt"].strftime("%Y-%m-%d %H:%M:%S")),
        "lat": critical.get("lat"),
        "lon": critical.get("lon"),
        "box_code": critical.get("box_code"),
        "trip_id": trip_id,
        "direction": critical.get("direction"),
        "event_start": event_start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end": event_end.strftime("%Y-%m-%d %H:%M:%S"),
        "event_duration_s": (event_end - event_start).total_seconds(),
        "event_dmin": event_dmin,
        "event_dmax": event_dmax,
        "event_var": event_var,
        "corridor_mean": sdf["corridor_width"].mean(),
        "corridor_min": sdf["corridor_width"].min(),
        "roughness_event_mean": sdf["roughness_score"].mean() if "roughness_score" in sdf.columns else None,
        "roughness_event_max": sdf["roughness_score"].max() if "roughness_score" in sdf.columns else None,
        "overtaking_pct_max": sdf["overtaking_manoeuvre"].max(),
        "overtaking_pct_mean": sdf["overtaking_manoeuvre"].mean(),
        "n_observations_event": len(sdf),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 02 — Overtaking event detection")
    parser.add_argument("--input", default="outcome_01_standardized.csv")
    parser.add_argument("--output", default="overtaking_events_final.csv")
    parser.add_argument("--candidates-output", default="overtaking_candidates_debug.csv")
    parser.add_argument("--window-seconds", type=int, default=2)
    parser.add_argument("--standing-threshold", type=float, default=0.95)
    parser.add_argument("--standing-buffer-seconds", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Recreate timestamp_dt in Step 02 if the compact Step 01 output only has timestamp_qgis.
    if "timestamp_dt" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], errors="coerce", utc=True)
    elif "timestamp_qgis" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_qgis"], errors="coerce", utc=True)
    else:
        raise SystemExit("ERROR: input needs either 'timestamp_dt' or 'timestamp_qgis'.")
    df["timestamp_dt"] = df["timestamp_dt"].dt.tz_convert(None)

    # Standardize roughness column name expected by this script.
    if "roughness_score" not in df.columns and "roughness_index" in df.columns:
        df["roughness_score"] = df["roughness_index"]

    for col in ["lat", "lon", "distance_left", "distance_right", "corridor_width",
                "overtaking_manoeuvre", "standing", "roughness_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    # Calculate cumulative distance if it was not already exported by Step 01.
    # This preserves spatial extent filtering for parked-car patterns.
    if "distance_along_trip_m" not in df.columns:
        df = add_distance_along_trip(df)
    else:
        df["distance_along_trip_m"] = pd.to_numeric(df["distance_along_trip_m"], errors="coerce").fillna(0)

    # Distance filter applied in Step 02.
    df = df[(df["distance_left"] >= 30) & (df["distance_left"] <= 300)].copy()
    after_distance = len(df)

    df = remove_standing_buffer(df, args.standing_threshold, args.standing_buffer_seconds)
    after_standing = len(df)

    candidates = detect_candidates(df, args.window_seconds)
    candidates_clean = remove_parked_car_patterns(candidates)
    events = consolidate_events(candidates_clean, df, tolerance_s=1)

    candidates_clean.to_csv(args.candidates_output, index=False)
    events.to_csv(args.output, index=False)

    print("Step 02 completed")
    print("Input rows:", before)
    print("After distance filter:", after_distance)
    print("After standing buffer filter:", after_standing)
    print("Candidates before parked-car removal:", len(candidates))
    print("Candidates after parked-car removal:", len(candidates_clean))
    print("Final events:", len(events))
    print("Candidates debug output:", args.candidates_output)
    print("Final output:", args.output)


if __name__ == "__main__":
    main()
