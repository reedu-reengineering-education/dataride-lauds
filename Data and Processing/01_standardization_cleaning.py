#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01 — Standardization & Initial Cleaning

Input folder: CSV files containing senseBox/openSenseMap records.
Output: outcome_01_standardized.csv

Main operations:
- Read all CSV files from input folder
- Preserve source_file internally for traceability, but do not export it in the final dataset
- Create anonymized_box_id and box_code for each box
- Export a box mapping table for audit/linking
- Standardize timestamps for QGIS
- Sort by box and timestamp
- Create trip_id using time gaps > 5 minutes across the whole box timeline
- Add simple outbound/return direction heuristic
- Compute corridor_width = distance_left + distance_right
- Compute roughness_score on a 0–1 scale from surface columns
- Remove rows without valid GPS
- Remove rows without relevant sensor measurements
- Remove first/last 15 seconds of each trip
- Recalculate and export clean distance_along_trip_m after trimming
- Export a compact CSV containing only columns required for downstream processing and QGIS

Distance filtering is NOT applied here. It is applied in later processing steps.
"""

import argparse
import os
import glob
import math
import re
from typing import Dict, Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "box_name", "box_id", "timestamp", "lat", "lon",
    "distance_left", "distance_right",
    "finedust_pm1", "finedust_pm10", "finedust_pm2.5", "finedust_pm4",
    "overtaking_manoeuvre", "speed", "standing",
    "surface_asphalt", "surface_compacted", "surface_paving", "surface_sett",
]

# Known campaign boxes. This keeps the exported box_code stable and easy to audit.
BOX_CODE_BY_ID: Dict[str, str] = {
    "698ccd2105a5710008551a2b": "box01",  # gilbert
    "698cdc5005a5710008710636": "box02",  # 5464...
    "698ddaf2f5b607000a02edf7": "box03",  # hugo
    "698ed8ef6733b60007e073ab": "box04",  # alexgab
    "69b02f7fd3b9ec00072884f8": "box05",  # XDL LIMITED EDITION
    "69b45478f419d40007dd5793": "box06",  # Gilbert
    "69bd02eecbc1700007543bc3": "box07",  # filletourbillon
    "69ef9da2f3b80b000802e1ca": "box08",  # axelos
    "69f1f5f087cacc00086c810e": "box09",  # murda
    "69e6180f5a890400074564d9": "box10",  # paquerette
    "69f31490d4712600087295a6": "box11",  # épice
}


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return 0.0
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def clean_name(s) -> str:
    return (
        str(s)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )


def infer_box_code(row: pd.Series, fallback_number: int = None) -> str:
    """Return a stable campaign box code such as box01, box02, ..."""
    box_id = str(row.get("box_id", "") or "").strip()
    if box_id in BOX_CODE_BY_ID:
        return BOX_CODE_BY_ID[box_id]

    # Try to recover box number from box_name or source_file, e.g. gilbert_box01.
    text = f"{row.get('box_name', '')} {row.get('source_file', '')}"
    match = re.search(r"box[_\s-]*0?(\d{1,2})", str(text), flags=re.IGNORECASE)
    if match:
        return f"box{int(match.group(1)):02d}"

    if fallback_number is not None:
        return f"box{fallback_number:02d}"

    return "box_unknown"


def read_raw_csvs(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise SystemExit("No CSV files found in input folder.")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    for col in REQUIRED_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged


def assign_box_identifiers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add box_code and anonymized_box_id.

    box_code is a campaign-friendly code such as box01.
    anonymized_box_id is numeric and stable within this run.
    A mapping table is returned for export.
    """
    df = df.copy()

    key = df["box_id"].fillna("").astype(str).str.strip()
    fallback = df["box_name"].fillna(df["source_file"]).astype(str).str.strip()
    key = key.where(key != "", fallback)
    df["_box_key"] = key

    # First assign known/inferred box codes row-wise.
    df["box_code"] = df.apply(infer_box_code, axis=1)

    # For any remaining unknowns, assign deterministic unknown codes.
    unknown_keys = sorted(df.loc[df["box_code"] == "box_unknown", "_box_key"].dropna().unique())
    unknown_mapping = {k: f"box_unknown_{i + 1:02d}" for i, k in enumerate(unknown_keys)}
    if unknown_mapping:
        df.loc[df["box_code"] == "box_unknown", "box_code"] = df.loc[
            df["box_code"] == "box_unknown", "_box_key"
        ].map(unknown_mapping)

    # Numeric anonymized IDs follow the box_code order.
    box_codes = sorted(df["box_code"].dropna().unique())
    anon_mapping = {code: i + 1 for i, code in enumerate(box_codes)}
    df["anonymized_box_id"] = df["box_code"].map(anon_mapping).astype("Int64")

    mapping_df = (
        df[["box_code", "anonymized_box_id", "box_id", "box_name", "_box_key"]]
        .drop_duplicates()
        .sort_values(["anonymized_box_id", "box_code", "box_id", "box_name"], na_position="last")
        .reset_index(drop=True)
    )

    return df, mapping_df


def compute_roughness_score_0_1(df: pd.DataFrame) -> pd.Series:
    """
    Compute roughness on a 0–1 scale where:
    - 0 = smoothest/asphalt-dominant
    - 1 = roughest/sett-dominant

    Surface columns may be probabilities in 0–1 or percentages in 0–100.
    The calculation normalizes by the sum of available surface scores, so both
    scales work as long as all surface columns use the same scale.
    """
    surface_cols = ["surface_asphalt", "surface_paving", "surface_compacted", "surface_sett"]
    surfaces = df[surface_cols].copy()

    # Treat missing surface probabilities as zero for weighted aggregation,
    # but return NaN when no surface information is present.
    valid_any = surfaces.notna().any(axis=1)
    surfaces = surfaces.fillna(0)

    surface_sum = surfaces.sum(axis=1)
    weighted_sum = (
        1 * surfaces["surface_asphalt"] +
        2 * surfaces["surface_paving"] +
        3 * surfaces["surface_compacted"] +
        4 * surfaces["surface_sett"]
    )

    weighted_avg = weighted_sum / surface_sum.where(surface_sum != 0)
    score_0_1 = ((weighted_avg - 1.0) / 3.0).clip(lower=0, upper=1)
    score_0_1 = score_0_1.where(valid_any)
    return score_0_1


def add_distance_along_trip(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate clean cumulative distance after any trimming/filtering."""
    df = df.sort_values(["box_code", "trip_id", "timestamp_dt"]).copy()
    df["distance_along_trip_m"] = 0.0

    for trip_id, idx in df.groupby("trip_id", sort=False).groups.items():
        idx = list(idx)
        cumulative = [0.0]
        for i in range(1, len(idx)):
            a = df.loc[idx[i - 1]]
            b = df.loc[idx[i]]
            cumulative.append(cumulative[-1] + haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]))
        df.loc[idx, "distance_along_trip_m"] = cumulative

    return df


def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, box_mapping = assign_box_identifiers(df)

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_dt"]).copy()
    df["timestamp_qgis"] = df["timestamp_dt"].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")

    for col in [
        "lat", "lon", "distance_left", "distance_right", "overtaking_manoeuvre",
        "speed", "standing", "surface_asphalt", "surface_compacted",
        "surface_paving", "surface_sett",
    ]:
        df[col] = to_num(df[col])

    # Remove invalid GPS rows.
    before_gps = len(df)
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()
    removed_gps = before_gps - len(df)

    # Remove rows without any relevant sensor measurements.
    sensor_columns = [
        "distance_left",
        "distance_right",
        "overtaking_manoeuvre",
        "speed",
        "standing",
    ]
    existing_sensor_columns = [c for c in sensor_columns if c in df.columns]
    before_sensor_empty = len(df)
    if existing_sensor_columns:
        df = df.dropna(how="all", subset=existing_sensor_columns).copy()
    removed_empty_sensor_rows = before_sensor_empty - len(df)

    # Corridor width: left + right, only when both values exist.
    df["corridor_width"] = df["distance_left"] + df["distance_right"]

    # Roughness score on 0–1 scale, robust to input scale 0–1 or 0–100.
    df["roughness_score"] = compute_roughness_score_0_1(df)

    # Sort by the whole box timeline. source_file is not used for trip creation anymore.
    df = df.sort_values(["box_code", "timestamp_dt"]).reset_index(drop=True)

    # Trip creation by temporal discontinuity > 5 minutes within each box.
    group_cols = ["box_code"]
    df["time_gap_s"] = df.groupby(group_cols)["timestamp_dt"].diff().dt.total_seconds()
    df["new_trip"] = (df["time_gap_s"].isna()) | (df["time_gap_s"] > 300)
    df["trip_number"] = df.groupby(group_cols)["new_trip"].cumsum().astype(int)
    df["trip_id"] = df["box_code"] + "_trip" + df["trip_number"].astype(str).str.zfill(4)

    # Simple outbound/return heuristic based on the dominant displacement per trip.
    directions = {}
    for trip_id, g in df.groupby("trip_id", sort=False):
        first = g.iloc[0]
        last = g.iloc[-1]
        dx = last["lon"] - first["lon"]
        dy = last["lat"] - first["lat"]
        directions[trip_id] = "outbound" if (abs(dx) >= abs(dy) and dx >= 0) or (abs(dy) > abs(dx) and dy >= 0) else "return"
    df["direction"] = df["trip_id"].map(directions)

    # Trim first/last 15 seconds of each trip.
    trim_s = 15
    trip_start = df.groupby("trip_id")["timestamp_dt"].transform("min")
    trip_end = df.groupby("trip_id")["timestamp_dt"].transform("max")
    keep = (df["timestamp_dt"] >= trip_start + pd.to_timedelta(trim_s, unit="s")) & \
           (df["timestamp_dt"] <= trip_end - pd.to_timedelta(trim_s, unit="s"))
    removed_trim = int((~keep).sum())
    df = df[keep].copy()

    # Recalculate clean distance along trip after trimming, so each exported trip starts at 0 m.
    df = add_distance_along_trip(df)

    # Compact export schema. source_file is intentionally not exported.
    keep_columns = [
        "timestamp",
        "timestamp_qgis",
        "lat",
        "lon",
        "box_code",
        "anonymized_box_id",
        "trip_id",
        "direction",
        "distance_along_trip_m",
        "distance_left",
        "distance_right",
        "corridor_width",
        "finedust_pm1",
        "finedust_pm10",
        "finedust_pm2.5",
        "finedust_pm4",
        "overtaking_manoeuvre",
        "speed",
        "standing",
        "roughness_score",
    ]

    df = df[[c for c in keep_columns if c in df.columns]].copy()

    df.attrs["removed_gps"] = removed_gps
    df.attrs["removed_empty_sensor_rows"] = removed_empty_sensor_rows
    df.attrs["removed_trim"] = removed_trim
    return df, box_mapping


def main():
    parser = argparse.ArgumentParser(description="Step 01 — Standardization & initial cleaning")
    parser.add_argument("--input-dir", default="/Users/arturkimshum/nancy_campaign_V5/data/raw")
    parser.add_argument("--output", default="outcome_01_standardized.csv")
    parser.add_argument("--box-mapping-output", default="data/logs/box_code_mapping.csv")
    args = parser.parse_args()

    df_raw = read_raw_csvs(args.input_dir)
    original_rows = len(df_raw)
    df, box_mapping = standardize(df_raw)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mapping_dir = os.path.dirname(args.box_mapping_output)
    if mapping_dir:
        os.makedirs(mapping_dir, exist_ok=True)

    df.to_csv(args.output, index=False)
    box_mapping.to_csv(args.box_mapping_output, index=False)

    print("Step 01 completed")
    print("Input rows:", original_rows)
    print("Output rows:", len(df))
    print("Removed invalid GPS:", df.attrs.get("removed_gps", "n/a"))
    print("Removed empty sensor rows:", df.attrs.get("removed_empty_sensor_rows", "n/a"))
    print("Removed by trip trimming:", df.attrs.get("removed_trim", "n/a"))
    print("Output:", args.output)
    print("Box mapping:", args.box_mapping_output)


if __name__ == "__main__":
    main()
