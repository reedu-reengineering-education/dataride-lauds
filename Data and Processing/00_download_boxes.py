import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

BOXES = {
    "box01": "INSERT_OPENSENSEMAP_BOX_ID_01",
    "box02": "INSERT_OPENSENSEMAP_BOX_ID_02",
    "box03": "INSERT_OPENSENSEMAP_BOX_ID_03",
    "box04": "INSERT_OPENSENSEMAP_BOX_ID_04",
    "box05": "INSERT_OPENSENSEMAP_BOX_ID_05",
    "box06": "INSERT_OPENSENSEMAP_BOX_ID_06",
    "box07": "INSERT_OPENSENSEMAP_BOX_ID_07",
    "box08": "INSERT_OPENSENSEMAP_BOX_ID_08",
    "box09": "INSERT_OPENSENSEMAP_BOX_ID_09",
    "box10": "INSERT_OPENSENSEMAP_BOX_ID_10",
    "box11": "INSERT_OPENSENSEMAP_BOX_ID_11",
}

BASE = "https://api.opensensemap.org/boxes"

START = datetime(2026, 2, 1)
END = datetime(2026, 5, 21, 23, 59, 59)
STEP_DAYS = 3

def clean_name(s):
    return (
        str(s)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )

def make_chunks(start, end, step_days):
    chunks = []
    current = start

    while current < end:
        nxt = min(current + timedelta(days=step_days), end)
        label = f"{current.strftime('%Y%m%d')}_{nxt.strftime('%Y%m%d')}"
        chunks.append((
            label,
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            nxt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ))
        current = nxt

    return chunks

CHUNKS = make_chunks(START, END, STEP_DAYS)

def get_json(url, params=None, retries=6, sleep_seconds=8):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90)

            if r.status_code in [429, 500, 502, 503, 504]:
                print(f"  HTTP {r.status_code} attempt {attempt}/{retries}, retrying...")
                time.sleep(sleep_seconds * attempt)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            print(f"  request error attempt {attempt}/{retries}: {e}")
            time.sleep(sleep_seconds * attempt)

    return None

def download_box_chunk(box_name, box_id, chunk_label, from_date, to_date):
    print(f"\nDownloading {box_name} | {chunk_label}")

    meta = get_json(f"{BASE}/{box_id}")
    if not meta:
        return None, []

    sensors = meta.get("sensors", [])
    if not sensors:
        return None, []

    all_records = []
    log_rows = []

    for sensor in sensors:
        sensor_id = sensor.get("_id")
        sensor_title = sensor.get("title", "unknown")
        sensor_unit = sensor.get("unit", "")

        if not sensor_id:
            continue

        url = f"{BASE}/{box_id}/data/{sensor_id}"
        params = {
            "from-date": from_date,
            "to-date": to_date,
        }

        measurements = get_json(url, params=params)

        n = len(measurements) if isinstance(measurements, list) else 0
        truncated = n >= 10000

        log_rows.append({
            "box_name": box_name,
            "box_id": box_id,
            "chunk": chunk_label,
            "from_date": from_date,
            "to_date": to_date,
            "sensor_title": sensor_title,
            "sensor_id": sensor_id,
            "records": n,
            "possible_truncation": truncated,
        })

        if n == 0:
            print(f"  empty: {sensor_title}")
            time.sleep(0.5)
            continue

        if truncated:
            print(f"  WARNING possible truncation: {sensor_title}: {n} records")
        else:
            print(f"  {sensor_title}: {n} records")

        for m in measurements:
            loc = m.get("location") or []
            lon, lat = None, None

            if isinstance(loc, list) and len(loc) >= 2:
                lon, lat = loc[0], loc[1]

            all_records.append({
                "box_name": box_name,
                "box_id": box_id,
                "chunk": chunk_label,
                "sensor_id": sensor_id,
                "sensor_title": sensor_title,
                "sensor_unit": sensor_unit,
                "value": m.get("value"),
                "timestamp": m.get("createdAt"),
                "lat": lat,
                "lon": lon,
            })

        time.sleep(1)

    if not all_records:
        print(f"  No measurements found for {box_name} | {chunk_label}")
        return None, log_rows

    long_df = pd.DataFrame(all_records)

    long_path = f"data/raw/{clean_name(box_name)}_{chunk_label}_long.csv"
    long_df.to_csv(long_path, index=False)

    long_df["sensor_col"] = long_df["sensor_title"].apply(clean_name)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    wide = (
        long_df
        .pivot_table(
            index=["box_name", "box_id", "timestamp", "lat", "lon"],
            columns="sensor_col",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    wide_path = f"data/raw/{clean_name(box_name)}_{chunk_label}_wide.csv"
    wide.to_csv(wide_path, index=False)

    print(f"  Saved: {wide_path}")
    return wide_path, log_rows

def main():
    saved_files = []
    logs = []

    for box_name, box_id in tqdm(BOXES.items()):
        for chunk_label, from_date, to_date in CHUNKS:
            try:
                path, log_rows = download_box_chunk(
                    box_name, box_id, chunk_label, from_date, to_date
                )

                logs.extend(log_rows)

                if path:
                    saved_files.append(path)

            except Exception as e:
                print(f"FAILED {box_name} {chunk_label}: {e}")

            time.sleep(2)

    pd.DataFrame({"saved_file": saved_files}).to_csv(
        "data/logs/downloaded_files_chunked.csv", index=False
    )

    log_df = pd.DataFrame(logs)
    log_df.to_csv("data/logs/download_log_chunked.csv", index=False)

    print("\nDownload finished.")
    print(f"Files saved: {len(saved_files)}")
    print("Log: data/logs/download_log_chunked.csv")

    if not log_df.empty:
        trunc = log_df[log_df["possible_truncation"] == True]
        print(f"Possible truncations: {len(trunc)}")
        if len(trunc):
            print(trunc[["box_name", "chunk", "sensor_title", "records"]].head(20))

if __name__ == "__main__":
    main()
