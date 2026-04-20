"""
UMAFall dataset loader.

Each CSV has a comment header (lines starting with '%' or blank) followed by
semicolon-delimited data rows:

    TimeStamp; SampleNo; X; Y; Z; SensorType; SensorID

Label (fall vs ADL) and subject ID are parsed from the filename:
    UMAFall_Subject_XX_{ADL|Fall}_ActivityName_Trial_Date.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    NUM_CHANNELS,
    TARGET_SENSOR_ID,
    TARGET_SENSOR_TYPES,
)

_FILENAME_RE = re.compile(
    r"UMAFall_(Subject_\d+)_(ADL|Fall)_(.+?)_\d+_\d{4}-\d{2}-\d{2}"
)

DATA_COLUMNS = ["timestamp", "sample_no", "x", "y", "z", "sensor_type", "sensor_id"]


def _parse_filename(path: Path) -> dict | None:
    """Extract subject_id, label (1=fall, 0=ADL), and activity from filename."""
    m = _FILENAME_RE.search(path.stem)
    if m is None:
        return None
    return {
        "subject_id": m.group(1),
        "label": 1 if m.group(2).lower() == "fall" else 0,
        "activity": m.group(3),
    }


def _load_single_csv(path: Path) -> pd.DataFrame | None:
    """Read one UMAFall CSV, returning cleaned rows with metadata columns."""
    meta = _parse_filename(path)
    if meta is None:
        return None

    rows: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("%"):
                continue
            rows.append(stripped)

    if not rows:
        return None

    from io import StringIO
    df = pd.read_csv(
        StringIO("\n".join(rows)),
        sep=";",
        header=None,
        names=DATA_COLUMNS,
    )

    df["subject_id"] = meta["subject_id"]
    df["label"] = meta["label"]
    df["activity"] = meta["activity"]
    df["source_file"] = path.name
    return df


def load_umafall(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load all UMAFall CSVs into a single DataFrame."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")

    frames = []
    for f in csv_files:
        df = _load_single_csv(f)
        if df is not None:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Loaded {len(csv_files)} files → {len(combined):,} rows")
    return combined


def extract_sensor_arrays(
    df: pd.DataFrame,
    sensor_id: int = TARGET_SENSOR_ID,
    sensor_types: list[int] = TARGET_SENSOR_TYPES,
) -> list[dict]:
    """Per-file, extract aligned accel+gyro arrays and metadata.

    Returns a list of dicts, each containing:
        - data: ndarray of shape (T, 6) — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        - label: int (0 or 1)
        - subject_id: str
        - activity: str
    """
    records: list[dict] = []
    grouped = df.groupby("source_file")

    for fname, file_df in grouped:
        sensor_df = file_df[file_df["sensor_id"].astype(float) == float(sensor_id)]
        if sensor_df.empty:
            continue

        arrays_by_type: dict[int, np.ndarray] = {}
        for st in sensor_types:
            sub = sensor_df[sensor_df["sensor_type"].astype(float) == float(st)].sort_values("sample_no")
            if sub.empty:
                break
            arrays_by_type[st] = sub[["x", "y", "z"]].values.astype(np.float64)

        if len(arrays_by_type) != len(sensor_types):
            continue

        # Align to shortest length across sensor types
        min_len = min(arr.shape[0] for arr in arrays_by_type.values())
        aligned = np.hstack([arrays_by_type[st][:min_len] for st in sensor_types])

        if aligned.shape[1] != NUM_CHANNELS:
            continue

        meta = file_df.iloc[0]
        records.append({
            "data": aligned,
            "label": int(meta["label"]),
            "subject_id": meta["subject_id"],
            "activity": meta["activity"],
        })

    n_fall = sum(1 for r in records if r["label"] == 1)
    n_adl = sum(1 for r in records if r["label"] == 0)
    print(f"[data_loader] Extracted {len(records)} recordings "
          f"({n_fall} fall, {n_adl} ADL)")
    return records
