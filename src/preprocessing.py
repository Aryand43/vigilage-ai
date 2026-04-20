"""
Preprocessing pipeline: Butterworth filter → sliding windows → z-score normalisation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from src.config import (
    BUTTER_CUTOFF,
    BUTTER_ORDER,
    SAMPLING_RATE,
    WINDOW_SIZE,
    WINDOW_STEP,
)


# ── Stage 1: Butterworth low-pass filter ─────────────────────

def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = BUTTER_CUTOFF,
    fs: float = SAMPLING_RATE,
    order: int = BUTTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter applied per channel.

    Parameters
    ----------
    data : (T, C) or (T,)
    cutoff : cutoff frequency in Hz
    fs : sampling rate in Hz
    order : filter order

    Returns
    -------
    Filtered array with the same shape.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    if data.ndim == 1:
        return filtfilt(b, a, data).astype(np.float32)
    return np.column_stack(
        [filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]
    ).astype(np.float32)


# ── Stage 2: Sliding window segmentation ─────────────────────

def create_sliding_windows(
    data: np.ndarray,
    label: int,
    window_size: int = WINDOW_SIZE,
    step_size: int = WINDOW_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment a single recording into fixed-length overlapping windows.

    Parameters
    ----------
    data : (T, C)
    label : scalar label applied to every window from this recording

    Returns
    -------
    X : (num_windows, window_size, C)
    y : (num_windows,)
    """
    T, C = data.shape
    starts = range(0, T - window_size + 1, step_size)
    if len(starts) == 0:
        return np.empty((0, window_size, C), dtype=np.float32), np.empty(0, dtype=np.int32)

    X = np.stack([data[s : s + window_size] for s in starts], dtype=np.float32)
    y = np.full(len(starts), label, dtype=np.int32)
    return X, y


# ── Stage 3: Z-score normalisation per window ────────────────

def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score normalisation.

    Parameters
    ----------
    X : (N, T, C)

    Returns
    -------
    X_norm : same shape, zero mean / unit variance per channel per window.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


# ── Full pipeline for a list of recording dicts ──────────────

def preprocess_recordings(
    records: list[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete preprocessing pipeline on extracted recordings.

    Parameters
    ----------
    records : list of dicts from ``data_loader.extract_sensor_arrays``.

    Returns
    -------
    X : (N, WINDOW_SIZE, NUM_CHANNELS) normalised windows
    y : (N,) labels
    subjects : (N,) subject ID repeated per window
    """
    all_X, all_y, all_subjects = [], [], []

    for rec in records:
        filtered = butter_lowpass_filter(rec["data"])
        X_w, y_w = create_sliding_windows(filtered, rec["label"])
        if len(y_w) == 0:
            continue
        all_X.append(X_w)
        all_y.append(y_w)
        all_subjects.append(np.full(len(y_w), rec["subject_id"], dtype=object))

    X = zscore_normalize(np.concatenate(all_X))
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)

    print(f"[preprocessing] {X.shape[0]:,} windows  "
          f"({y.sum()} fall, {(y == 0).sum()} ADL)  "
          f"shape={X.shape}")
    return X, y, subjects
