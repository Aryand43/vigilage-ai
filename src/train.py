"""
Training utilities: subject-level split and model fitting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.model import build_lstm_model


# ── Subject-level split ──────────────────────────────────────

def subject_level_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, np.ndarray]:
    """70 / 15 / 15 split with no subject overlap between sets."""
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=1 - TRAIN_RATIO, random_state=RANDOM_SEED,
    )
    train_idx, tmp_idx = next(gss1.split(X, y, groups=subjects))

    X_train, y_train = X[train_idx], y[train_idx]
    X_tmp, y_tmp = X[tmp_idx], y[tmp_idx]
    subj_tmp = subjects[tmp_idx]

    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=1 - relative_val, random_state=RANDOM_SEED,
    )
    val_idx, test_idx = next(gss2.split(X_tmp, y_tmp, groups=subj_tmp))

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_tmp[val_idx], "y_val": y_tmp[val_idx],
        "X_test": X_tmp[test_idx], "y_test": y_tmp[test_idx],
    }

    for key in ("train", "val", "test"):
        n = len(splits[f"y_{key}"])
        pos = int(splits[f"y_{key}"].sum())
        print(f"  {key:>5s}: {n:>6,} windows  ({pos} fall, {n - pos} ADL)")

    return splits


# ── Training ─────────────────────────────────────────────────

def train_model(
    splits: dict[str, np.ndarray],
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Build, train, and save the LSTM model."""
    input_shape = (splits["X_train"].shape[1], splits["X_train"].shape[2])
    model = build_lstm_model(input_shape)
    model.summary()

    history = model.fit(
        splits["X_train"],
        splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\n[train] Model saved to {MODEL_PATH}")
    return model, history
