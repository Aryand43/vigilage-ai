#!/usr/bin/env python3
"""
VigilAge AI — End-to-end pipeline
==================================
Usage:
    python main.py                  # full pipeline (load → train → evaluate → plot)
    python main.py --eval-only      # load saved model and evaluate on test set
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import tensorflow as tf

from src.config import DATA_DIR, MODEL_PATH, RANDOM_SEED
from src.data_loader import extract_sensor_arrays, load_umafall
from src.evaluate import compute_metrics, print_classification_report, print_metrics
from src.preprocessing import butter_lowpass_filter, preprocess_recordings
from src.train import subject_level_split, train_model
from src.visualize import plot_confusion_matrix, plot_raw_vs_filtered, plot_training_history


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(eval_only: bool = False) -> None:
    set_seeds()

    # ── 1. Load ──────────────────────────────────────────────
    print("=" * 60)
    print("  VigilAge AI — Proactive Fall Detection Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading UMAFall dataset …")
    df = load_umafall(DATA_DIR)
    records = extract_sensor_arrays(df)

    # ── 2. Preprocess ────────────────────────────────────────
    print("\n[2/5] Preprocessing (filter → window → normalise) …")
    X, y, subjects = preprocess_recordings(records)

    # ── Optional: plot one raw-vs-filtered signal ────────────
    sample_rec = records[0]["data"]
    filtered_rec = butter_lowpass_filter(sample_rec)
    plot_raw_vs_filtered(sample_rec, filtered_rec, save=True)

    # ── 3. Split ─────────────────────────────────────────────
    print("\n[3/5] Subject-level split (70 / 15 / 15) …")
    splits = subject_level_split(X, y, subjects)

    if eval_only:
        # ── Load existing model ──────────────────────────────
        print(f"\n[4/5] Loading saved model from {MODEL_PATH} …")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        # ── 4. Train ─────────────────────────────────────────
        print("\n[4/5] Training …")
        model, history = train_model(splits)
        plot_training_history(history, save=True)

    # ── 5. Evaluate ──────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    metrics = compute_metrics(model, splits["X_test"], splits["y_test"])
    print_metrics(metrics)
    print_classification_report(model, splits["X_test"], splits["y_test"])
    plot_confusion_matrix(model, splits["X_test"], splits["y_test"], save=True)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilAge AI pipeline")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load saved model and evaluate.",
    )
    args = parser.parse_args()
    main(eval_only=args.eval_only)
