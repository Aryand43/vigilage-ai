"""
Model evaluation: metrics, confusion matrix, and classification report.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and specificity."""
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    print("\n╔══════════════════════════════════════════╗")
    print("║       VigilAge AI — Test-Set Metrics     ║")
    print("╠══════════════════════════════════════════╣")
    for name, value in metrics.items():
        print(f"║  {name:<14s}   {value:>7.4f}   ({value * 100:.1f}%)   ║")
    print("╚══════════════════════════════════════════╝\n")


def print_classification_report(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Print a full sklearn classification report."""
    y_pred = (model.predict(X, verbose=0).ravel() >= threshold).astype(int)
    print(classification_report(
        y_true, y_pred, target_names=["ADL", "Fall"], zero_division=0,
    ))
