"""
Plotting utilities for training curves, confusion matrices, and signal inspection.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import OUTPUT_DIR


def plot_training_history(
    history: tf.keras.callbacks.History,
    save: bool = True,
) -> None:
    """Plot training/validation loss and accuracy side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.history["loss"]) + 1)

    ax1.plot(epochs, history.history["loss"], label="Training")
    ax1.plot(epochs, history.history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history["accuracy"], label="Training")
    ax2.plot(epochs, history.history["val_accuracy"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("VigilAge AI — Training Curves", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "training_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved → {path}")
    plt.show()


def plot_confusion_matrix(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    save: bool = True,
) -> None:
    """Plot and optionally save the confusion matrix."""
    y_pred = (model.predict(X, verbose=0).ravel() >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["ADL", "Fall"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("VigilAge AI — Confusion Matrix (Test Set)")
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved → {path}")
    plt.show()


def plot_raw_vs_filtered(
    raw: np.ndarray,
    filtered: np.ndarray,
    channel_names: Optional[list[str]] = None,
    max_samples: int = 500,
    save: bool = True,
) -> None:
    """Compare raw and filtered signals for the first few hundred samples."""
    n_ch = raw.shape[1]
    if channel_names is None:
        channel_names = [
            "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
        ][:n_ch]

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    t = np.arange(min(max_samples, raw.shape[0]))
    for i, ax in enumerate(axes):
        ax.plot(t, raw[t, i], alpha=0.4, label="Raw")
        ax.plot(t, filtered[t, i], label="Filtered (20 Hz LP)")
        ax.set_ylabel(channel_names[i])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Sample")
    fig.suptitle("Raw vs. Butterworth-Filtered IMU Signal", fontsize=13, y=1.01)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "raw_vs_filtered.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved → {path}")
    plt.show()
