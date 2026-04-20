"""
LSTM model definition for VigilAge AI fall detection.

Architecture (≈ 31 k trainable parameters):
    Input (200, 6) → LSTM-64 → Dropout(0.3)
                    → LSTM-32 → Dropout(0.3)
                    → Dense-16 (ReLU)
                    → Dense-1  (Sigmoid)
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from src.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    LSTM1_UNITS,
    LSTM2_UNITS,
)


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build and compile the two-layer LSTM fall detector.

    Parameters
    ----------
    input_shape : (time_steps, num_channels), e.g. (200, 6).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(LSTM1_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.LSTM(LSTM2_UNITS),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(DENSE_UNITS, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
