"""
Central configuration for the VigilAge AI pipeline.
All hyperparameters, paths, and constants live here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "vigilage_lstm.keras"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Sensor ───────────────────────────────────────────────────
SAMPLING_RATE = 100          # Hz (UMAFall native rate)
SENSOR_TYPES = {0: "accelerometer", 1: "gyroscope", 2: "magnetometer"}
# We use accel (0) + gyro (1) from a single SensorTag node.
# Sensor ID 3 = wrist SensorTag (best coverage: 726/746 files).
TARGET_SENSOR_ID = 3
TARGET_SENSOR_TYPES = [0, 1]  # accelerometer + gyroscope
NUM_CHANNELS = 6              # 3 accel axes + 3 gyro axes

# ── Preprocessing ────────────────────────────────────────────
BUTTER_CUTOFF = 20.0         # Hz
BUTTER_ORDER = 3
WINDOW_SIZE = 200            # samples (2 s at 100 Hz)
WINDOW_STEP = 100            # samples (50 % overlap → 1 s step)

# ── Training ─────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

LSTM1_UNITS = 64
LSTM2_UNITS = 32
DENSE_UNITS = 16

RANDOM_SEED = 42
