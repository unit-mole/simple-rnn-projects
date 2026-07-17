"""Central configuration for the Google stock Simple RNN project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATA_PATH = DATA_DIR / "sample_google_stock_data.csv"
MODEL_PATH = MODEL_DIR / "google_stock_rnn_model.keras"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET_COLUMN = "Close"
WINDOW_SIZE = 10
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
RNN_UNITS = 16
DROPOUT_RATE = 0.05
BATCH_SIZE = 16
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
