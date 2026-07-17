"""Project-wide configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "consumption_kwh"
DEFAULT_FREQUENCY = "h"
DEFAULT_LOOKBACK = 24
DEFAULT_FORECAST_HORIZON = 24
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

CALENDAR_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "weekend_flag",
    "peak_flag",
]
MODEL_FEATURE_COLUMNS = ["consumption_scaled", *CALENDAR_FEATURE_COLUMNS]
