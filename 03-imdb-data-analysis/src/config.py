"""Project-wide configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "imdb_simple_rnn_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

SAMPLE_DATA_PATH = DATA_DIR / "sample_reviews.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.csv"

TEXT_COLUMN_CANDIDATES = (
    "review",
    "text",
    "review_text",
    "comment",
    "content",
)

LABEL_COLUMN_CANDIDATES = (
    "label",
    "sentiment",
    "target",
)

MAX_VOCABULARY = 10_000
CHUNK_LENGTH = 80
CHUNK_STRIDE = 60
MAX_CHUNKS = 6
DECISION_THRESHOLD = 0.43
RANDOM_SEED = 42

MAX_BATCH_REVIEWS = 1_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_REVIEW_CHARACTERS = 50_000

LIVE_APP_URL = (
    "https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/"
)
