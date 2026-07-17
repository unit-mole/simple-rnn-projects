"""Project-wide configuration constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "sms_spam_simple_rnn_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

SAMPLE_DATA_PATH = DATA_DIR / "sample_sms_messages.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.csv"

DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/justmarkham/"
    "pycon-2016-tutorial/master/data/sms.tsv"
)

MESSAGE_COLUMN_CANDIDATES = ("message", "text", "sms", "body", "content", "v2")
LABEL_COLUMN_CANDIDATES = ("label", "target", "class", "category", "v1")

MAX_VOCABULARY = 5_000
MAX_SEQUENCE_LENGTH = 50
DECISION_THRESHOLD = 0.255
RANDOM_SEED = 42

MAX_BATCH_MESSAGES = 1_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_MESSAGE_CHARACTERS = 10_000
