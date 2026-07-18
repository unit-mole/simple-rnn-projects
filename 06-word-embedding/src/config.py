from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

DEFAULT_DATA_PATH = DATA_DIR / "sample_text.csv"
TOPIC_LEXICON_PATH = DATA_DIR / "topic_lexicon.json"

MODEL_PATH = MODEL_DIR / "word_embedding_model.pt"
VOCAB_PATH = MODEL_DIR / "vocabulary.json"
EMBEDDING_MATRIX_PATH = MODEL_DIR / "embedding_matrix.npy"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TRAINING_CONFIG_PATH = MODEL_DIR / "training_config.json"

SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
