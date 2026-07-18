import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import (
    EMBEDDING_MATRIX_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    TRAINING_CONFIG_PATH,
    VOCAB_PATH,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_metadata_vocabulary_and_matrix_are_consistent():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vocabulary = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    matrix = np.load(EMBEDDING_MATRIX_PATH)
    assert len(vocabulary["word_to_index"]) == metadata["vocabulary_size"]
    assert matrix.shape == (
        metadata["vocabulary_size"],
        metadata["embedding_dimension"],
    )


def test_artifact_checksums_match():
    checksums = json.loads(
        (MODEL_DIR / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    paths = [
        MODEL_PATH,
        VOCAB_PATH,
        EMBEDDING_MATRIX_PATH,
        METADATA_PATH,
        TRAINING_CONFIG_PATH,
        Path(__file__).resolve().parents[1] / "data" / "sample_text.csv",
    ]
    for path in paths:
        assert checksums[path.name] == _sha256(path)
