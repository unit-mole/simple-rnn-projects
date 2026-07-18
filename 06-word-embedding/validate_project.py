from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MATRIX_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    TRAINING_CONFIG_PATH,
    VOCAB_PATH,
)
from src.embedding_pipeline import load_embedding_bundle


REQUIRED_FILES = [
    ".gitignore",
    ".python-version",
    "PROJECT_REVIEW.md",
    "README.md",
    "README_HOSTING.md",
    "requirements.txt",
    "requirements-ci.txt",
    "run_app.bat",
    "train_model.py",
    "validate_project.py",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/README_data.md",
    "data/sample_text.csv",
    "data/sample_sentences.txt",
    "data/topic_lexicon.json",
    "models/word_embedding_model.pt",
    "models/embedding_matrix.npy",
    "models/vocabulary.json",
    "models/model_metadata.json",
    "models/training_config.json",
    "models/artifact_checksums.json",
    "models/MODEL_CARD.md",
    "notebooks/word_embedding.ipynb",
    "notebooks/archive/word_embedding_original.ipynb",
    "outputs/model_metrics.json",
    "outputs/training_history.csv",
    "outputs/training_curve.png",
    "outputs/embedding_visualization_2d.png",
    "outputs/embedding_norm_distribution.png",
    "outputs/representation_comparison.csv",
    "src/config.py",
    "src/data_preprocessing.py",
    "src/text_preprocessing.py",
    "src/sequence_generation.py",
    "src/embedding_training.py",
    "src/embedding_analysis.py",
    "src/embedding_pipeline.py",
    "src/model_evaluation.py",
    "tests/test_model_loading.py",
    "tests/test_artifact_consistency.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_required_files() -> dict:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).exists()]
    return {"passed": not missing, "missing": missing}


def check_artifacts() -> dict:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vocabulary = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    matrix = np.load(EMBEDDING_MATRIX_PATH)
    expected_shape = (
        int(metadata["vocabulary_size"]),
        int(metadata["embedding_dimension"]),
    )
    shape_matches = matrix.shape == expected_shape
    vocabulary_matches = len(vocabulary["word_to_index"]) == expected_shape[0]
    finite = bool(np.isfinite(matrix).all())
    return {
        "passed": shape_matches and vocabulary_matches and finite,
        "expected_shape": list(expected_shape),
        "actual_shape": list(matrix.shape),
        "vocabulary_entries": len(vocabulary["word_to_index"]),
        "finite_values": finite,
    }


def check_model_inference() -> dict:
    bundle = load_embedding_bundle()
    nearest = bundle.nearest("quality", top_k=5)
    sentence = bundle.encode_sentence("quality inspection finds defect")
    passed = (
        len(nearest) == 5
        and nearest["status"].eq("in_vocabulary").all()
        and len(sentence["known_tokens"]) >= 3
        and np.isfinite(sentence["sentence_vector"]).all()
    )
    return {
        "passed": bool(passed),
        "nearest_words": nearest["word"].tolist(),
        "known_sentence_tokens": sentence["known_tokens"],
    }


def check_checksums() -> dict:
    expected = json.loads(
        (MODEL_DIR / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    path_by_name = {
        MODEL_PATH.name: MODEL_PATH,
        VOCAB_PATH.name: VOCAB_PATH,
        EMBEDDING_MATRIX_PATH.name: EMBEDDING_MATRIX_PATH,
        METADATA_PATH.name: METADATA_PATH,
        TRAINING_CONFIG_PATH.name: TRAINING_CONFIG_PATH,
        "sample_text.csv": PROJECT_ROOT / "data" / "sample_text.csv",
    }
    mismatches = {
        name: {"expected": expected.get(name), "actual": sha256(path)}
        for name, path in path_by_name.items()
        if expected.get(name) != sha256(path)
    }
    return {"passed": not mismatches, "mismatches": mismatches}


def check_notebooks() -> dict:
    notebook_results = {}
    for relative in [
        "notebooks/word_embedding.ipynb",
        "notebooks/archive/word_embedding_original.ipynb",
    ]:
        path = PROJECT_ROOT / relative
        try:
            notebook = nbformat.read(path, as_version=4)
            notebook_results[relative] = {
                "valid": True,
                "cells": len(notebook.cells),
            }
        except Exception as exc:
            notebook_results[relative] = {
                "valid": False,
                "error": str(exc),
            }
    return {
        "passed": all(item.get("valid") for item in notebook_results.values()),
        "notebooks": notebook_results,
    }


def check_readme() -> dict:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = [
        "## NLP Problem",
        "## Project Highlights",
        "## Dataset",
        "## Word Embedding Approach",
        "## Evaluation",
        "## Streamlit Application",
        "## Run Locally",
        "## Deployment",
        "## Known Limitations",
        "## Skills Demonstrated",
        "## Portfolio Description",
    ]
    missing = [section for section in required_sections if section not in text]
    return {"passed": not missing, "missing_sections": missing}


def main() -> None:
    checks = {
        "required_files": check_required_files(),
        "artifacts": check_artifacts(),
        "model_inference": check_model_inference(),
        "checksums": check_checksums(),
        "notebooks": check_notebooks(),
        "readme": check_readme(),
    }
    overall = all(item["passed"] for item in checks.values())
    report = {"overall_passed": overall, "checks": checks}
    output_path = PROJECT_ROOT / "VALIDATION_REPORT.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Word Embedding Project Validation")
    print("=" * 34)
    for name, result in checks.items():
        print(f"{name}: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"overall: {'PASSED' if overall else 'FAILED'}")

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
