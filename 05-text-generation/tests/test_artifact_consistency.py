"""Consistency checks for committed inference artifacts and deployment metadata."""

from __future__ import annotations

import json
from pathlib import Path

from src.text_generator import load_generation_bundle, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_metadata_matches_committed_artifacts() -> None:
    metadata = json.loads(
        (PROJECT_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8")
    )
    model_path = PROJECT_ROOT / "models" / "text_generation_simple_rnn_model.pt"
    corpus_path = PROJECT_ROOT / "data" / "sample_text.txt"

    assert metadata["model_sha256"] == sha256_file(model_path)
    assert metadata["corpus_sha256"] == sha256_file(corpus_path)
    assert metadata["model_size_bytes"] == model_path.stat().st_size


def test_loaded_model_parameter_count_matches_metadata() -> None:
    bundle = load_generation_bundle(PROJECT_ROOT / "models")
    parameter_count = sum(parameter.numel() for parameter in bundle.model.parameters())
    assert parameter_count == int(bundle.metadata["model_parameters"])
