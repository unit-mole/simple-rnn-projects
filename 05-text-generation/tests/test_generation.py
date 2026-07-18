"""Unit tests for deterministic character sampling and prompt preparation."""

from __future__ import annotations

import numpy as np
import pytest

from src.text_generator import prepare_seed, sample_next_index
from src.text_preprocessing import CharacterVocabulary


def test_sample_next_index_is_reproducible() -> None:
    logits = np.asarray([0.1, 0.4, 0.2, 0.8], dtype=np.float64)
    first = sample_next_index(logits, temperature=0.7, top_k=3, rng=np.random.default_rng(42))
    second = sample_next_index(logits, temperature=0.7, top_k=3, rng=np.random.default_rng(42))
    assert first == second


def test_sample_next_index_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        sample_next_index(np.asarray([0.2, 0.8]), temperature=0.0)


def test_prepare_seed_left_pads_to_sequence_length() -> None:
    vocabulary = CharacterVocabulary.fit("Alice was beginning ")
    prepared = prepare_seed("Alice", sequence_length=10, vocabulary=vocabulary)
    assert len(prepared) == 10
    assert prepared.endswith("Alice")
