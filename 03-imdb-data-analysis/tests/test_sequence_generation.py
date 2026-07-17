import numpy as np

from src.sequence_generation import (
    aggregate_chunk_probabilities,
    create_review_chunks,
)
from src.text_preprocessing import VocabularyTokenizer


def test_review_chunk_shape_and_report():
    tokenizer = VocabularyTokenizer(max_words=50).fit(
        ["one two three four five six seven eight nine ten"]
    )
    chunks, report = create_review_chunks(
        tokenizer,
        "one two three four five six seven eight nine ten",
        chunk_length=4,
        stride=3,
        max_chunks=3,
    )
    assert chunks.shape == (3, 4)
    assert report["chunk_count"] == 3
    assert report["token_count"] == 10


def test_chunk_probability_aggregation():
    probabilities = np.array([0.2, 0.4, 0.8])
    review_ids = np.array([0, 0, 1])
    aggregated = aggregate_chunk_probabilities(
        probabilities,
        review_ids,
        review_count=2,
    )
    np.testing.assert_allclose(aggregated, [0.3, 0.8])
