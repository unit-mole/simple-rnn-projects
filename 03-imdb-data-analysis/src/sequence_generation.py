"""Review chunking, padding, and probability aggregation."""

from __future__ import annotations

import numpy as np

from .text_preprocessing import VocabularyTokenizer


def create_review_chunks(
    tokenizer: VocabularyTokenizer,
    text: object,
    chunk_length: int = 80,
    stride: int = 60,
    max_chunks: int = 6,
) -> tuple[np.ndarray, dict]:
    """Convert one review into padded, overlapping integer-token chunks."""
    if chunk_length <= 0 or stride <= 0 or max_chunks <= 0:
        raise ValueError("chunk_length, stride, and max_chunks must be positive.")

    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = [1]

    starts = list(range(0, max(1, len(token_ids) - chunk_length + 1), stride))
    final_start = max(0, len(token_ids) - chunk_length)
    if final_start not in starts:
        starts.append(final_start)
    starts = sorted(set(starts))

    if len(starts) > max_chunks:
        selected = np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)
        starts = [starts[index] for index in selected]

    chunks = np.zeros((len(starts), chunk_length), dtype=np.int32)
    for row, start in enumerate(starts):
        sequence = token_ids[start : start + chunk_length]
        chunks[row, : len(sequence)] = sequence

    report = {
        "token_count": int(len(token_ids)),
        "chunk_count": int(len(chunks)),
        "chunk_length": int(chunk_length),
        "stride": int(stride),
        "max_chunks": int(max_chunks),
    }
    return chunks, report


def create_batch_chunks(
    tokenizer: VocabularyTokenizer,
    texts: list[object],
    chunk_length: int = 80,
    stride: int = 60,
    max_chunks: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create chunks for multiple reviews and retain review-to-chunk mapping."""
    arrays: list[np.ndarray] = []
    review_ids: list[int] = []
    reports: list[dict] = []

    for review_id, text in enumerate(texts):
        chunks, report = create_review_chunks(
            tokenizer,
            text,
            chunk_length=chunk_length,
            stride=stride,
            max_chunks=max_chunks,
        )
        arrays.append(chunks)
        review_ids.extend([review_id] * len(chunks))
        reports.append(report)

    return (
        np.concatenate(arrays, axis=0),
        np.asarray(review_ids, dtype=np.int32),
        reports,
    )


def aggregate_chunk_probabilities(
    chunk_probabilities: np.ndarray,
    review_ids: np.ndarray,
    review_count: int,
) -> np.ndarray:
    """Average chunk-level probabilities into one probability per review."""
    probabilities = np.asarray(chunk_probabilities, dtype=float).reshape(-1)
    review_ids = np.asarray(review_ids, dtype=np.int32).reshape(-1)

    if len(probabilities) != len(review_ids):
        raise ValueError("Probability and review-id arrays must have equal length.")
    if review_count <= 0:
        raise ValueError("review_count must be positive.")

    sums = np.bincount(review_ids, weights=probabilities, minlength=review_count)
    counts = np.bincount(review_ids, minlength=review_count)

    if np.any(counts == 0):
        raise ValueError("Every review must have at least one chunk.")

    return sums / counts
