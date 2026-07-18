from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .text_preprocessing import encode_tokens


@dataclass(frozen=True)
class PairSplit:
    train_centers: np.ndarray
    train_contexts: np.ndarray
    validation_centers: np.ndarray
    validation_contexts: np.ndarray


def create_skipgram_pairs(
    tokenized_sentences: Iterable[list[str]],
    word_to_index: dict[str, int],
    window_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create center-context pairs without crossing sentence boundaries."""
    centers: list[int] = []
    contexts: list[int] = []
    for tokens in tokenized_sentences:
        encoded = encode_tokens(tokens, word_to_index)
        for center_position, center_index in enumerate(encoded):
            left = max(0, center_position - window_size)
            right = min(len(encoded), center_position + window_size + 1)
            for context_position in range(left, right):
                if context_position == center_position:
                    continue
                centers.append(center_index)
                contexts.append(encoded[context_position])
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def split_pairs(
    centers: np.ndarray,
    contexts: np.ndarray,
    validation_fraction: float = 0.15,
    random_seed: int = 42,
) -> PairSplit:
    if len(centers) != len(contexts):
        raise ValueError("Center and context arrays must have equal length.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(centers))
    validation_size = max(1, int(round(len(order) * validation_fraction)))
    validation_indices = order[:validation_size]
    train_indices = order[validation_size:]
    return PairSplit(
        train_centers=centers[train_indices],
        train_contexts=contexts[train_indices],
        validation_centers=centers[validation_indices],
        validation_contexts=contexts[validation_indices],
    )
