import numpy as np

from src.sequence_generation import create_skipgram_pairs, split_pairs
from src.text_preprocessing import build_vocabulary


def test_pairs_do_not_cross_sentence_boundaries():
    sentences = [["alpha", "beta"], ["gamma", "delta"]]
    word_to_index, _, _ = build_vocabulary(
        sentences, min_frequency=1, max_vocabulary_size=20
    )
    centers, contexts = create_skipgram_pairs(sentences, word_to_index, window_size=1)
    pairs = set(zip(centers.tolist(), contexts.tolist()))
    assert (word_to_index["beta"], word_to_index["gamma"]) not in pairs
    assert (word_to_index["gamma"], word_to_index["beta"]) not in pairs


def test_pair_split_is_complete_and_disjoint_by_position():
    centers = np.arange(100, dtype=np.int64)
    contexts = np.arange(100, dtype=np.int64)[::-1]
    split = split_pairs(centers, contexts, validation_fraction=0.2, random_seed=42)
    assert len(split.train_centers) == 80
    assert len(split.validation_centers) == 20
    assert set(split.train_centers).isdisjoint(set(split.validation_centers))
