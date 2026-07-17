import numpy as np

from src.sequence_generation import chronological_boundaries, create_partitioned_sequences


def test_sequence_targets_follow_chronological_boundaries():
    features = np.arange(100, dtype=float).reshape(20, 5)
    target = np.arange(20, dtype=float)
    boundaries = chronological_boundaries(20, 0.6, 0.2)
    partitions = create_partitioned_sequences(features, target, 4, boundaries)
    assert partitions["train"][2].max() < boundaries.train_end
    assert partitions["validation"][2].min() >= boundaries.train_end
    assert partitions["test"][2].min() >= boundaries.validation_end
    assert partitions["train"][0].shape[1:] == (4, 5)
