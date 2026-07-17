"""Leakage-aware chronological splitting and sequence-window generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: int
    validation_end: int
    total_rows: int


def chronological_boundaries(
    n_rows: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> SplitBoundaries:
    if n_rows < 10:
        raise ValueError("At least 10 rows are required.")
    train_end = int(n_rows * train_ratio)
    validation_end = int(n_rows * (train_ratio + validation_ratio))
    if not 0 < train_end < validation_end < n_rows:
        raise ValueError("Invalid chronological split ratios.")
    return SplitBoundaries(train_end, validation_end, n_rows)


def create_partitioned_sequences(
    feature_matrix: np.ndarray,
    scaled_target: np.ndarray,
    lookback: int,
    boundaries: SplitBoundaries,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create one-step-ahead windows and assign them by target timestamp.

    Validation and test sequences may use earlier historical context, which is
    available at prediction time, but their targets never leak into training.
    """
    features = np.asarray(feature_matrix, dtype=np.float32)
    target = np.asarray(scaled_target, dtype=np.float32).reshape(-1)
    if len(features) != len(target):
        raise ValueError("Feature and target lengths do not match.")
    if lookback < 2 or len(features) <= lookback:
        raise ValueError("Lookback is too large for the available data.")

    partitions: dict[str, list[list[np.ndarray] | list[int]]] = {
        "train": [[], [], []],
        "validation": [[], [], []],
        "test": [[], [], []],
    }
    for target_index in range(lookback, len(features)):
        if target_index < boundaries.train_end:
            name = "train"
        elif target_index < boundaries.validation_end:
            name = "validation"
        else:
            name = "test"
        partitions[name][0].append(features[target_index - lookback : target_index])
        partitions[name][1].append(target[target_index])
        partitions[name][2].append(target_index)

    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (x_values, y_values, indices) in partitions.items():
        output[name] = (
            np.asarray(x_values, dtype=np.float32),
            np.asarray(y_values, dtype=np.float32),
            np.asarray(indices, dtype=int),
        )
    return output
