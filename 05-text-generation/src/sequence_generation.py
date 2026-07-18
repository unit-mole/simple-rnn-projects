"""Integer-encoded sequence-window generation for next-character prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_preprocessing import chronological_text_split
from .text_preprocessing import CharacterVocabulary


@dataclass(frozen=True)
class SequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    vocabulary: CharacterVocabulary
    train_characters: int
    validation_characters: int


def create_input_target_pairs(
    encoded_text: np.ndarray,
    *,
    sequence_length: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length input windows and next-character targets."""
    encoded_text = np.asarray(encoded_text, dtype=np.int32)
    if sequence_length < 5:
        raise ValueError("sequence_length must be at least 5")
    if step < 1:
        raise ValueError("step must be at least 1")
    if encoded_text.size <= sequence_length:
        raise ValueError("Text is shorter than the requested sequence length")

    starts = np.arange(0, encoded_text.size - sequence_length, step, dtype=np.int32)
    X = np.stack([encoded_text[start : start + sequence_length] for start in starts])
    y = encoded_text[starts + sequence_length]
    return X.astype(np.int32), y.astype(np.int32)


def build_train_validation_sequences(
    text: str,
    *,
    sequence_length: int = 80,
    step: int = 3,
    validation_fraction: float = 0.10,
) -> SequenceDataset:
    """Split text first, fit vocabulary on training text, then create windows."""
    split = chronological_text_split(text, validation_fraction=validation_fraction)
    vocabulary = CharacterVocabulary.fit(split.train_text)

    train_encoded = vocabulary.encode(split.train_text)
    validation_encoded = vocabulary.encode(split.validation_text)

    X_train, y_train = create_input_target_pairs(
        train_encoded,
        sequence_length=sequence_length,
        step=step,
    )
    X_validation, y_validation = create_input_target_pairs(
        validation_encoded,
        sequence_length=sequence_length,
        step=max(1, step),
    )

    return SequenceDataset(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        vocabulary=vocabulary,
        train_characters=len(split.train_text),
        validation_characters=len(split.validation_text),
    )
