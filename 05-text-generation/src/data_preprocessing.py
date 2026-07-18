"""Corpus loading, normalization, and chronological splitting utilities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusSplit:
    """Chronological train/validation text split."""

    train_text: str
    validation_text: str
    split_index: int


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    preserve_newlines: bool = True,
) -> str:
    """Normalize Unicode and whitespace without discarding useful punctuation.

    Character-level generation benefits from punctuation, capitalization, and
    paragraph boundaries. For that reason, the default pipeline keeps them.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    if preserve_newlines:
        lines = [re.sub(r"[ ]{2,}", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    text = text.strip()
    return text.lower() if lowercase else text


def load_text_corpus(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    lowercase: bool = False,
    preserve_newlines: bool = True,
    max_characters: int | None = None,
) -> str:
    """Load and normalize a plain-text corpus."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("This project expects a plain-text or Markdown corpus.")

    text = path.read_text(encoding=encoding)
    text = normalize_text(
        text,
        lowercase=lowercase,
        preserve_newlines=preserve_newlines,
    )
    if max_characters is not None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")
        text = text[:max_characters]
    if len(text) < 500:
        raise ValueError("Corpus is too small. Provide at least 500 characters.")
    return text


def chronological_text_split(text: str, validation_fraction: float = 0.10) -> CorpusSplit:
    """Split the corpus by position before sequence-window creation.

    Splitting before creating overlapping windows prevents near-duplicate
    windows from leaking across the train and validation sets.
    """
    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")
    split_index = int(len(text) * (1.0 - validation_fraction))
    if split_index <= 0 or split_index >= len(text):
        raise ValueError("Unable to create a valid train/validation split")
    return CorpusSplit(text[:split_index], text[split_index:], split_index)
