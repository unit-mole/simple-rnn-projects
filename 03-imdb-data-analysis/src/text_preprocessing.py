"""Text cleaning and deterministic vocabulary management."""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import re
from typing import Iterable


TOKEN_PATTERN = re.compile(r"<unk>|[a-z0-9]+(?:'[a-z]+)?|[!?]")


def clean_text(text: object) -> str:
    """Clean IMDb-style text while retaining useful sentiment signals."""
    value = html.unescape("" if text is None else str(text))
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bSTART\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bUNK\b", " <unk> ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bbr\b", " ", value, flags=re.IGNORECASE)
    value = value.lower()
    value = re.sub(r"[^a-z0-9'!?.,;:\-\s<>]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: object) -> list[str]:
    """Tokenize cleaned text without removing negations or sentiment words."""
    return TOKEN_PATTERN.findall(clean_text(text))


class VocabularyTokenizer:
    """Small JSON-serializable tokenizer used consistently in training and inference."""

    def __init__(self, max_words: int = 10_000) -> None:
        if max_words < 3:
            raise ValueError("max_words must be at least 3.")
        self.max_words = int(max_words)
        self.word_index: dict[str, int] = {"<pad>": 0, "<oov>": 1}

    def fit(self, texts: Iterable[object]) -> "VocabularyTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        self.word_index = {"<pad>": 0, "<oov>": 1}
        for word, _ in counter.most_common(self.max_words - 2):
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)
        return self

    def encode(self, text: object) -> list[int]:
        return [self.word_index.get(token, 1) for token in tokenize(text)]

    def to_dict(self) -> dict:
        return {
            "max_words": self.max_words,
            "word_index": self.word_index,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VocabularyTokenizer":
        tokenizer = cls(max_words=int(payload.get("max_words", 10_000)))
        tokenizer.word_index = {
            str(word): int(index)
            for word, index in payload["word_index"].items()
        }
        return tokenizer

    def save(self, path: str | Path, **extra: object) -> None:
        payload = self.to_dict()
        payload.update(extra)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["VocabularyTokenizer", dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload), payload
