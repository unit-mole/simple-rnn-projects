from __future__ import annotations

import html
import re
from collections import Counter
from typing import Iterable

from .config import PAD_TOKEN, UNK_TOKEN

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(text: object) -> str:
    """Normalize text without introducing private-data placeholders into the vocabulary."""
    value = html.unescape(str(text or "")).lower()
    value = _HTML_TAG_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = re.sub(r"[^a-z'\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: object) -> list[str]:
    """Return lowercase alphabetic tokens after deterministic cleaning."""
    return _TOKEN_RE.findall(clean_text(text))


def build_vocabulary(
    tokenized_sentences: Iterable[list[str]],
    min_frequency: int = 2,
    max_vocabulary_size: int | None = 500,
) -> tuple[dict[str, int], dict[int, str], Counter[str]]:
    """Create a frequency-sorted vocabulary with stable PAD and UNK indices."""
    counts: Counter[str] = Counter(
        token for sentence in tokenized_sentences for token in sentence
    )
    ordered = sorted(
        (
            (token, frequency)
            for token, frequency in counts.items()
            if frequency >= min_frequency
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if max_vocabulary_size is not None:
        ordered = ordered[: max(0, max_vocabulary_size - 2)]

    word_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in ordered:
        word_to_index[token] = len(word_to_index)
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word, counts


def encode_tokens(tokens: Iterable[str], word_to_index: dict[str, int]) -> list[int]:
    """Encode tokens with a deterministic out-of-vocabulary fallback."""
    unknown_index = word_to_index[UNK_TOKEN]
    return [word_to_index.get(token, unknown_index) for token in tokens]


def decode_indices(indices: Iterable[int], index_to_word: dict[int, str]) -> list[str]:
    return [index_to_word.get(int(index), UNK_TOKEN) for index in indices]
