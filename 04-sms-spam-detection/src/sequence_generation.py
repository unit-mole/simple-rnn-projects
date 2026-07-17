"""Integer sequence creation and padding."""
from __future__ import annotations
import numpy as np
from .text_preprocessing import VocabularyTokenizer

def pad_sequence(token_ids: list[int], max_length: int) -> np.ndarray:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    output = np.zeros(max_length, dtype=np.int32)
    selected = token_ids[:max_length]
    output[:len(selected)] = selected
    return output

def texts_to_padded_sequences(
    tokenizer: VocabularyTokenizer,
    texts: list[object],
    max_length: int,
) -> np.ndarray:
    return np.vstack([
        pad_sequence(tokenizer.encode(text), max_length)
        for text in texts
    ])

def sequence_report(
    tokenizer: VocabularyTokenizer,
    text: object,
    max_length: int,
) -> dict:
    token_ids = tokenizer.encode(text)
    return {
        "token_count": int(len(token_ids)),
        "tokens_used": int(min(len(token_ids), max_length)),
        "truncated_tokens": int(max(0, len(token_ids) - max_length)),
        "out_of_vocabulary_tokens": int(sum(token_id == 1 for token_id in token_ids)),
        "maximum_sequence_length": int(max_length),
    }
