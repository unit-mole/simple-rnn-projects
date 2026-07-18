from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import (
    EMBEDDING_MATRIX_PATH,
    METADATA_PATH,
    MODEL_PATH,
    TRAINING_CONFIG_PATH,
    VOCAB_PATH,
)
from .embedding_analysis import nearest_words, sentence_embedding
from .embedding_training import SkipGramEmbeddingModel


@dataclass
class EmbeddingBundle:
    model: SkipGramEmbeddingModel
    word_to_index: dict[str, int]
    index_to_word: dict[int, str]
    embedding_matrix: np.ndarray
    metadata: dict
    training_config: dict

    def nearest(self, word: str, top_k: int = 8):
        return nearest_words(
            word,
            self.word_to_index,
            self.index_to_word,
            self.embedding_matrix,
            top_k=top_k,
        )

    def encode_sentence(self, text: str):
        return sentence_embedding(text, self.word_to_index, self.embedding_matrix)


def load_embedding_bundle(
    model_path: str | Path = MODEL_PATH,
    vocabulary_path: str | Path = VOCAB_PATH,
    embedding_matrix_path: str | Path = EMBEDDING_MATRIX_PATH,
    metadata_path: str | Path = METADATA_PATH,
    training_config_path: str | Path = TRAINING_CONFIG_PATH,
) -> EmbeddingBundle:
    with Path(vocabulary_path).open("r", encoding="utf-8") as handle:
        vocabulary_payload = json.load(handle)
    word_to_index = {
        str(word): int(index)
        for word, index in vocabulary_payload["word_to_index"].items()
    }
    index_to_word = {index: word for word, index in word_to_index.items()}

    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with Path(training_config_path).open("r", encoding="utf-8") as handle:
        training_config = json.load(handle)

    model = SkipGramEmbeddingModel(
        vocabulary_size=int(metadata["vocabulary_size"]),
        embedding_dimension=int(metadata["embedding_dimension"]),
    )
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    embedding_matrix = np.load(embedding_matrix_path).astype(np.float32)

    expected_shape = (
        int(metadata["vocabulary_size"]),
        int(metadata["embedding_dimension"]),
    )
    if embedding_matrix.shape != expected_shape:
        raise ValueError(
            f"Embedding matrix shape {embedding_matrix.shape} does not match "
            f"metadata {expected_shape}."
        )
    return EmbeddingBundle(
        model=model,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        embedding_matrix=embedding_matrix,
        metadata=metadata,
        training_config=training_config,
    )
