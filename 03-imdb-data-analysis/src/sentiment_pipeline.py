"""Saved-artifact inference pipeline for manual and batch sentiment scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CHUNK_LENGTH,
    CHUNK_STRIDE,
    DECISION_THRESHOLD,
    MAX_CHUNKS,
    METADATA_PATH,
    MODEL_PATH,
    TOKENIZER_PATH,
)
from .sequence_generation import (
    aggregate_chunk_probabilities,
    create_batch_chunks,
)
from .text_preprocessing import VocabularyTokenizer, clean_text


@dataclass
class SentimentArtifacts:
    model: object
    tokenizer: VocabularyTokenizer
    tokenizer_metadata: dict
    model_metadata: dict


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> SentimentArtifacts:
    """Load the saved Keras model, tokenizer, and metadata."""
    import keras

    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    metadata_path = Path(metadata_path)

    for path in (model_path, tokenizer_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact was not found: {path}")

    model = keras.models.load_model(model_path, compile=False)
    tokenizer, tokenizer_metadata = VocabularyTokenizer.load(tokenizer_path)
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return SentimentArtifacts(
        model=model,
        tokenizer=tokenizer,
        tokenizer_metadata=tokenizer_metadata,
        model_metadata=model_metadata,
    )


def confidence_band(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.65:
        return "Moderate"
    return "Low"


def interpretation_text(
    sentiment: str,
    positive_probability: float,
    confidence: float,
    threshold: float,
) -> str:
    direction = (
        "positive sequence patterns"
        if sentiment == "positive"
        else "negative sequence patterns"
    )
    uncertainty = (
        " The score is close to the decision boundary, so the result should be treated as uncertain."
        if confidence < 0.65
        else ""
    )
    return (
        f"The model found stronger {direction} in the processed review chunks. "
        f"The positive-sentiment probability is {positive_probability:.1%}, "
        f"using a validation-selected threshold of {threshold:.2f}."
        f"{uncertainty}"
    )


def predict_reviews(
    texts: list[object],
    artifacts: SentimentArtifacts,
    threshold: float | None = None,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, list[dict]]:
    """Score reviews and return one aggregated prediction per review."""
    if not texts:
        raise ValueError("At least one review is required.")

    threshold = float(
        threshold
        if threshold is not None
        else artifacts.model_metadata.get(
            "decision_threshold",
            DECISION_THRESHOLD,
        )
    )

    chunk_length = int(
        artifacts.tokenizer_metadata.get("chunk_length", CHUNK_LENGTH)
    )
    stride = int(
        artifacts.tokenizer_metadata.get("chunk_stride", CHUNK_STRIDE)
    )
    max_chunks = int(
        artifacts.tokenizer_metadata.get("max_chunks", MAX_CHUNKS)
    )

    chunks, review_ids, reports = create_batch_chunks(
        artifacts.tokenizer,
        texts,
        chunk_length=chunk_length,
        stride=stride,
        max_chunks=max_chunks,
    )
    chunk_probabilities = artifacts.model.predict(
        chunks,
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)
    positive_probabilities = aggregate_chunk_probabilities(
        chunk_probabilities,
        review_ids,
        len(texts),
    )

    labels = (positive_probabilities >= threshold).astype(int)
    sentiments = np.where(labels == 1, "positive", "negative")
    confidences = np.where(
        labels == 1,
        positive_probabilities,
        1.0 - positive_probabilities,
    )

    result = pd.DataFrame(
        {
            "review": [str(text) for text in texts],
            "cleaned_review": [clean_text(text) for text in texts],
            "positive_probability": positive_probabilities,
            "predicted_label": labels,
            "predicted_sentiment": sentiments,
            "confidence": confidences,
            "confidence_band": [
                confidence_band(float(value))
                for value in confidences
            ],
        }
    )
    result["interpretation"] = [
        interpretation_text(
            str(sentiment),
            float(probability),
            float(confidence),
            threshold,
        )
        for sentiment, probability, confidence in zip(
            sentiments,
            positive_probabilities,
            confidences,
        )
    ]
    return result, reports
