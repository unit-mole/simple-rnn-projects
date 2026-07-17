#!/usr/bin/env python
"""Retrain the IMDb Simple RNN and regenerate all portfolio artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_generation import save_portfolio_artifacts
from src.config import (
    CHUNK_LENGTH,
    CHUNK_STRIDE,
    MAX_CHUNKS,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from src.data_preprocessing import clean_review_frame
from src.model_training import train_simple_rnn


def decode_keras_imdb_sequences(sequences) -> list[str]:
    """Decode Keras IMDb integer sequences into readable text."""
    import keras

    word_index = keras.datasets.imdb.get_word_index()
    reverse_index = {
        index + 3: word
        for word, index in word_index.items()
    }
    reverse_index[0] = "<pad>"
    reverse_index[1] = "<start>"
    reverse_index[2] = "<unk>"
    reverse_index[3] = "<unused>"

    return [
        " ".join(
            reverse_index.get(int(token), "<unk>")
            for token in sequence
        )
        for sequence in sequences
    ]


def load_keras_imdb(
    train_limit: int,
    test_limit: int,
    max_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and decode the Keras IMDb training and test partitions."""
    import keras

    (x_train, y_train), (x_test, y_test) = (
        keras.datasets.imdb.load_data(num_words=max_words)
    )

    train_limit = min(train_limit, len(x_train))
    test_limit = min(test_limit, len(x_test))

    train = pd.DataFrame(
        {
            "review": decode_keras_imdb_sequences(
                x_train[:train_limit]
            ),
            "label": np.asarray(
                y_train[:train_limit],
                dtype=int,
            ),
        }
    )
    test = pd.DataFrame(
        {
            "review": decode_keras_imdb_sequences(
                x_test[:test_limit]
            ),
            "label": np.asarray(
                y_test[:test_limit],
                dtype=int,
            ),
        }
    )

    train, _ = clean_review_frame(
        train,
        text_column="review",
        label_column="label",
    )
    test, _ = clean_review_frame(
        test,
        text_column="review",
        label_column="label",
    )
    return train, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-limit",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = load_keras_imdb(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        max_words=args.max_words,
    )

    result = train_simple_rnn(
        train_frame,
        test_frame,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model_path = MODEL_DIR / "imdb_simple_rnn_model.keras"
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    metadata_path = MODEL_DIR / "model_metadata.json"

    result.model.save(model_path)
    result.tokenizer.save(
        tokenizer_path,
        chunk_length=CHUNK_LENGTH,
        chunk_stride=CHUNK_STRIDE,
        max_chunks=MAX_CHUNKS,
        padding="post",
        truncation="overlapping_chunks",
    )

    metadata = {
        "project_name": (
            "IMDb Movie Review Sentiment Analysis using Simple RNN"
        ),
        "artifact_format": "Keras v3 .keras",
        "training_source": "TensorFlow/Keras IMDb dataset",
        "training_pool_reviews": int(len(train_frame)),
        "model_fit_reviews": int(result.model_fit_reviews),
        "validation_reviews": int(result.validation_reviews),
        "test_reviews": int(len(test_frame)),
        "model_selection_split": (
            "80/20 stratified split within the training pool"
        ),
        "post_selection_refit": False,
        "best_epoch": int(result.best_epoch),
        "vocabulary_size": int(
            len(result.tokenizer.word_index)
        ),
        "max_vocabulary": int(args.max_words),
        "chunk_length": CHUNK_LENGTH,
        "chunk_stride": CHUNK_STRIDE,
        "max_chunks_per_review": MAX_CHUNKS,
        "embedding_dimension": 80,
        "simple_rnn_units": 48,
        "dense_units": 32,
        "dropout": 0.30,
        "prediction_aggregation": (
            "mean positive probability across review chunks"
        ),
        "decision_threshold": float(result.threshold),
        "threshold_selection": (
            "F1-maximizing threshold selected on the validation "
            "split before untouched test evaluation"
        ),
        "baseline_comparison_scope": (
            "TF-IDF baseline trained on the same model-fitting "
            "partition as the Simple RNN"
        ),
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "baseline_metrics": result.baseline_metrics,
        "majority_metrics": result.majority_metrics,
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    save_portfolio_artifacts(
        train_frame=train_frame,
        test_frame=test_frame,
        result=result,
        output_dir=OUTPUT_DIR,
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Regenerated outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
