#!/usr/bin/env python
"""Retrain the IMDb Simple RNN and regenerate portfolio artifacts."""

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

from src.config import MODEL_DIR, OUTPUT_DIR
from src.data_preprocessing import clean_review_frame
from src.model_evaluation import classify_probabilities, classification_report_frame
from src.model_training import train_simple_rnn


def decode_keras_imdb_sequences(sequences, max_words: int) -> list[str]:
    import keras

    word_index = keras.datasets.imdb.get_word_index()
    reverse_index = {index + 3: word for word, index in word_index.items()}
    reverse_index[0] = "<pad>"
    reverse_index[1] = "<start>"
    reverse_index[2] = "<unk>"
    reverse_index[3] = "<unused>"

    return [
        " ".join(reverse_index.get(int(token), "<unk>") for token in sequence)
        for sequence in sequences
    ]


def load_keras_imdb(train_limit: int, test_limit: int, max_words: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_words
    )

    train_limit = min(train_limit, len(x_train))
    test_limit = min(test_limit, len(x_test))

    train = pd.DataFrame(
        {
            "review": decode_keras_imdb_sequences(
                x_train[:train_limit],
                max_words,
            ),
            "label": np.asarray(y_train[:train_limit], dtype=int),
        }
    )
    test = pd.DataFrame(
        {
            "review": decode_keras_imdb_sequences(
                x_test[:test_limit],
                max_words,
            ),
            "label": np.asarray(y_test[:test_limit], dtype=int),
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
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=5_000)
    parser.add_argument("--max-words", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
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
        chunk_length=80,
        chunk_stride=60,
        max_chunks=6,
        padding="post",
        truncation="overlapping_chunks",
    )

    metadata = {
        "project_name": "IMDb Movie Review Sentiment Analysis using Simple RNN",
        "training_reviews": int(len(train_frame)),
        "test_reviews": int(len(test_frame)),
        "decision_threshold": float(result.threshold),
        "test_metrics": result.test_metrics,
        "baseline_metrics": result.baseline_metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    predictions = classify_probabilities(
        result.test_probabilities,
        result.threshold,
    )
    prediction_frame = pd.DataFrame(
        {
            "review_id": [
                f"test_{index + 1:05d}"
                for index in range(len(test_frame))
            ],
            "true_label": test_frame["label"].to_numpy(),
            "positive_probability": result.test_probabilities,
            "predicted_label": predictions,
        }
    )
    prediction_frame.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    classification_report_frame(
        test_frame["label"].to_numpy(),
        predictions,
    ).to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=False,
    )
    result.history.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )
    result.threshold_table.to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            {"model": "Simple RNN", **result.test_metrics},
            {
                "model": "TF-IDF + Logistic Regression",
                **result.baseline_metrics,
            },
        ]
    )
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
