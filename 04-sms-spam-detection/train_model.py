#!/usr/bin/env python
"""Retrain the SMS Simple RNN and regenerate the complete model release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_generation import (
    save_portfolio_artifacts,
    write_model_card,
)
from src.config import (
    DATA_SOURCE_URL,
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIMENSION,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCABULARY,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    RNN_UNITS,
)
from src.data_preprocessing import (
    clean_sms_frame,
    load_tab_separated_sms,
)
from src.model_training import train_sms_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="",
        help=(
            "Optional local CSV or TSV path. When omitted, "
            "the public source used by the original notebook "
            "is downloaded."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    return parser.parse_args()


def load_frame(
    path: str,
) -> tuple[pd.DataFrame, dict, str]:
    if not path:
        frame, report = load_tab_separated_sms(
            DATA_SOURCE_URL
        )
        return frame, report, DATA_SOURCE_URL

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.suffix.lower() in {
        ".tsv",
        ".txt",
    }:
        frame, report = load_tab_separated_sms(
            input_path
        )
    else:
        frame, report = clean_sms_frame(
            pd.read_csv(input_path)
        )

    return (
        frame,
        report,
        str(input_path.resolve()),
    )


def main() -> None:
    import keras

    args = parse_args()
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame, quality_report, data_source = (
        load_frame(args.data)
    )
    result = train_sms_models(
        frame,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model_path = (
        MODEL_DIR
        / "sms_spam_simple_rnn_model.keras"
    )
    tokenizer_path = (
        MODEL_DIR / "tokenizer.json"
    )
    metadata_path = (
        MODEL_DIR / "model_metadata.json"
    )

    result.model.save(model_path)
    result.tokenizer.save(
        tokenizer_path,
        max_sequence_length=(
            MAX_SEQUENCE_LENGTH
        ),
        padding="post",
        truncation="post",
        cleaning={
            "lowercase": True,
            "url_token": "<url>",
            "phone_token": "<phone>",
            "currency_token": "<currency>",
            "number_token": "<number>",
            "preserved_symbols": [
                "!",
                "?",
                "%",
            ],
        },
    )

    dataset_summary = save_portfolio_artifacts(
        frame=frame,
        result=result,
        output_dir=OUTPUT_DIR,
        quality_report=quality_report,
    )

    metadata = {
        "project_name": (
            "SMS Spam Detection using Simple RNN"
        ),
        "artifact_format": "Keras v3 .keras",
        "keras_version": keras.__version__,
        "training_backend": (
            keras.backend.backend()
        ),
        "data_source": data_source,
        **dataset_summary,
        "maximum_vocabulary": (
            MAX_VOCABULARY
        ),
        "learned_vocabulary_size": int(
            len(result.tokenizer.word_index)
        ),
        "maximum_sequence_length": (
            MAX_SEQUENCE_LENGTH
        ),
        "embedding_dimension": (
            EMBEDDING_DIMENSION
        ),
        "simple_rnn_units": RNN_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "class_weights": (
            result.class_weights
        ),
        "decision_threshold": (
            result.threshold
        ),
        "threshold_selection": (
            "Maximum validation F1 with precision "
            "and balanced accuracy as tie-breakers"
        ),
        "model_selection": (
            "Early stopping on validation loss; "
            "the final test set was not used for "
            "training or threshold selection"
        ),
        "training_epochs_completed": int(
            len(result.history)
        ),
        "best_epoch": result.best_epoch,
        "training_seconds": (
            result.training_seconds
        ),
        "test_metrics": (
            result.test_metrics
        ),
        "baseline_metrics": {
            "tfidf_logistic_regression": (
                result.logistic_metrics
            ),
            "multinomial_naive_bayes": (
                result.naive_bayes_metrics
            ),
            "majority_class": (
                result.majority_metrics
            ),
        },
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_model_card(
        metadata,
        MODEL_DIR / "MODEL_CARD.md",
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved tokenizer: {tokenizer_path}")
    print(
        "Regenerated tables, charts, error analysis, "
        "dataset summary, and model card."
    )


if __name__ == "__main__":
    main()
