"""Validate repository and deployment artifacts without loading TensorFlow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
LIVE_APP_URL = "https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/"

REQUIRED_FILES = [
    "README.md",
    "README_HOSTING.md",
    "PROJECT_REVIEW.md",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/README_data.md",
    "data/sample_sms_messages.csv",
    "models/sms_spam_simple_rnn_model.keras",
    "models/tokenizer.json",
    "models/model_metadata.json",
    "models/MODEL_CARD.md",
    "outputs/model_metrics.json",
    "outputs/model_comparison.csv",
    "outputs/test_predictions.csv",
    "outputs/classification_report.csv",
    "outputs/error_analysis.csv",
    "outputs/dataset_summary.json",
    "outputs/training_history.csv",
    "outputs/threshold_analysis.csv",
    "outputs/top_terms.csv",
    "outputs/class_distribution.png",
    "outputs/message_length_distribution.png",
    "outputs/average_message_length.png",
    "outputs/top_spam_terms.png",
    "outputs/top_ham_terms.png",
    "outputs/confusion_matrix.png",
    "outputs/roc_curve.png",
    "outputs/precision_recall_curve.png",
    "outputs/threshold_analysis.png",
    "outputs/training_accuracy.png",
    "outputs/training_loss.png",
    "outputs/model_comparison.png",
    "outputs/probability_distribution.png",
    "images/01_streamlit_application_overview.png",
    "images/02_single_message_prediction.png",
    "images/03_batch_spam_detection_workflow.png",
    "images/04_model_performance_and_error_analysis.png",
    "src/artifact_generation.py",
    "runtime_smoke_test.py",
]

EXPECTED_MODELS = {
    "Majority Class",
    "Multinomial Naive Bayes",
    "TF-IDF + Logistic Regression",
    "Simple RNN",
}

METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def _read_json(
    path: Path,
    errors: list[str],
) -> dict:
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        errors.append(
            f"Invalid JSON in {path.name}: {exc}"
        )
        return {}


def validate_project(
    root: Path = PROJECT_ROOT,
) -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.exists():
            errors.append(
                f"Missing required file: {relative}"
            )
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(
                f"Empty required file: {relative}"
            )

    model_path = (
        root
        / "models"
        / "sms_spam_simple_rnn_model.keras"
    )
    tokenizer_path = (
        root / "models" / "tokenizer.json"
    )
    if model_path.exists() and model_path.stat().st_size < 100_000:
        errors.append(
            "Saved model is unexpectedly small; "
            "check for an incomplete file or Git LFS pointer."
        )
    if tokenizer_path.exists() and tokenizer_path.stat().st_size < 1_000:
        errors.append(
            "Tokenizer is unexpectedly small or incomplete."
        )

    for relative in [
        "README.md",
        "README_HOSTING.md",
    ]:
        path = root / relative
        if path.exists():
            content = path.read_text(
                encoding="utf-8"
            )
            if LIVE_APP_URL not in content:
                errors.append(
                    f"{relative} does not contain the live app URL."
                )
            if "ADD-LIVE-STREAMLIT-URL-HERE" in content:
                errors.append(
                    f"{relative} still contains a URL placeholder."
                )

    metadata_path = (
        root / "models" / "model_metadata.json"
    )
    metrics_path = (
        root / "outputs" / "model_metrics.json"
    )
    summary_path = (
        root / "outputs" / "dataset_summary.json"
    )
    comparison_path = (
        root / "outputs" / "model_comparison.csv"
    )
    predictions_path = (
        root / "outputs" / "test_predictions.csv"
    )
    sample_path = (
        root / "data" / "sample_sms_messages.csv"
    )

    metadata = (
        _read_json(metadata_path, errors)
        if metadata_path.exists()
        else {}
    )
    metrics = (
        _read_json(metrics_path, errors)
        if metrics_path.exists()
        else {}
    )
    summary = (
        _read_json(summary_path, errors)
        if summary_path.exists()
        else {}
    )

    if metadata:
        split_total = sum(
            int(metadata.get(key, -1))
            for key in [
                "train_rows",
                "validation_rows",
                "test_rows",
            ]
        )
        if split_total != int(
            metadata.get("modeling_rows", -2)
        ):
            errors.append(
                "Split rows do not equal modeling rows."
            )

        for key in [
            "train_validation_overlap",
            "train_test_overlap",
            "validation_test_overlap",
        ]:
            if int(metadata.get(key, -1)) != 0:
                errors.append(
                    f"{key} is not zero."
                )

        threshold = float(
            metadata.get(
                "decision_threshold",
                -1,
            )
        )
        if not 0.0 < threshold < 1.0:
            errors.append(
                "Decision threshold must be between 0 and 1."
            )

        class_weights = metadata.get(
            "class_weights",
            {},
        )
        if set(map(str, class_weights.keys())) != {"0", "1"}:
            errors.append(
                "Metadata must contain class weights for labels 0 and 1."
            )

    if metadata and metrics:
        test_metrics = metadata.get(
            "test_metrics",
            {},
        )
        for key in METRIC_KEYS:
            left = float(
                test_metrics.get(key, -1)
            )
            right = float(
                metrics.get(key, -2)
            )
            if abs(left - right) > 1e-10:
                errors.append(
                    f"Metric mismatch for {key}."
                )

    if metadata and summary:
        for key in [
            "modeling_rows",
            "train_rows",
            "validation_rows",
            "test_rows",
            "ham_rows",
            "spam_rows",
        ]:
            if int(metadata.get(key, -1)) != int(
                summary.get(key, -2)
            ):
                errors.append(
                    f"Dataset-summary mismatch for {key}."
                )

    if comparison_path.exists():
        comparison = pd.read_csv(
            comparison_path
        )
        if set(
            comparison["model"].astype(str)
        ) != EXPECTED_MODELS:
            errors.append(
                "Unexpected model comparison rows."
            )
        for key in METRIC_KEYS:
            if key not in comparison.columns:
                errors.append(
                    f"Model comparison is missing {key}."
                )
            elif not comparison[key].between(
                0.0,
                1.0,
                inclusive="both",
            ).all():
                errors.append(
                    f"Model comparison {key} values are outside [0, 1]."
                )

    if predictions_path.exists() and metadata:
        predictions = pd.read_csv(
            predictions_path
        )
        if len(predictions) != int(
            metadata.get("test_rows", -1)
        ):
            errors.append(
                "Prediction rows do not equal metadata test rows."
            )
        required_prediction_columns = {
            "message_id",
            "true_label",
            "spam_probability",
            "predicted_label",
        }
        if not required_prediction_columns.issubset(
            predictions.columns
        ):
            errors.append(
                "Test predictions are missing required columns."
            )
        if "message" in predictions.columns:
            errors.append(
                "Test predictions should not publish full SMS text."
            )
        if (
            "message_id" in predictions.columns
            and not predictions["message_id"].is_unique
        ):
            errors.append(
                "Test prediction message IDs are not unique."
            )

    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        required_sample_columns = {
            "message",
            "illustrative_category",
        }
        if not required_sample_columns.issubset(
            sample.columns
        ):
            errors.append(
                "Sample CSV schema is invalid."
            )
        elif not sample["message"].astype(
            str
        ).str.strip().ne("").all():
            errors.append(
                "Sample CSV contains a blank message."
            )

    if tokenizer_path.exists() and metadata:
        tokenizer = _read_json(
            tokenizer_path,
            errors,
        )
        if int(
            tokenizer.get(
                "max_sequence_length",
                -1,
            )
        ) != int(
            metadata.get(
                "maximum_sequence_length",
                -2,
            )
        ):
            errors.append(
                "Tokenizer and metadata sequence lengths differ."
            )

    return errors


def main() -> int:
    errors = validate_project()
    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
