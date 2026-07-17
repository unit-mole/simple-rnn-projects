"""Validate repository artifacts without loading TensorFlow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

LIVE_URL = "https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/"

REQUIRED_FILES = [
    "README.md",
    "README_HOSTING.md",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/sample_reviews.csv",
    "models/imdb_simple_rnn_model.keras",
    "models/tokenizer.json",
    "models/model_metadata.json",
    "outputs/model_metrics.json",
    "outputs/model_comparison.csv",
    "outputs/error_analysis.csv",
    "outputs/confusion_matrix.png",
    "outputs/roc_curve.png",
    "outputs/precision_recall_curve.png",
    "outputs/training_accuracy.png",
    "outputs/training_loss.png",
    "images/01_streamlit_application_overview.png",
    "images/02_single_review_prediction.png",
    "images/03_batch_sentiment_workflow.png",
    "images/04_model_performance_and_error_analysis.png",
]

METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def validate_project(project_root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        path = project_root / relative
        if not path.exists():
            errors.append(f"Missing required file: {relative}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"Empty required file: {relative}")

    readme_path = project_root / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        if "ADD-LIVE-STREAMLIT-URL-HERE" in readme:
            errors.append("README still contains the live-URL placeholder.")
        if LIVE_URL not in readme:
            errors.append("README does not contain the deployed app URL.")

    metadata_path = project_root / "models" / "model_metadata.json"
    metrics_path = project_root / "outputs" / "model_metrics.json"
    comparison_path = (
        project_root / "outputs" / "model_comparison.csv"
    )

    if metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        fit_rows = int(metadata.get("model_fit_reviews", -1))
        validation_rows = int(
            metadata.get("validation_reviews", -1)
        )
        pool_rows = int(
            metadata.get("training_pool_reviews", -1)
        )
        if fit_rows + validation_rows != pool_rows:
            errors.append(
                "Model-fit and validation rows do not equal "
                "the training pool."
            )

    if (
        metadata_path.exists()
        and metrics_path.exists()
    ):
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        metadata_metrics = metadata.get("test_metrics", {})
        for key in METRIC_KEYS:
            left = float(metadata_metrics.get(key, -1))
            right = float(metrics.get(key, -2))
            if abs(left - right) > 1e-10:
                errors.append(
                    f"Metric mismatch for {key}: "
                    f"metadata={left}, output={right}"
                )

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        expected_models = {
            "Majority Class",
            "Simple RNN",
            "TF-IDF + Logistic Regression",
        }
        actual_models = set(comparison["model"].astype(str))
        if actual_models != expected_models:
            errors.append(
                "Model comparison does not contain the expected "
                "three models."
            )

    sample_path = project_root / "data" / "sample_reviews.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        required_columns = {
            "review",
            "illustrative_tone",
        }
        if not required_columns.issubset(sample.columns):
            errors.append(
                "Sample CSV is missing required columns: "
                + ", ".join(sorted(required_columns))
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
