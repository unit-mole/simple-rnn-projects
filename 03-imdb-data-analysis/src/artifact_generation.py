"""Generate reproducible evaluation tables and portfolio visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .model_evaluation import classify_probabilities
from .model_training import TrainingResult
from .text_preprocessing import clean_text


def _metric_row(
    model_name: str,
    threshold: float,
    metrics: dict,
) -> dict:
    return {
        "model": model_name,
        "threshold": float(threshold),
        **metrics,
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    plt.figure(figsize=(6.4, 5.2))
    image = plt.imshow(matrix)
    plt.title("Simple RNN Confusion Matrix")
    plt.xticks(
        [0, 1],
        ["Predicted negative", "Predicted positive"],
        rotation=20,
    )
    plt.yticks(
        [0, 1],
        ["Actual negative", "Actual positive"],
    )
    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=14,
            )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.colorbar(image)
    _save_figure(output_path)


def _save_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probabilities,
    )
    plt.figure(figsize=(6.8, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Simple RNN (AUC = {roc_auc:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.title("Simple RNN ROC Curve")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend()
    _save_figure(output_path)


def _save_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    plt.figure(figsize=(6.8, 5))
    plt.plot(
        recall,
        precision,
        label=f"Simple RNN (PR-AUC = {pr_auc:.3f})",
    )
    plt.title("Simple RNN Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    _save_figure(output_path)


def _save_training_curves(
    history: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7.5, 4.7))
    plt.plot(
        history["epoch"],
        history["accuracy"],
        marker="o",
        label="Training chunk accuracy",
    )
    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        marker="o",
        label="Validation chunk accuracy",
    )
    plt.title("Simple RNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Chunk-level accuracy")
    plt.legend()
    _save_figure(output_dir / "training_accuracy.png")

    plt.figure(figsize=(7.5, 4.7))
    plt.plot(
        history["epoch"],
        history["loss"],
        marker="o",
        label="Training loss",
    )
    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation loss",
    )
    plt.title("Simple RNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    _save_figure(output_dir / "training_loss.png")


def _save_eda(
    train_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    working = train_frame.copy()
    working["sentiment"] = working["label"].map(
        {0: "negative", 1: "positive"}
    )
    working["review_length"] = (
        working["review"].map(clean_text).str.split().str.len()
    )

    counts = (
        working["sentiment"]
        .value_counts()
        .reindex(["negative", "positive"], fill_value=0)
    )
    plt.figure(figsize=(7, 4.5))
    plt.bar(counts.index, counts.values)
    plt.title("IMDb Training Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Reviews")
    for index, value in enumerate(counts.values):
        plt.text(
            index,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )
    _save_figure(output_dir / "sentiment_distribution.png")

    plt.figure(figsize=(8, 4.8))
    plt.hist(working["review_length"], bins=35)
    plt.title("IMDb Review Length Distribution")
    plt.xlabel("Review length (words)")
    plt.ylabel("Reviews")
    _save_figure(output_dir / "review_length_distribution.png")

    means = (
        working.groupby("sentiment")["review_length"]
        .mean()
        .reindex(["negative", "positive"])
    )
    plt.figure(figsize=(7, 4.5))
    plt.bar(means.index, means.values)
    plt.title("Average Review Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average words")
    for index, value in enumerate(means.values):
        plt.text(
            index,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )
    _save_figure(output_dir / "average_review_length.png")


def _save_baseline_terms(
    result: TrainingResult,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    terms = result.baseline_vectorizer.get_feature_names_out()
    weights = result.baseline_classifier.coef_[0]

    positive_indices = np.argsort(weights)[-top_n:][::-1]
    negative_indices = np.argsort(weights)[:top_n]

    positive_terms = terms[positive_indices]
    positive_weights = weights[positive_indices]
    negative_terms = terms[negative_indices]
    negative_weights = weights[negative_indices]

    term_table = pd.DataFrame(
        {
            "positive_term": positive_terms,
            "positive_weight": positive_weights,
            "negative_term": negative_terms,
            "negative_weight": negative_weights,
        }
    )
    term_table.to_csv(
        output_dir / "top_terms.csv",
        index=False,
    )

    positive_plot = term_table.head(15).sort_values(
        "positive_weight"
    )
    plt.figure(figsize=(8, 5.8))
    plt.barh(
        positive_plot["positive_term"],
        positive_plot["positive_weight"],
    )
    plt.title("Terms Most Associated with Positive Sentiment")
    plt.xlabel("TF-IDF logistic-regression coefficient")
    _save_figure(output_dir / "top_positive_terms.png")

    negative_plot = term_table.head(15).copy()
    negative_plot["magnitude"] = (
        negative_plot["negative_weight"].abs()
    )
    negative_plot = negative_plot.sort_values("magnitude")
    plt.figure(figsize=(8, 5.8))
    plt.barh(
        negative_plot["negative_term"],
        negative_plot["magnitude"],
    )
    plt.title("Terms Most Associated with Negative Sentiment")
    plt.xlabel("Absolute TF-IDF logistic-regression coefficient")
    _save_figure(output_dir / "top_negative_terms.png")


def save_portfolio_artifacts(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    result: TrainingResult,
    output_dir: str | Path,
) -> None:
    """Regenerate all tables and plots referenced by the README and app."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = test_frame["label"].to_numpy(dtype=int)
    rnn_predictions = classify_probabilities(
        result.test_probabilities,
        result.threshold,
    )
    baseline_predictions = classify_probabilities(
        result.baseline_probabilities,
        0.5,
    )
    majority_predictions = classify_probabilities(
        result.majority_probabilities,
        0.5,
    )

    result.history.to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    result.threshold_table.to_csv(
        output_dir / "threshold_analysis.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame(
        {
            "review_id": [
                f"test_{index + 1:05d}"
                for index in range(len(test_frame))
            ],
            "true_label": y_true,
            "positive_probability": result.test_probabilities,
            "predicted_label": rnn_predictions,
            "predicted_sentiment": np.where(
                rnn_predictions == 1,
                "positive",
                "negative",
            ),
            "is_correct": rnn_predictions == y_true,
        }
    )
    prediction_frame.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    report = classification_report(
        y_true,
        rnn_predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    (
        pd.DataFrame(report)
        .transpose()
        .reset_index(names="class")
        .to_csv(
            output_dir / "classification_report.csv",
            index=False,
        )
    )

    comparison = pd.DataFrame(
        [
            _metric_row(
                "Majority Class",
                0.5,
                result.majority_metrics,
            ),
            _metric_row(
                "Simple RNN",
                result.threshold,
                result.test_metrics,
            ),
            _metric_row(
                "TF-IDF + Logistic Regression",
                0.5,
                result.baseline_metrics,
            ),
        ]
    )
    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    (output_dir / "model_metrics.json").write_text(
        json.dumps(
            {
                "model": "Simple RNN",
                "threshold": float(result.threshold),
                **result.test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    error_frame = test_frame.loc[
        rnn_predictions != y_true,
        ["review", "label"],
    ].copy()
    error_indices = np.flatnonzero(rnn_predictions != y_true)
    error_frame["positive_probability"] = (
        result.test_probabilities[error_indices]
    )
    error_frame["predicted_label"] = (
        rnn_predictions[error_indices]
    )
    error_frame["error_type"] = np.where(
        (error_frame["label"] == 0)
        & (error_frame["predicted_label"] == 1),
        "false_positive",
        "false_negative",
    )
    error_frame["review_excerpt"] = (
        error_frame["review"]
        .map(clean_text)
        .str.slice(0, 300)
    )
    error_frame["text_length"] = (
        error_frame["review"].map(clean_text).str.split().str.len()
    )
    error_frame[
        [
            "error_type",
            "label",
            "predicted_label",
            "positive_probability",
            "text_length",
            "review_excerpt",
        ]
    ].head(80).to_csv(
        output_dir / "error_analysis.csv",
        index=False,
    )

    _save_eda(train_frame, output_dir)
    _save_baseline_terms(result, output_dir)

    _save_confusion_matrix(
        y_true,
        rnn_predictions,
        output_dir / "confusion_matrix.png",
    )
    _save_roc_curve(
        y_true,
        result.test_probabilities,
        result.test_metrics["roc_auc"],
        output_dir / "roc_curve.png",
    )
    _save_precision_recall_curve(
        y_true,
        result.test_probabilities,
        result.test_metrics["pr_auc"],
        output_dir / "precision_recall_curve.png",
    )
    _save_training_curves(
        result.history,
        output_dir,
    )

    plt.figure(figsize=(8, 4.8))
    plt.bar(
        comparison["model"],
        comparison["accuracy"],
    )
    plt.title("IMDb Sentiment Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=12)
    for index, value in enumerate(comparison["accuracy"]):
        plt.text(
            index,
            value,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )
    _save_figure(output_dir / "baseline_comparison.png")

    plt.figure(figsize=(8, 4.8))
    plt.hist(result.test_probabilities, bins=25)
    plt.axvline(
        result.threshold,
        linestyle="--",
        label=f"Decision threshold = {result.threshold:.2f}",
    )
    plt.title(
        "Simple RNN Positive-Sentiment Probability Distribution"
    )
    plt.xlabel("Predicted positive probability")
    plt.ylabel("Reviews")
    plt.legend()
    _save_figure(output_dir / "probability_distribution.png")
