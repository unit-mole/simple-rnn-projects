"""Generate every published evaluation table and visualization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .model_evaluation import (
    classification_report_frame,
    classify_probabilities,
)
from .model_training import TrainingResult
from .text_preprocessing import clean_text


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def _metric_row(
    model_name: str,
    metrics: dict,
) -> dict:
    return {
        "model": model_name,
        **metrics,
    }


def _save_eda(
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    working = frame.copy()
    working["class_name"] = working["label"].map(
        {0: "ham", 1: "spam"}
    )
    working["message_length_words"] = (
        working["clean_message"]
        .str.split()
        .str.len()
    )

    counts = (
        working["class_name"]
        .value_counts()
        .reindex(["ham", "spam"], fill_value=0)
    )
    plt.figure(figsize=(7, 4.5))
    plt.bar(counts.index, counts.values)
    plt.title("SMS Spam Collection Class Distribution")
    plt.xlabel("Message class")
    plt.ylabel("Messages")
    for index, value in enumerate(counts.values):
        plt.text(
            index,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )
    _save_figure(
        output_dir / "class_distribution.png"
    )

    plt.figure(figsize=(8, 4.8))
    for class_name, group in working.groupby(
        "class_name"
    ):
        plt.hist(
            group["message_length_words"],
            bins=35,
            alpha=0.55,
            label=class_name,
        )
    plt.title("SMS Message Length Distribution")
    plt.xlabel("Message length (tokens)")
    plt.ylabel("Messages")
    plt.legend()
    _save_figure(
        output_dir
        / "message_length_distribution.png"
    )

    mean_lengths = (
        working.groupby("class_name")[
            "message_length_words"
        ]
        .mean()
        .reindex(["ham", "spam"])
    )
    plt.figure(figsize=(7, 4.5))
    plt.bar(
        mean_lengths.index,
        mean_lengths.values,
    )
    plt.title("Average Message Length by Class")
    plt.xlabel("Message class")
    plt.ylabel("Average tokens")
    for index, value in enumerate(
        mean_lengths.values
    ):
        plt.text(
            index,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )
    _save_figure(
        output_dir / "average_message_length.png"
    )


def _save_evaluation_plots(
    result: TrainingResult,
    output_dir: Path,
) -> None:
    y_true = result.test_frame[
        "label"
    ].to_numpy(dtype=int)
    predictions = classify_probabilities(
        result.test_probabilities,
        result.threshold,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )
    plt.figure(figsize=(6.4, 5.2))
    image = plt.imshow(matrix)
    plt.title("Simple RNN Confusion Matrix")
    plt.xticks(
        [0, 1],
        ["Predicted ham", "Predicted spam"],
        rotation=15,
    )
    plt.yticks(
        [0, 1],
        ["Actual ham", "Actual spam"],
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
    _save_figure(
        output_dir / "confusion_matrix.png"
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            result.test_probabilities,
        )
    )
    plt.figure(figsize=(6.8, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            "Simple RNN "
            f"(AUC = {result.test_metrics['roc_auc']:.3f})"
        ),
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
    _save_figure(
        output_dir / "roc_curve.png"
    )

    precision, recall, _ = precision_recall_curve(
        y_true,
        result.test_probabilities,
    )
    plt.figure(figsize=(6.8, 5))
    plt.plot(
        recall,
        precision,
        label=(
            "Simple RNN "
            f"(PR-AUC = {result.test_metrics['pr_auc']:.3f})"
        ),
    )
    plt.title("Simple RNN Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    _save_figure(
        output_dir / "precision_recall_curve.png"
    )

    plt.figure(figsize=(7.5, 4.7))
    plt.plot(
        result.history["epoch"],
        result.history["accuracy"],
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        result.history["epoch"],
        result.history["val_accuracy"],
        marker="o",
        label="Validation accuracy",
    )
    plt.title(
        "Simple RNN Training and Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    _save_figure(
        output_dir / "training_accuracy.png"
    )

    plt.figure(figsize=(7.5, 4.7))
    plt.plot(
        result.history["epoch"],
        result.history["loss"],
        marker="o",
        label="Training loss",
    )
    plt.plot(
        result.history["epoch"],
        result.history["val_loss"],
        marker="o",
        label="Validation loss",
    )
    plt.title(
        "Simple RNN Training and Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    _save_figure(
        output_dir / "training_loss.png"
    )

    plt.figure(figsize=(8, 4.8))
    plt.hist(
        result.test_probabilities,
        bins=30,
    )
    plt.axvline(
        result.threshold,
        linestyle="--",
        label=(
            "Decision threshold = "
            f"{result.threshold:.3f}"
        ),
    )
    plt.title(
        "Simple RNN Spam-Probability Distribution"
    )
    plt.xlabel("Predicted spam probability")
    plt.ylabel("Messages")
    plt.legend()
    _save_figure(
        output_dir
        / "probability_distribution.png"
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        result.threshold_table["threshold"],
        result.threshold_table["precision"],
        label="Precision",
    )
    plt.plot(
        result.threshold_table["threshold"],
        result.threshold_table["recall"],
        label="Recall",
    )
    plt.plot(
        result.threshold_table["threshold"],
        result.threshold_table["f1"],
        label="F1-score",
    )
    plt.axvline(
        result.threshold,
        linestyle="--",
        label="Selected threshold",
    )
    plt.title("Validation Threshold Analysis")
    plt.xlabel("Spam decision threshold")
    plt.ylabel("Metric")
    plt.legend()
    _save_figure(
        output_dir / "threshold_analysis.png"
    )


def _save_baseline_terms(
    result: TrainingResult,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    feature_names = (
        result.tfidf_vectorizer
        .get_feature_names_out()
    )
    coefficients = (
        result.logistic_classifier.coef_[0]
    )

    spam_indices = np.argsort(
        coefficients
    )[-top_n:][::-1]
    ham_indices = np.argsort(
        coefficients
    )[:top_n]

    terms = pd.DataFrame(
        {
            "spam_term": feature_names[
                spam_indices
            ],
            "spam_weight": coefficients[
                spam_indices
            ],
            "ham_term": feature_names[
                ham_indices
            ],
            "ham_weight": coefficients[
                ham_indices
            ],
        }
    )
    terms.to_csv(
        output_dir / "top_terms.csv",
        index=False,
    )

    spam_plot = terms.head(15).sort_values(
        "spam_weight"
    )
    plt.figure(figsize=(8, 5.8))
    plt.barh(
        spam_plot["spam_term"],
        spam_plot["spam_weight"],
    )
    plt.title("Terms Most Associated with Spam")
    plt.xlabel(
        "TF-IDF logistic-regression coefficient"
    )
    _save_figure(
        output_dir / "top_spam_terms.png"
    )

    ham_plot = terms.head(15).copy()
    ham_plot["magnitude"] = (
        ham_plot["ham_weight"].abs()
    )
    ham_plot = ham_plot.sort_values(
        "magnitude"
    )
    plt.figure(figsize=(8, 5.8))
    plt.barh(
        ham_plot["ham_term"],
        ham_plot["magnitude"],
    )
    plt.title(
        "Terms Most Associated with Ham"
    )
    plt.xlabel(
        "Absolute TF-IDF "
        "logistic-regression coefficient"
    )
    _save_figure(
        output_dir / "top_ham_terms.png"
    )


def build_dataset_summary(
    frame: pd.DataFrame,
    result: TrainingResult,
    quality_report: dict,
) -> dict:
    """Create the dataset summary used by metadata and validation."""
    train_text = set(
        result.train_frame["clean_message"]
    )
    validation_text = set(
        result.validation_frame["clean_message"]
    )
    test_text = set(
        result.test_frame["clean_message"]
    )

    return {
        "input_rows": int(
            quality_report.get(
                "input_rows",
                len(frame),
            )
        ),
        "missing_label_rows_removed": int(
            quality_report.get(
                "missing_label_rows_removed",
                0,
            )
        ),
        "blank_message_rows_removed": int(
            quality_report.get(
                "blank_messages_removed",
                0,
            )
        ),
        "normalized_duplicate_messages_removed": int(
            quality_report.get(
                "duplicate_messages_found",
                0,
            )
        ),
        "modeling_rows": int(len(frame)),
        "ham_rows": int(
            (frame["label"] == 0).sum()
        ),
        "spam_rows": int(
            (frame["label"] == 1).sum()
        ),
        "spam_rate": float(
            frame["label"].mean()
        ),
        "train_rows": int(
            len(result.train_frame)
        ),
        "validation_rows": int(
            len(result.validation_frame)
        ),
        "test_rows": int(
            len(result.test_frame)
        ),
        "train_validation_overlap": int(
            len(
                train_text.intersection(
                    validation_text
                )
            )
        ),
        "train_test_overlap": int(
            len(
                train_text.intersection(
                    test_text
                )
            )
        ),
        "validation_test_overlap": int(
            len(
                validation_text.intersection(
                    test_text
                )
            )
        ),
    }


def save_portfolio_artifacts(
    frame: pd.DataFrame,
    result: TrainingResult,
    output_dir: str | Path,
    quality_report: dict,
) -> dict:
    """Regenerate every output referenced by the README and app."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true = result.test_frame[
        "label"
    ].to_numpy(dtype=int)
    predictions = classify_probabilities(
        result.test_probabilities,
        result.threshold,
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
            "message_id": [
                f"test_{index + 1:05d}"
                for index in range(
                    len(result.test_frame)
                )
            ],
            "true_label": y_true,
            "spam_probability": (
                result.test_probabilities
            ),
            "predicted_label": predictions,
            "predicted_class": np.where(
                predictions == 1,
                "spam",
                "ham",
            ),
            "is_correct": (
                predictions == y_true
            ),
        }
    )
    prediction_frame.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    classification_report_frame(
        y_true,
        predictions,
    ).to_csv(
        output_dir
        / "classification_report.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            _metric_row(
                "Majority Class",
                result.majority_metrics,
            ),
            _metric_row(
                "Multinomial Naive Bayes",
                result.naive_bayes_metrics,
            ),
            _metric_row(
                "TF-IDF + Logistic Regression",
                result.logistic_metrics,
            ),
            _metric_row(
                "Simple RNN",
                result.test_metrics,
            ),
        ]
    )
    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    (
        output_dir / "model_metrics.json"
    ).write_text(
        json.dumps(
            {
                "model": "Simple RNN",
                **result.test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    error_indices = np.flatnonzero(
        predictions != y_true
    )
    error_frame = result.test_frame.iloc[
        error_indices
    ][["message", "label"]].copy()
    error_frame["spam_probability"] = (
        result.test_probabilities[
            error_indices
        ]
    )
    error_frame["predicted_label"] = (
        predictions[error_indices]
    )
    error_frame["error_type"] = np.where(
        (error_frame["label"] == 0)
        & (
            error_frame[
                "predicted_label"
            ] == 1
        ),
        "false_positive",
        "false_negative",
    )
    error_frame["message_excerpt"] = (
        error_frame["message"]
        .map(clean_text)
        .str.slice(0, 220)
    )
    error_frame[
        [
            "error_type",
            "label",
            "predicted_label",
            "spam_probability",
            "message_excerpt",
        ]
    ].head(80).to_csv(
        output_dir / "error_analysis.csv",
        index=False,
    )

    dataset_summary = build_dataset_summary(
        frame,
        result,
        quality_report,
    )
    (
        output_dir / "dataset_summary.json"
    ).write_text(
        json.dumps(
            dataset_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_eda(frame, output_dir)
    _save_evaluation_plots(
        result,
        output_dir,
    )
    _save_baseline_terms(
        result,
        output_dir,
    )

    plt.figure(figsize=(9, 4.8))
    plt.bar(
        comparison["model"],
        comparison["f1"],
    )
    plt.title(
        "SMS Spam Model F1-score Comparison"
    )
    plt.xlabel("Model")
    plt.ylabel("Spam-class F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=12)
    for index, value in enumerate(
        comparison["f1"]
    ):
        plt.text(
            index,
            value,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )
    _save_figure(
        output_dir / "model_comparison.png"
    )

    return dataset_summary


def write_model_card(
    metadata: dict,
    path: str | Path,
) -> None:
    """Regenerate the model card after every training run."""
    test = metadata["test_metrics"]
    logistic = metadata["baseline_metrics"][
        "tfidf_logistic_regression"
    ]

    content = f"""# Model Card — SMS Spam Simple RNN

## Intended use

Educational and portfolio demonstration of recurrent SMS spam classification.

## Architecture

```text
Embedding({metadata['embedding_dimension']})
→ SimpleRNN({metadata['simple_rnn_units']})
→ Dense({metadata['dense_units']})
→ Dropout({metadata['dropout_rate']:.2f})
→ Sigmoid
```

## Data and split

| Component | Messages |
|---|---:|
| Cleaned modeling corpus | {metadata['modeling_rows']:,} |
| Training | {metadata['train_rows']:,} |
| Validation | {metadata['validation_rows']:,} |
| Test | {metadata['test_rows']:,} |

Normalized duplicate messages are removed before splitting. All recorded
cross-partition overlap counts are zero. The test set is untouched during model
training and threshold selection.

## Imbalance handling

- Stratified partitions
- Balanced training class weights
- Validation-selected threshold
- Spam-focused precision, recall, F1, and PR-AUC

## Simple RNN test results

| Metric | Result |
|---|---:|
| Accuracy | {test['accuracy']:.2%} |
| Spam precision | {test['precision']:.2%} |
| Spam recall | {test['recall']:.2%} |
| Spam F1 | {test['f1']:.2%} |
| ROC-AUC | {test['roc_auc']:.3f} |
| PR-AUC | {test['pr_auc']:.3f} |

## Baseline conclusion

TF-IDF + Logistic Regression achieved {logistic['accuracy']:.2%} accuracy and
{logistic['f1']:.2%} spam F1-score. It is stronger for this evaluated split.
The Simple RNN remains the primary model because the project demonstrates
recurrent sequence modeling.

## Limitations

- Small, historical English-language corpus
- Uncalibrated probabilities
- Susceptibility to obfuscation and domain drift
- Potential mistakes on legitimate promotional text
- Fixed sequence length
- Not suitable as an autonomous production filter

## Responsible use

Do not use this model as the sole basis for blocking or monitoring private
messages. Human review and organization-specific validation are required for
consequential use.
"""
    Path(path).write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
