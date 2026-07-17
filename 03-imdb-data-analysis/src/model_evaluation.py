"""Evaluation utilities for binary sentiment classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classify_probabilities(
    positive_probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    return (probabilities >= threshold).astype(int)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    positive_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    predictions = classify_probabilities(probabilities, threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_report_frame(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index(names="class")


def select_threshold(
    y_true: np.ndarray,
    positive_probabilities: np.ndarray,
    minimum: float = 0.20,
    maximum: float = 0.80,
    steps: int = 121,
) -> tuple[float, pd.DataFrame]:
    """Select the F1-maximizing threshold using validation data only."""
    rows = []
    for threshold in np.linspace(minimum, maximum, steps):
        predictions = classify_probabilities(positive_probabilities, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, predictions)
                ),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, predictions, zero_division=0)
                ),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
            }
        )

    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    ).iloc[0]
    return float(best["threshold"]), table
