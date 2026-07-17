"""Evaluation and validation-threshold selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

def classify_probabilities(spam_probabilities: np.ndarray, threshold: float) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    return (np.asarray(spam_probabilities, dtype=float) >= threshold).astype(int)

def evaluate_binary_classifier(
    y_true: np.ndarray,
    spam_probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(spam_probabilities, dtype=float)
    predictions = classify_probabilities(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "test_rows": int(len(y_true)),
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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def select_threshold(
    y_true: np.ndarray,
    spam_probabilities: np.ndarray,
    minimum: float = 0.10,
    maximum: float = 0.90,
    steps: int = 161,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(minimum, maximum, steps):
        predictions = classify_probabilities(spam_probabilities, float(threshold))
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["f1", "precision", "balanced_accuracy"], ascending=False
    ).iloc[0]
    return float(best["threshold"]), table

def classification_report_frame(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    report = classification_report(
        y_true, predictions, target_names=["ham", "spam"],
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index(names="class")
