import numpy as np

from src.model_evaluation import (
    classify_probabilities,
    evaluate_binary_classifier,
    select_threshold,
)


def test_threshold_classification():
    probabilities = np.array([0.2, 0.42, 0.43, 0.9])
    predictions = classify_probabilities(probabilities, threshold=0.43)
    assert predictions.tolist() == [0, 0, 1, 1]


def test_evaluation_metrics_are_valid():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate_binary_classifier(y_true, probabilities, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_threshold_selection_uses_candidate_range():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.45, 0.8])
    threshold, table = select_threshold(
        y_true,
        probabilities,
        minimum=0.3,
        maximum=0.6,
        steps=7,
    )
    assert 0.3 <= threshold <= 0.6
    assert len(table) == 7
