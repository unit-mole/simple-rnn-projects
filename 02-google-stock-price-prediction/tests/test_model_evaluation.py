import numpy as np

from src.model_evaluation import reconstruct_close, regression_metrics


def test_reconstruct_close_from_log_return():
    current = np.array([100.0])
    predicted_return = np.array([np.log(1.02)])
    predicted = reconstruct_close(current, predicted_return)
    assert np.isclose(predicted[0], 102.0)


def test_regression_metrics_are_zero_for_perfect_prediction():
    actual = np.array([100.0, 102.0, 101.0])
    metrics = regression_metrics(actual, actual)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0
    assert metrics["R2"] == 1.0
