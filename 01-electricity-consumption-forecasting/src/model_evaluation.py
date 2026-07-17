"""Forecast metrics and transparent baseline comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100)


def regression_metrics(y_true, y_pred, model_name: str) -> dict[str, float | str | int]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "model": model_name,
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape_pct": safe_mape(actual, predicted),
        "smape_pct": smape(actual, predicted),
        "mean_residual": float(np.mean(actual - predicted)),
    }


def baseline_predictions(
    full_target: np.ndarray,
    target_indices: np.ndarray,
    seasonal_lag: int = 24,
    moving_average_window: int = 24,
) -> dict[str, np.ndarray]:
    values = np.asarray(full_target, dtype=float).reshape(-1)
    indices = np.asarray(target_indices, dtype=int)
    naive = np.asarray([values[index - 1] for index in indices])
    seasonal = np.asarray([
        values[index - seasonal_lag] if index >= seasonal_lag else values[index - 1]
        for index in indices
    ])
    moving_average = np.asarray([
        values[max(0, index - moving_average_window) : index].mean()
        for index in indices
    ])
    return {
        "Naive previous value": naive,
        f"Seasonal naive ({seasonal_lag} steps)": seasonal,
        f"Moving average ({moving_average_window} steps)": moving_average,
    }


def comparison_table(y_true, rnn_prediction, baselines: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = [regression_metrics(y_true, rnn_prediction, "Simple RNN")]
    rows.extend(regression_metrics(y_true, values, name) for name, values in baselines.items())
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
