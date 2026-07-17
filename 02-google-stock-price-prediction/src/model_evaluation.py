"""Forecast metrics, baseline comparison, and interpretation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def reconstruct_close(current_close, predicted_log_return):
    """Convert a predicted log return into the next closing price."""
    return np.asarray(current_close, dtype=float) * np.exp(
        np.asarray(predicted_log_return, dtype=float)
    )


def regression_metrics(actual, predicted, current_close=None) -> dict:
    """Calculate price-error and optional direction metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-9))) * 100)
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    if current_close is not None:
        current = np.asarray(current_close, dtype=float)
        actual_direction = np.sign(actual - current)
        predicted_direction = np.sign(predicted - current)
        metrics["Directional_Accuracy"] = float(np.mean(actual_direction == predicted_direction))
    return metrics


def build_baseline_predictions(current_close, raw_return_windows) -> dict[str, np.ndarray]:
    """Build naive and five-day mean-return forecasting baselines."""
    current = np.asarray(current_close, dtype=float)
    raw_returns = np.asarray(raw_return_windows, dtype=float)
    mean_return_5 = raw_returns[:, -5:].mean(axis=1)
    return {
        "Naive Previous Close": current.copy(),
        "5-Day Mean Return": reconstruct_close(current, mean_return_5),
    }


def comparison_table(actual, current_close, rnn_predictions, raw_return_windows) -> pd.DataFrame:
    """Compare the Simple RNN against transparent time-series baselines."""
    rows = []
    predictions = {"Simple RNN": np.asarray(rnn_predictions, dtype=float)}
    predictions.update(build_baseline_predictions(current_close, raw_return_windows))
    for model_name, values in predictions.items():
        row = {"Model": model_name}
        row.update(regression_metrics(actual, values, current_close=current_close))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def forecast_interpretation(current_close: float, predicted_close: float) -> str:
    """Return a cautious directional interpretation for the app."""
    pct_change = (predicted_close / current_close - 1.0) * 100
    if pct_change > 0.25:
        direction = "slight upward movement"
    elif pct_change < -0.25:
        direction = "slight downward movement"
    else:
        direction = "broadly flat movement"
    return (
        f"The model indicates {direction} for the next trading session "
        f"({pct_change:+.2f}% versus the latest close)."
    )
