"""Chronological sequence generation and train-only scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    current_close: np.ndarray
    target_close: np.ndarray
    target_dates: np.ndarray
    raw_returns: np.ndarray


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    current_close_test: np.ndarray
    target_close_test: np.ndarray
    target_dates_test: np.ndarray
    raw_returns_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    split_indices: dict


def create_return_sequences(frame: pd.DataFrame, window_size: int = 10) -> SequenceData:
    """Use the previous `window_size` log returns to predict the next return."""
    if len(frame) < window_size + 5:
        raise ValueError(
            f"At least {window_size + 5} engineered rows are required; received {len(frame)}."
        )

    X, y, current, target, dates, raw = [], [], [], [], [], []
    for i in range(window_size - 1, len(frame)):
        return_window = frame["Log_Return"].iloc[i - window_size + 1 : i + 1].to_numpy()
        X.append(return_window.reshape(window_size, 1))
        y.append(frame["Target_Return"].iloc[i])
        current.append(frame["Current_Close"].iloc[i])
        target.append(frame["Target_Close"].iloc[i])
        dates.append(frame["Target_Date"].iloc[i])
        raw.append(return_window)

    return SequenceData(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        current_close=np.asarray(current, dtype=float),
        target_close=np.asarray(target, dtype=float),
        target_dates=np.asarray(dates),
        raw_returns=np.asarray(raw, dtype=float),
    )


def chronological_split_and_scale(
    data: SequenceData,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> SplitData:
    """Split in time order and fit both scalers on training observations only."""
    n_samples = len(data.X)
    train_end = int(n_samples * train_ratio)
    validation_end = train_end + int(n_samples * validation_ratio)

    if train_end < 20 or validation_end >= n_samples:
        raise ValueError("The dataset is too small for the requested chronological split.")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(data.X[:train_end].reshape(-1, data.X.shape[-1]))
    target_scaler.fit(data.y[:train_end].reshape(-1, 1))

    X_scaled = feature_scaler.transform(
        data.X.reshape(-1, data.X.shape[-1])
    ).reshape(data.X.shape).astype(np.float32)
    y_scaled = target_scaler.transform(data.y.reshape(-1, 1)).astype(np.float32)

    return SplitData(
        X_train=X_scaled[:train_end],
        y_train=y_scaled[:train_end],
        X_validation=X_scaled[train_end:validation_end],
        y_validation=y_scaled[train_end:validation_end],
        X_test=X_scaled[validation_end:],
        y_test=y_scaled[validation_end:],
        current_close_test=data.current_close[validation_end:],
        target_close_test=data.target_close[validation_end:],
        target_dates_test=data.target_dates[validation_end:],
        raw_returns_test=data.raw_returns[validation_end:],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        split_indices={
            "n_total_sequences": int(n_samples),
            "train_end": int(train_end),
            "validation_end": int(validation_end),
            "n_train": int(train_end),
            "n_validation": int(validation_end - train_end),
            "n_test": int(n_samples - validation_end),
        },
    )


def scale_latest_window(
    log_returns: np.ndarray,
    feature_scaler: StandardScaler,
    window_size: int,
) -> np.ndarray:
    """Prepare the latest return sequence for model inference."""
    values = np.asarray(log_returns, dtype=float)
    if len(values) < window_size:
        raise ValueError(f"At least {window_size} valid returns are required for forecasting.")
    latest = values[-window_size:].reshape(window_size, 1)
    scaled = feature_scaler.transform(latest)
    return scaled.reshape(1, window_size, 1).astype(np.float32)
