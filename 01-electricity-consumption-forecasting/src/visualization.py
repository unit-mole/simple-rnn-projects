"""Matplotlib figures saved for GitHub and Streamlit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_consumption_trend(df: pd.DataFrame, target_column: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df[target_column], linewidth=1)
    ax.set(title="Electricity Consumption Trend", xlabel="Timestamp", ylabel="Consumption (kWh)")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.get("loss", []), label="Training loss")
    ax.plot(history.get("val_loss", []), label="Validation loss")
    ax.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="MSE loss")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_actual_vs_predicted(timestamps, actual, predicted, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, actual, label="Actual", linewidth=1.5)
    ax.plot(timestamps, predicted, label="Simple RNN", linewidth=1.3)
    ax.set(title="Actual vs Predicted Electricity Consumption", xlabel="Timestamp", ylabel="Consumption (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_residuals(timestamps, actual, predicted, path: str | Path) -> None:
    residuals = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, residuals, linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(title="Forecast Residuals", xlabel="Timestamp", ylabel="Actual - Prediction")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_error_distribution(actual, predicted, path: str | Path) -> None:
    absolute_error = np.abs(np.asarray(actual) - np.asarray(predicted))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(absolute_error, bins=30, edgecolor="black", alpha=0.8)
    ax.set(title="Absolute Error Distribution", xlabel="Absolute error (kWh)", ylabel="Frequency")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, target_column: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    recent = history_df.tail(7 * 24)
    ax.plot(recent["timestamp"], recent[target_column], label="Recent history")
    ax.plot(forecast_df["timestamp"], forecast_df["forecasted_consumption_kwh"], label="Forecast")
    ax.axvline(recent["timestamp"].iloc[-1], linestyle="--", linewidth=1)
    ax.set(title="Recursive Multi-step Electricity Forecast", xlabel="Timestamp", ylabel="Consumption (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)
