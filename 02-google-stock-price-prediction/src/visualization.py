"""Reusable Matplotlib visualizations for saved project outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_stock_trend(df: pd.DataFrame, target_column: str, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df[target_column], label=target_column)
    ax.set_title("Google Stock Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_actual_vs_predicted(predictions: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(predictions["Date"], predictions["Actual_Close"], label="Actual")
    ax.plot(predictions["Date"], predictions["Predicted_Close"], label="Simple RNN")
    ax.set_title("Actual vs Predicted Next-Day Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_residual_plot(predictions: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0, linewidth=1)
    ax.scatter(predictions["Date"], predictions["Residual"], s=24)
    ax.set_title("Forecast Residuals Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Actual − Predicted (USD)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_curve(history: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history["Epoch"], history["loss"], label="Training loss")
    ax.plot(history["Epoch"], history["val_loss"], label="Validation loss")
    ax.set_title("Simple RNN Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_next_forecast_plot(clean_df: pd.DataFrame, target_column: str, next_forecast: dict, output_path) -> None:
    recent = clean_df.tail(40)
    future_date = pd.to_datetime(next_forecast["estimated_next_business_date"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent["Date"], recent[target_column], label="Observed close")
    ax.plot(
        [recent["Date"].iloc[-1], future_date],
        [recent[target_column].iloc[-1], next_forecast["predicted_close"]],
        marker="o",
        label="Next-session forecast",
    )
    ax.set_title("Next-Session Google Closing-Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_baseline_comparison(comparison: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(comparison["Model"], comparison["RMSE"])
    ax.set_title("RMSE Baseline Comparison")
    ax.set_ylabel("RMSE (USD)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
