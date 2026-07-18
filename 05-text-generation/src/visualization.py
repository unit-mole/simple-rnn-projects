"""Plotting utilities for training and generation evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_df.index + 1, history_df["loss"], marker="o", label="Training loss")
    if "val_loss" in history_df:
        axis.plot(history_df.index + 1, history_df["val_loss"], marker="o", label="Validation loss")
    axis.set_title("Simple RNN Training Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Sparse categorical cross-entropy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_temperature_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        metrics_df["temperature"],
        metrics_df["unique_trigram_ratio"],
        marker="o",
        label="Unique trigram ratio",
    )
    axis.plot(
        metrics_df["temperature"],
        metrics_df["repeated_trigram_ratio"],
        marker="o",
        label="Repeated trigram ratio",
    )
    axis.set_title("Effect of Temperature on Generated Text")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Ratio")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_sequence_summary(summary: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Training windows", "Validation windows", "Vocabulary size", "Sequence length"]
    values = [
        summary["training_sequences"],
        summary["validation_sequences"],
        summary["vocabulary_size"],
        summary["sequence_length"],
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values)
    axis.set_title("Sequence Dataset Summary")
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
