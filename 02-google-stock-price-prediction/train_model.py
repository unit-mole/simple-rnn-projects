#!/usr/bin/env python
"""Train and export the Google stock Simple RNN portfolio artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATA_PATH, OUTPUT_DIR, TARGET_COLUMN, WINDOW_SIZE
from src.forecasting_pipeline import train_evaluate_pipeline
from src.visualization import (
    save_actual_vs_predicted,
    save_baseline_comparison,
    save_next_forecast_plot,
    save_residual_plot,
    save_stock_trend,
    save_training_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV file path")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Price target column")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="Return window size")
    return parser.parse_args()


def main():
    args = parse_args()
    results = train_evaluate_pipeline(
        data_path=args.data,
        project_root=PROJECT_ROOT,
        target_column=args.target,
        window_size=args.window,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_stock_trend(results["clean_data"], args.target, OUTPUT_DIR / "stock_price_trend.png")
    save_actual_vs_predicted(results["predictions"], OUTPUT_DIR / "actual_vs_predicted.png")
    save_residual_plot(results["predictions"], OUTPUT_DIR / "residual_plot.png")
    save_training_curve(results["history"], OUTPUT_DIR / "training_curve.png")
    save_next_forecast_plot(
        results["clean_data"],
        results["metadata"]["target_column"],
        results["next_forecast"],
        OUTPUT_DIR / "forecast_plot.png",
    )
    save_baseline_comparison(results["comparison"], OUTPUT_DIR / "baseline_comparison.png")

    print("Training complete.")
    print(f"Model: {PROJECT_ROOT / 'models' / 'google_stock_rnn_model.keras'}")
    print("Test metrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value:.6f}")
    print("Next-session forecast:")
    for key, value in results["next_forecast"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
