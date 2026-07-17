"""Train and evaluate the GitHub-ready Simple RNN forecasting project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DEFAULT_FREQUENCY,
    DEFAULT_LOOKBACK,
    MODEL_DIR,
    MODEL_FEATURE_COLUMNS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_RATIO,
    TIMESTAMP_COLUMN,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from src.data_preprocessing import load_csv, prepare_time_series
from src.feature_engineering import add_calendar_features
from src.model_evaluation import baseline_predictions, comparison_table, regression_metrics
from src.model_training import build_simple_rnn_model, save_artifacts, train_simple_rnn
from src.sequence_generation import chronological_boundaries, create_partitioned_sequences
from src.visualization import (
    plot_actual_vs_predicted,
    plot_consumption_trend,
    plot_error_distribution,
    plot_forecast,
    plot_residuals,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_input.csv")
    parser.add_argument("--timestamp-column", default=TIMESTAMP_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--frequency", default=DEFAULT_FREQUENCY)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    import keras
    keras.utils.set_random_seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    raw = load_csv(data_path)
    cleaned, quality_report = prepare_time_series(
        raw,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
        frequency=args.frequency,
    )
    cleaned = cleaned.rename(
        columns={args.timestamp_column: "timestamp", args.target_column: TARGET_COLUMN}
    )
    featured = add_calendar_features(cleaned)
    boundaries = chronological_boundaries(
        len(featured), train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO
    )

    scaler = MinMaxScaler()
    scaler.fit(featured[[TARGET_COLUMN]].iloc[: boundaries.train_end].to_numpy())
    featured["consumption_scaled"] = scaler.transform(featured[[TARGET_COLUMN]].to_numpy())

    feature_matrix = featured[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaled_target = featured["consumption_scaled"].to_numpy(dtype=np.float32)
    partitions = create_partitioned_sequences(
        feature_matrix, scaled_target, args.lookback, boundaries
    )
    x_train, y_train, _ = partitions["train"]
    x_validation, y_validation, _ = partitions["validation"]
    x_test, y_test, test_indices = partitions["test"]

    model = build_simple_rnn_model(
        lookback=args.lookback,
        n_features=x_train.shape[-1],
        rnn_units=64,
        dense_units=32,
        dropout_rate=0.05,
        learning_rate=0.001,
    )
    history = train_simple_rnn(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0 if args.quiet else 1,
    )

    predicted_scaled = model.predict(x_test, verbose=0).reshape(-1)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(-1)
    test_timestamps = featured["timestamp"].iloc[test_indices].reset_index(drop=True)

    metrics = regression_metrics(actual, predicted, "Simple RNN")
    baselines = baseline_predictions(
        featured[TARGET_COLUMN].to_numpy(), test_indices, seasonal_lag=24, moving_average_window=24
    )
    comparison = comparison_table(actual, predicted, baselines)
    predictions = pd.DataFrame(
        {
            "timestamp": test_timestamps,
            "actual_consumption_kwh": actual,
            "predicted_consumption_kwh": predicted,
            "residual": actual - predicted,
            "absolute_error": np.abs(actual - predicted),
        }
    )

    output_dir = project_root / OUTPUT_DIR.relative_to(project_root)
    model_dir = project_root / MODEL_DIR.relative_to(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": "Electricity Consumption Forecasting with a Simple RNN",
        "data_file": str(data_path.name),
        "data_is_synthetic_demo": bool(
            "source_type" in raw.columns and raw["source_type"].astype(str).str.contains("synthetic").all()
        ),
        "timestamp_column": "timestamp",
        "target_column": TARGET_COLUMN,
        "frequency": args.frequency,
        "lookback": args.lookback,
        "forecast_horizon_training": 1,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "train_rows": boundaries.train_end,
        "validation_rows": boundaries.validation_end - boundaries.train_end,
        "test_rows": boundaries.total_rows - boundaries.validation_end,
        "split_strategy": "chronological_70_15_15",
        "scaler_fit_scope": "training_target_only",
        "keras_backend": __import__("keras").backend.backend(),
        "epochs_completed": len(history.history["loss"]),
        "metrics": metrics,
        "data_quality_report": quality_report.to_dict(),
    }
    save_artifacts(model, scaler, metadata, model_dir)

    history_df = pd.DataFrame(history.history)
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality_report.to_dict(), indent=2), encoding="utf-8"
    )

    plot_consumption_trend(featured, TARGET_COLUMN, output_dir / "consumption_trend.png")
    plot_training_history(history.history, output_dir / "training_curve.png")
    plot_actual_vs_predicted(
        test_timestamps, actual, predicted, output_dir / "actual_vs_predicted.png"
    )
    plot_residuals(test_timestamps, actual, predicted, output_dir / "residual_plot.png")
    plot_error_distribution(actual, predicted, output_dir / "error_distribution.png")

    # Generate a portfolio forecast from the end of the sample series.
    from src.forecasting_pipeline import ElectricityForecastingPipeline

    pipeline = ElectricityForecastingPipeline(model_dir)
    forecast_df = pipeline.forecast(featured[["timestamp", TARGET_COLUMN]], horizon=24)
    forecast_df.to_csv(output_dir / "next_24_hour_forecast.csv", index=False)
    plot_forecast(
        featured[["timestamp", TARGET_COLUMN]],
        forecast_df,
        TARGET_COLUMN,
        output_dir / "forecast_plot.png",
    )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print("Best comparison row:")
    print(comparison.head(1).to_string(index=False))
    print(f"Artifacts saved to: {model_dir}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
