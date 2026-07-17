"""End-to-end training, evaluation, and next-day forecasting pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd

from src.data_preprocessing import load_and_standardize, standardize_stock_data
from src.feature_engineering import build_return_forecast_frame
from src.model_evaluation import (
    comparison_table,
    forecast_interpretation,
    reconstruct_close,
    regression_metrics,
)
from src.model_training import build_simple_rnn_model, save_training_artifacts, train_model
from src.sequence_generation import (
    chronological_split_and_scale,
    create_return_sequences,
    scale_latest_window,
)


def train_evaluate_pipeline(
    data_path,
    project_root: str | Path,
    target_column: str = "Close",
    window_size: int = 10,
    random_seed: int = 42,
) -> dict:
    """Train the Simple RNN, evaluate it, save artifacts, and return result tables."""
    project_root = Path(project_root)
    model_dir = project_root / "models"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(random_seed)
    clean_df, selected_target, quality = load_and_standardize(data_path, target_column)
    frame = build_return_forecast_frame(clean_df, selected_target)
    sequences = create_return_sequences(frame, window_size=window_size)
    split = chronological_split_and_scale(sequences)

    model = build_simple_rnn_model(window_size=window_size, n_features=1)
    history = train_model(
        model,
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
    )

    predicted_scaled = model.predict(split.X_test, verbose=0).reshape(-1, 1)
    predicted_returns = split.target_scaler.inverse_transform(predicted_scaled).ravel()
    actual_returns = split.target_scaler.inverse_transform(split.y_test).ravel()
    predicted_close = reconstruct_close(split.current_close_test, predicted_returns)

    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(split.target_dates_test),
            "Current_Close": split.current_close_test,
            "Actual_Close": split.target_close_test,
            "Predicted_Close": predicted_close,
            "Actual_Log_Return": actual_returns,
            "Predicted_Log_Return": predicted_returns,
        }
    )
    predictions["Residual"] = predictions["Actual_Close"] - predictions["Predicted_Close"]
    predictions["Absolute_Error"] = predictions["Residual"].abs()

    metrics = regression_metrics(
        split.target_close_test,
        predicted_close,
        current_close=split.current_close_test,
    )
    comparison = comparison_table(
        split.target_close_test,
        split.current_close_test,
        predicted_close,
        split.raw_returns_test,
    )

    all_log_returns = np.log(
        clean_df[selected_target].astype(float) / clean_df[selected_target].astype(float).shift(1)
    ).dropna()
    latest_window = scale_latest_window(all_log_returns.to_numpy(), split.feature_scaler, window_size)
    next_scaled = model.predict(latest_window, verbose=0).reshape(-1, 1)
    next_return = float(split.target_scaler.inverse_transform(next_scaled).ravel()[0])
    latest_close = float(clean_df[selected_target].iloc[-1])
    predicted_next_close = float(reconstruct_close([latest_close], [next_return])[0])
    forecast_date = (pd.Timestamp(clean_df["Date"].iloc[-1]) + pd.offsets.BDay(1)).date().isoformat()

    metadata = {
        "project_name": "Google Stock Price Prediction using Simple RNN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_start": clean_df["Date"].min().date().isoformat(),
        "training_data_end": clean_df["Date"].max().date().isoformat(),
        "target_column": selected_target,
        "forecast_target": "next-day log return reconstructed as next closing price",
        "window_size": window_size,
        "forecast_horizon": 1,
        "architecture": "SimpleRNN(16) -> Dropout(0.05) -> Dense(8) -> Dense(1)",
        "loss": "Huber",
        "optimizer": "Adam",
        "chronological_split": split.split_indices,
        "quality_report": quality,
        "test_metrics": metrics,
        "latest_close": latest_close,
        "predicted_next_close": predicted_next_close,
        "estimated_forecast_date": forecast_date,
        "disclaimer": "Educational portfolio demonstration only; not financial advice.",
    }
    artifact_paths = save_training_artifacts(
        model,
        split.feature_scaler,
        split.target_scaler,
        metadata,
        model_dir,
    )

    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "Epoch"
    history_df = history_df.reset_index()

    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    next_forecast = {
        "latest_observation_date": clean_df["Date"].iloc[-1].date().isoformat(),
        "estimated_next_business_date": forecast_date,
        "latest_close": latest_close,
        "predicted_close": predicted_next_close,
        "predicted_change_pct": (predicted_next_close / latest_close - 1.0) * 100,
        "interpretation": forecast_interpretation(latest_close, predicted_next_close),
        "financial_disclaimer": "Educational portfolio demonstration only; not financial advice.",
    }
    (output_dir / "next_day_forecast.json").write_text(
        json.dumps(next_forecast, indent=2), encoding="utf-8"
    )

    return {
        "clean_data": clean_df,
        "predictions": predictions,
        "comparison": comparison,
        "metrics": metrics,
        "history": history_df,
        "metadata": metadata,
        "next_forecast": next_forecast,
        "artifact_paths": artifact_paths,
    }


def load_inference_artifacts(model_dir: str | Path):
    """Load the saved model and train-fitted scalers."""
    model_dir = Path(model_dir)
    model = keras.models.load_model(model_dir / "google_stock_rnn_model.keras", compile=False)
    feature_scaler = joblib.load(model_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(model_dir / "target_scaler.joblib")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    return model, feature_scaler, target_scaler, metadata


def forecast_from_dataframe(
    df: pd.DataFrame,
    model,
    feature_scaler,
    target_scaler,
    metadata: dict,
    target_column: str | None = None,
) -> dict:
    """Generate a next-session forecast from a clean or uploaded dataframe."""
    clean_df, selected_target, quality = standardize_stock_data(df, target_column)
    window_size = int(metadata["window_size"])
    log_returns = np.log(
        clean_df[selected_target].astype(float) / clean_df[selected_target].astype(float).shift(1)
    ).dropna()
    latest_window = scale_latest_window(log_returns.to_numpy(), feature_scaler, window_size)
    predicted_scaled = model.predict(latest_window, verbose=0).reshape(-1, 1)
    predicted_return = float(target_scaler.inverse_transform(predicted_scaled).ravel()[0])
    latest_close = float(clean_df[selected_target].iloc[-1])
    predicted_close = float(reconstruct_close([latest_close], [predicted_return])[0])
    forecast_date = (pd.Timestamp(clean_df["Date"].iloc[-1]) + pd.offsets.BDay(1)).date().isoformat()
    return {
        "clean_data": clean_df,
        "selected_target": selected_target,
        "quality_report": quality,
        "latest_close": latest_close,
        "predicted_close": predicted_close,
        "predicted_log_return": predicted_return,
        "predicted_change_pct": (predicted_close / latest_close - 1.0) * 100,
        "estimated_forecast_date": forecast_date,
        "interpretation": forecast_interpretation(latest_close, predicted_close),
    }
