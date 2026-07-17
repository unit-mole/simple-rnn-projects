"""Reusable loading and recursive forecasting pipeline for the Streamlit app."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import joblib
import keras
import numpy as np
import pandas as pd

from src.feature_engineering import add_calendar_features, calendar_vector


class ElectricityForecastingPipeline:
    def __init__(self, model_dir: str | Path):
        model_path = Path(model_dir)
        self.model = keras.models.load_model(model_path / "electricity_rnn_model.keras", compile=False)
        self.scaler = joblib.load(model_path / "scaler.pkl")
        self.metadata = json.loads(
            (model_path / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.lookback = int(self.metadata["lookback"])
        self.target_column = self.metadata["target_column"]
        self.frequency = self.metadata["frequency"]

    def _history_matrix(self, history_df: pd.DataFrame) -> np.ndarray:
        featured = add_calendar_features(history_df)
        scaled_target = self.scaler.transform(
            featured[[self.target_column]].to_numpy(dtype=float)
        ).reshape(-1)
        featured["consumption_scaled"] = scaled_target
        columns = self.metadata["model_feature_columns"]
        return featured[columns].to_numpy(dtype=np.float32)

    def forecast(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least 1.")
        if len(history_df) < self.lookback:
            raise ValueError(f"At least {self.lookback} historical rows are required.")

        history = history_df[["timestamp", self.target_column]].copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        matrix = self._history_matrix(history).tolist()
        last_timestamp = history["timestamp"].iloc[-1]
        offset = pd.tseries.frequencies.to_offset(self.frequency)
        rows: list[dict[str, float | pd.Timestamp]] = []

        for step in range(1, horizon + 1):
            sequence = np.asarray(matrix[-self.lookback :], dtype=np.float32)[None, :, :]
            prediction_scaled = float(self.model.predict(sequence, verbose=0).reshape(-1)[0])
            prediction = float(
                self.scaler.inverse_transform([[prediction_scaled]]).reshape(-1)[0]
            )
            next_timestamp = last_timestamp + offset
            calendar = calendar_vector(next_timestamp)
            next_vector = np.concatenate(
                [np.asarray([prediction_scaled], dtype=np.float32), calendar]
            )
            matrix.append(next_vector.tolist())
            rows.append(
                {
                    "step": step,
                    "timestamp": next_timestamp,
                    "forecasted_consumption_kwh": prediction,
                }
            )
            last_timestamp = next_timestamp

        return pd.DataFrame(rows)

    def interpret(self, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict[str, str | float]:
        recent_mean = float(history_df[self.target_column].tail(self.lookback).mean())
        forecast_mean = float(forecast_df["forecasted_consumption_kwh"].mean())
        change_pct = 100 * (forecast_mean - recent_mean) / max(abs(recent_mean), 1e-8)
        first = float(forecast_df["forecasted_consumption_kwh"].iloc[0])
        last = float(forecast_df["forecasted_consumption_kwh"].iloc[-1])
        if last > first * 1.02:
            trend = "Increasing"
        elif last < first * 0.98:
            trend = "Decreasing"
        else:
            trend = "Relatively stable"
        peak_row = forecast_df.loc[forecast_df["forecasted_consumption_kwh"].idxmax()]
        if change_pct >= 5:
            message = "Demand is expected to rise; review capacity, staffing, and peak-demand monitoring."
        elif change_pct <= -5:
            message = "Demand is expected to ease; consider load shifting or maintenance opportunities."
        else:
            message = "Demand is expected to remain near the recent operating range; continue routine monitoring."
        return {
            "trend": trend,
            "forecast_mean_kwh": forecast_mean,
            "change_vs_recent_pct": change_pct,
            "peak_timestamp": str(peak_row["timestamp"]),
            "peak_consumption_kwh": float(peak_row["forecasted_consumption_kwh"]),
            "business_interpretation": message,
        }
