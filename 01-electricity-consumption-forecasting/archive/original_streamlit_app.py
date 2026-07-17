
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple RNN Electricity Forecasting App
Synthetic validation first, then real public electricity/time-series data through the same pipeline.
"""

import io
import json
import math
import time
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


APP_SEED = 42
np.random.seed(APP_SEED)

REAL_DATA_URLS = [
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
]


def ensure_output_dir(base="outputs"):
    root = Path(base)
    root.mkdir(exist_ok=True)
    run_dir = root / f"rnn_forecasting_streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_for_excel_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", text)
    return text[:32767]


def sanitize_dataframe_for_excel(df):
    safe = df.copy()
    safe.columns = [sanitize_for_excel_value(c)[:31] for c in safe.columns]
    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].map(sanitize_for_excel_value)
    return safe


def generate_synthetic_electricity_series(n_hours=1200, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    daily = 12 * np.sin(2 * np.pi * t / 24)
    weekly = 7 * np.sin(2 * np.pi * t / (24 * 7))
    trend = 0.015 * t
    temp_effect = 5 * np.sin(2 * np.pi * (t + 6) / 24)
    noise = rng.normal(0, 2.5, n_hours)
    load = 120 + daily + weekly + trend + temp_effect + noise
    dates = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"timestamp": dates, "load": load, "source_type": "synthetic"})


def load_real_public_series(max_rows=1800):
    last_error = None
    for url in REAL_DATA_URLS:
        try:
            df = pd.read_csv(url)
            if "date" in df.columns:
                ts = pd.to_datetime(df["date"], errors="coerce")
            elif "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], errors="coerce")
            else:
                ts = pd.date_range("2021-01-01", periods=len(df), freq="h")
            target_col = "OT" if "OT" in df.columns else None
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric target column found in real dataset.")
                target_col = numeric_cols[-1]
            out = pd.DataFrame({
                "timestamp": ts,
                "load": pd.to_numeric(df[target_col], errors="coerce"),
                "source_type": "real_public",
            }).dropna()
            out = out.head(max_rows).reset_index(drop=True)
            if len(out) < 200:
                raise ValueError("Real dataset loaded but too few valid rows.")
            return out, f"real_public:{url}"
        except Exception as exc:
            last_error = str(exc)
    fallback = generate_synthetic_electricity_series(max_rows, seed=999)
    fallback["source_type"] = "fallback_real_like"
    return fallback, f"fallback_real_like because {last_error}"


class MinMaxScaler1D:
    def fit(self, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        self.min_ = float(np.nanmin(arr))
        self.max_ = float(np.nanmax(arr))
        self.scale_ = max(self.max_ - self.min_, 1e-9)
        return self

    def transform(self, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        return (arr - self.min_) / self.scale_

    def inverse_transform(self, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr * self.scale_ + self.min_


def make_windows(series, lookback=24, horizon=1):
    arr = np.asarray(series, dtype=float).reshape(-1)
    X, y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback + horizon - 1])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


class SimpleRNNForecaster:
    """
    Lightweight Simple RNN encoder with fixed recurrent weights and ridge-regression output head.
    This keeps the project fast and stable while still using recurrent hidden-state dynamics.
    """
    def __init__(self, lookback=24, hidden_size=32, ridge_alpha=1e-3, seed=42):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.is_fitted = False

    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        self.W_in = rng.normal(0, 0.35, size=(1, self.hidden_size))
        self.W_h = rng.normal(0, 0.08, size=(self.hidden_size, self.hidden_size))
        self.b_h = np.zeros(self.hidden_size)

    def _hidden_features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        H = []
        for row in X:
            h = np.zeros(self.hidden_size)
            for value in row:
                inp = np.array([[value]])
                h = np.tanh(inp @ self.W_in + h @ self.W_h + self.b_h).reshape(-1)
            H.append(h)
        H = np.asarray(H)
        return np.c_[np.ones(len(H)), H]

    def fit(self, X, y):
        self._init_weights()
        Phi = self._hidden_features(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        reg = self.ridge_alpha * np.eye(Phi.shape[1])
        reg[0, 0] = 0
        self.W_out = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        Phi = self._hidden_features(X)
        return (Phi @ self.W_out).reshape(-1)


def evaluate_forecast(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100)
    return {"label": label, "rows": len(y_true), "mae": mae, "rmse": rmse, "mape_pct": mape}


def run_pipeline(df, label, lookback=24, train_ratio=0.75, hidden_size=32):
    values = df["load"].astype(float).values
    split = int(len(values) * train_ratio)
    train_values = values[:split]
    test_values = values[split - lookback:]
    scaler = MinMaxScaler1D().fit(train_values)
    X_train, y_train = make_windows(scaler.transform(train_values), lookback=lookback)
    X_test, y_test = make_windows(scaler.transform(test_values), lookback=lookback)
    model = SimpleRNNForecaster(lookback=lookback, hidden_size=hidden_size).fit(X_train, y_train)
    pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    metrics = evaluate_forecast(y_true, y_pred, label)
    pred_df = pd.DataFrame({
        "label": label,
        "step": np.arange(len(y_true)),
        "actual": y_true,
        "prediction": y_pred,
        "absolute_error": np.abs(y_true - y_pred),
    })
    return model, scaler, metrics, pred_df


def export_results(run_dir, metrics_df, pred_df, data_source_name):
    metrics_path = run_dir / "metrics.csv"
    pred_path = run_dir / "predictions.csv"
    excel_path = run_dir / "rnn_forecasting_report.xlsx"
    manifest_path = run_dir / "manifest.json"
    zip_path = run_dir / "rnn_forecasting_outputs.zip"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sanitize_dataframe_for_excel(metrics_df).to_excel(writer, index=False, sheet_name="metrics")
        sanitize_dataframe_for_excel(pred_df.head(1000)).to_excel(writer, index=False, sheet_name="predictions_sample")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "data_source": data_source_name,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(len(pred_df)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [metrics_path, pred_path, excel_path, manifest_path]:
            z.write(p, arcname=p.name)
    return excel_path, zip_path


st.set_page_config(page_title="Simple RNN Electricity Forecasting", layout="wide")
st.title("Simple RNN Electricity Forecasting")
st.caption("Synthetic validation first, then real public electricity/time-series data through the same pipeline.")

with st.sidebar:
    mode = st.selectbox("Dataset mode", ["synthetic_only", "synthetic_plus_real"], index=1)
    lookback = st.slider("Lookback window", 6, 72, 24)
    hidden_size = st.slider("RNN hidden size", 8, 96, 32)
    max_real_rows = st.slider("Max real rows", 300, 3000, 1200, step=100)

if st.button("Run forecasting pipeline"):
    run_dir = ensure_output_dir()
    synthetic_df = generate_synthetic_electricity_series(n_hours=1200)
    all_metrics = []
    all_predictions = []

    _, _, syn_metrics, syn_pred = run_pipeline(synthetic_df, "synthetic", lookback=lookback, hidden_size=hidden_size)
    all_metrics.append(syn_metrics)
    all_predictions.append(syn_pred)

    real_source = "not_loaded"
    if mode == "synthetic_plus_real":
        real_df, real_source = load_real_public_series(max_rows=max_real_rows)
        _, _, real_metrics, real_pred = run_pipeline(real_df, "real_or_fallback", lookback=lookback, hidden_size=hidden_size)
        all_metrics.append(real_metrics)
        all_predictions.append(real_pred)

    metrics_df = pd.DataFrame(all_metrics)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    excel_path, zip_path = export_results(run_dir, metrics_df, pred_df, real_source)

    st.subheader("Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Predictions")
    st.dataframe(pred_df.head(200), use_container_width=True)

    if HAS_MPL:
        st.subheader("Forecast plot")
        fig, ax = plt.subplots(figsize=(11, 4))
        for label, grp in pred_df.groupby("label"):
            ax.plot(grp["step"].head(150), grp["actual"].head(150), label=f"{label} actual")
            ax.plot(grp["step"].head(150), grp["prediction"].head(150), linestyle="--", label=f"{label} prediction")
        ax.set_xlabel("Step")
        ax.set_ylabel("Load")
        ax.legend()
        st.pyplot(fig)

    st.success(f"Outputs saved to: {run_dir}")
    st.download_button("Download Excel report", data=excel_path.read_bytes(), file_name=excel_path.name)
    st.download_button("Download ZIP bundle", data=zip_path.read_bytes(), file_name=zip_path.name)
