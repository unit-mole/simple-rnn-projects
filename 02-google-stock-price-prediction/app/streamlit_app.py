#!/usr/bin/env python
"""Interactive Google stock next-session forecasting demo."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATA_PATH, MODEL_DIR, OUTPUT_DIR
from src.data_preprocessing import standardize_stock_data
from src.feature_engineering import build_return_forecast_frame
from src.forecasting_pipeline import forecast_from_dataframe, load_inference_artifacts
from src.model_evaluation import comparison_table, reconstruct_close
from src.sequence_generation import create_return_sequences

DISCLAIMER = """
**Financial disclaimer:** This project is for educational and portfolio demonstration purposes only.  
It is not financial advice. Stock-price predictions are uncertain and should not be used for real
investment decisions. Please consult a qualified financial adviser before making investment decisions.
"""

st.set_page_config(
    page_title="Google Stock Simple RNN Forecast",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    return load_inference_artifacts(MODEL_DIR)


@st.cache_data
def load_sample_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def download_google_data(period: str):
    try:
        import yfinance as yf

        data = yf.download(
            "GOOG",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if data is None or data.empty:
            raise ValueError("No rows were returned by yfinance.")
        return data.reset_index(), None
    except Exception as exc:
        return None, str(exc)


def backtest_loaded_model(
    clean_df,
    selected_target,
    model,
    feature_scaler,
    target_scaler,
    window_size,
):
    frame = build_return_forecast_frame(clean_df, selected_target)
    sequences = create_return_sequences(frame, window_size)
    X_scaled = feature_scaler.transform(
        sequences.X.reshape(-1, sequences.X.shape[-1])
    ).reshape(sequences.X.shape)
    predicted_scaled = model.predict(X_scaled, verbose=0).reshape(-1, 1)
    predicted_returns = target_scaler.inverse_transform(predicted_scaled).ravel()
    predicted_close = reconstruct_close(sequences.current_close, predicted_returns)

    backtest = pd.DataFrame(
        {
            "Date": pd.to_datetime(sequences.target_dates),
            "Current_Close": sequences.current_close,
            "Actual_Close": sequences.target_close,
            "Predicted_Close": predicted_close,
        }
    )
    backtest["Residual"] = backtest["Actual_Close"] - backtest["Predicted_Close"]
    comparison = comparison_table(
        sequences.target_close,
        sequences.current_close,
        predicted_close,
        sequences.raw_returns,
    )
    return backtest, comparison


st.title("Google Stock Price Prediction using Simple RNN")
st.caption(
    "A one-step financial time-series demo that uses the previous 10 daily log returns "
    "to estimate the next trading session's closing price."
)
st.warning(DISCLAIMER)

model, feature_scaler, target_scaler, metadata = load_artifacts()
window_size = int(metadata["window_size"])

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose input",
        ["Bundled GOOG sample", "Upload CSV", "Optional live GOOG data"],
    )
    live_period = "5y"
    if source == "Optional live GOOG data":
        live_period = st.selectbox("History period", ["1y", "2y", "5y", "10y"], index=2)

    st.header("Model configuration")
    st.write(f"Input window: **{window_size} trading-day returns**")
    st.write("Forecast horizon: **next trading session**")
    st.write("Primary model: **Simple RNN**")
    st.write("Training target: **next-day log return**")

raw_df = None
source_label = source
if source == "Bundled GOOG sample":
    raw_df = load_sample_data()
elif source == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload a chronological stock CSV containing Date and Close/Adj Close",
        type=["csv"],
    )
    if uploaded is None:
        st.info("Upload a CSV to run the forecast, or choose the bundled sample.")
        st.stop()
    raw_df = pd.read_csv(uploaded)
    source_label = uploaded.name
else:
    with st.spinner("Requesting GOOG data from yfinance..."):
        raw_df, live_error = download_google_data(live_period)
    if live_error:
        st.error(f"Live data could not be loaded: {live_error}")
        st.info("The application remains usable with the bundled sample or an uploaded CSV.")
        st.stop()
    source_label = f"GOOG via yfinance ({live_period})"

try:
    forecast = forecast_from_dataframe(
        raw_df,
        model,
        feature_scaler,
        target_scaler,
        metadata,
    )
except Exception as exc:
    st.error(f"The selected dataset could not be processed: {exc}")
    st.stop()

clean_df = forecast["clean_data"]
target = forecast["selected_target"]

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(clean_df):,}")
col2.metric("Target", target)
col3.metric("Latest close", f"${forecast['latest_close']:,.2f}")
col4.metric("Latest date", clean_df["Date"].iloc[-1].date().isoformat())

with st.expander("Data quality and preview", expanded=False):
    st.json(forecast["quality_report"])
    st.dataframe(clean_df.tail(15), use_container_width=True)

trend_df = clean_df.set_index("Date")[[target]].rename(columns={target: "Closing price"})
st.line_chart(trend_df, height=330)

st.subheader("Next-session forecast")
forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
forecast_col1.metric("Predicted closing price", f"${forecast['predicted_close']:,.2f}")
forecast_col2.metric(
    "Expected movement",
    f"{forecast['predicted_change_pct']:+.2f}%",
)
forecast_col3.metric("Estimated business date", forecast["estimated_forecast_date"])
st.info(forecast["interpretation"])
st.caption(
    "The estimated date uses the next business day and does not adjust for exchange holidays."
)

forecast_download = pd.DataFrame(
    [
        {
            "data_source": source_label,
            "latest_observation_date": clean_df["Date"].iloc[-1].date().isoformat(),
            "estimated_next_business_date": forecast["estimated_forecast_date"],
            "target_column": target,
            "latest_close": forecast["latest_close"],
            "predicted_close": forecast["predicted_close"],
            "predicted_change_pct": forecast["predicted_change_pct"],
            "forecast_horizon": "next trading session",
            "financial_disclaimer": "Educational demonstration only; not financial advice.",
        }
    ]
)
st.download_button(
    "Download forecast CSV",
    data=forecast_download.to_csv(index=False).encode("utf-8"),
    file_name="google_stock_next_session_forecast.csv",
    mime="text/csv",
)

st.subheader("Model evaluation")
if source == "Bundled GOOG sample" and (OUTPUT_DIR / "test_predictions.csv").exists():
    backtest = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", parse_dates=["Date"])
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    st.caption(
        "The bundled evaluation uses the untouched chronological test period created during training."
    )
else:
    backtest, comparison = backtest_loaded_model(
        clean_df,
        target,
        model,
        feature_scaler,
        target_scaler,
        window_size,
    )
    recent_count = max(20, int(len(backtest) * 0.20))
    backtest = backtest.tail(recent_count).reset_index(drop=True)
    st.caption(
        "For uploaded or live data, this is an inference backtest using the saved model; "
        "the model is not retrained in the app."
    )

st.dataframe(
    comparison.style.format(
        {
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "MAPE": "{:.2f}%",
            "R2": "{:.3f}",
            "Directional_Accuracy": "{:.1%}",
        }
    ),
    use_container_width=True,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(backtest["Date"], backtest["Actual_Close"], label="Actual")
ax.plot(backtest["Date"], backtest["Predicted_Close"], label="Simple RNN")
ax.set_title("Actual vs Predicted Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axhline(0, linewidth=1)
ax.scatter(backtest["Date"], backtest["Residual"], s=22)
ax.set_title("Forecast Residuals")
ax.set_xlabel("Date")
ax.set_ylabel("Actual − Predicted (USD)")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.download_button(
    "Download backtest predictions",
    data=backtest.to_csv(index=False).encode("utf-8"),
    file_name="google_stock_backtest_predictions.csv",
    mime="text/csv",
)

with st.expander("How the forecast works"):
    st.markdown(
        f"""
1. Parse and sort the stock data chronologically.
2. Remove invalid rows and duplicate trading dates.
3. Convert closing prices into daily log returns.
4. Scale returns using statistics learned from the training period only.
5. Use the previous **{window_size}** return observations as the Simple RNN input.
6. Predict the next-day log return and reconstruct the next closing price.
7. Compare the model against previous-close and five-day mean-return baselines.
        """
    )

st.warning(DISCLAIMER)
