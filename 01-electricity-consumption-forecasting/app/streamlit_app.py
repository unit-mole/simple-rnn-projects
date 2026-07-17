"""Interactive Streamlit demo for Simple RNN electricity forecasting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, OUTPUT_DIR, TARGET_COLUMN  # noqa: E402
from src.data_preprocessing import prepare_time_series  # noqa: E402
from src.forecasting_pipeline import ElectricityForecastingPipeline  # noqa: E402


st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the trained Simple RNN model...")
def load_pipeline() -> ElectricityForecastingPipeline:
    return ElectricityForecastingPipeline(MODEL_DIR)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "sample_input.csv")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_header() -> None:
    st.title("⚡ Electricity Consumption Forecasting")
    st.caption("Simple RNN • Chronological time-series validation • Recursive future forecast")
    st.info(
        "Portfolio demonstration. The included model is trained on a safe synthetic hourly sample. "
        "Retrain with the documented UCI Tetouan electricity dataset or your own approved data before operational use."
    )


def select_data() -> tuple[pd.DataFrame | None, str | None, str | None]:
    mode = st.sidebar.radio("Input method", ["Use demo sample", "Upload CSV"])
    if mode == "Use demo sample":
        return load_sample_data(), "timestamp", "consumption_kwh"

    uploaded = st.sidebar.file_uploader("Upload electricity history", type=["csv"])
    if uploaded is None:
        st.warning("Upload a CSV file to continue, or select the demo sample.")
        return None, None, None
    raw = pd.read_csv(uploaded)
    st.subheader("Uploaded data preview")
    st.dataframe(raw.head(20), use_container_width=True)
    timestamp_column = st.sidebar.selectbox("Timestamp column", raw.columns.tolist())
    numeric_candidates = raw.select_dtypes(include="number").columns.tolist()
    if not numeric_candidates:
        st.error("The uploaded file needs at least one numeric consumption column.")
        return None, None, None
    target_column = st.sidebar.selectbox("Consumption column", numeric_candidates)
    return raw, timestamp_column, target_column


def show_saved_performance() -> None:
    st.subheader("Saved model performance")
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    comparison_path = OUTPUT_DIR / "baseline_comparison.csv"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("MAE", f"{metrics['mae']:.3f} kWh")
        cols[1].metric("RMSE", f"{metrics['rmse']:.3f} kWh")
        cols[2].metric("MAPE", f"{metrics['mape_pct']:.2f}%")
        cols[3].metric("R²", f"{metrics['r2']:.3f}")
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
    image_columns = st.columns(2)
    visuals = [
        ("actual_vs_predicted.png", "Actual vs predicted"),
        ("residual_plot.png", "Residual analysis"),
        ("training_curve.png", "Training curve"),
        ("error_distribution.png", "Error distribution"),
    ]
    for index, (filename, caption) in enumerate(visuals):
        path = OUTPUT_DIR / filename
        if path.exists():
            image_columns[index % 2].image(str(path), caption=caption, use_container_width=True)


def main() -> None:
    render_header()
    pipeline = load_pipeline()
    st.sidebar.header("Forecast controls")
    horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 48, 24)
    raw, timestamp_column, target_column = select_data()

    tabs = st.tabs(["Forecast", "Model performance", "Project details"])
    with tabs[0]:
        if raw is None or timestamp_column is None or target_column is None:
            st.stop()
        try:
            cleaned, report = prepare_time_series(
                raw,
                timestamp_column=timestamp_column,
                target_column=target_column,
                frequency=pipeline.frequency,
            )
        except Exception as exc:
            st.error(f"Data preparation failed: {exc}")
            st.stop()

        cleaned = cleaned.rename(columns={target_column: TARGET_COLUMN})
        st.subheader("Prepared electricity history")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Prepared rows", f"{len(cleaned):,}")
        metric_columns[1].metric("Frequency", report.inferred_frequency)
        metric_columns[2].metric("Missing values filled", report.missing_target_values_filled)
        metric_columns[3].metric("Outliers flagged", report.outlier_rows_flagged)
        st.line_chart(cleaned.set_index("timestamp")[TARGET_COLUMN])
        with st.expander("Data-quality details"):
            st.json(report.to_dict())
            st.dataframe(cleaned.tail(30), use_container_width=True)

        if len(cleaned) < pipeline.lookback:
            st.error(f"At least {pipeline.lookback} rows are required for this model.")
            st.stop()

        training_min = float(pipeline.scaler.data_min_[0])
        training_max = float(pipeline.scaler.data_max_[0])
        current_min = float(cleaned[TARGET_COLUMN].min())
        current_max = float(cleaned[TARGET_COLUMN].max())
        if current_min < training_min * 0.7 or current_max > training_max * 1.3:
            st.warning(
                "The uploaded consumption range differs materially from the model's training range. "
                "For reliable results, retrain the model on that dataset using train_model.py."
            )

        forecast = pipeline.forecast(cleaned[["timestamp", TARGET_COLUMN]], horizon=horizon)
        interpretation = pipeline.interpret(cleaned, forecast)
        st.subheader("Forecast output")
        cards = st.columns(4)
        cards[0].metric("Forecast horizon", f"Next {horizon} hours")
        cards[1].metric("Average forecast", f"{interpretation['forecast_mean_kwh']:.2f} kWh")
        cards[2].metric("Trend", str(interpretation["trend"]))
        cards[3].metric(
            "Change vs recent",
            f"{interpretation['change_vs_recent_pct']:+.1f}%",
        )

        combined = pd.concat(
            [
                cleaned[["timestamp", TARGET_COLUMN]].tail(7 * 24).rename(
                    columns={TARGET_COLUMN: "consumption_kwh"}
                ).assign(series="Recent history"),
                forecast.rename(
                    columns={"forecasted_consumption_kwh": "consumption_kwh"}
                )[["timestamp", "consumption_kwh"]].assign(series="Forecast"),
            ],
            ignore_index=True,
        )
        st.line_chart(combined, x="timestamp", y="consumption_kwh", color="series")
        st.success(str(interpretation["business_interpretation"]))
        st.caption(
            f"Expected peak: {interpretation['peak_consumption_kwh']:.2f} kWh at "
            f"{interpretation['peak_timestamp']}"
        )
        st.dataframe(forecast, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            data=csv_bytes(forecast),
            file_name="electricity_consumption_forecast.csv",
            mime="text/csv",
        )

    with tabs[1]:
        show_saved_performance()

    with tabs[2]:
        st.subheader("How the model works")
        st.markdown(
            f"""
- **Input window:** previous **{pipeline.lookback} hourly steps**.
- **Target:** the next one-hour electricity-consumption value.
- **Architecture:** Keras `SimpleRNN` → Dropout → Dense → linear regression output.
- **Leakage control:** chronological 70/15/15 split and target scaler fitted only on training rows.
- **App forecast:** recursive one-step predictions for the selected future horizon.
- **Model limitation:** the included artifact is a portfolio demo model, not a production energy-control system.
"""
        )
        st.json(pipeline.metadata)


if __name__ == "__main__":
    main()
