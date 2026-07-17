# Electricity Consumption Forecasting using a Simple RNN

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-SimpleRNN-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Electricity RNN CI](https://github.com/unit-mole/simple-rnn-projects/actions/workflows/electricity-rnn-ci.yml/badge.svg)](https://github.com/unit-mole/simple-rnn-projects/actions/workflows/electricity-rnn-ci.yml)

An end-to-end electricity-demand forecasting project that uses a trainable
**Simple Recurrent Neural Network** to learn sequential consumption patterns and
forecast near-term electricity usage. The project combines leakage-aware
time-series preprocessing, chronological model validation, calendar feature
engineering, transparent baseline comparison, residual diagnostics, recursive
multi-hour forecasting, saved inference artifacts, automated testing, and a
deployed Streamlit application.

**Status:** Portfolio-ready  
**Live demo:** [Open the Streamlit application](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)  
**Primary stack:** Python · Keras · TensorFlow · pandas · scikit-learn · Streamlit

---

## Business Problem

Electricity demand changes by hour, day, season, and operating context.
Organizations need reliable near-term forecasts to support capacity planning,
peak-demand monitoring, maintenance scheduling, energy purchasing, and
operational decision-making.

This project answers:

> Given historical electricity-consumption data, what will near-term
> electricity demand look like?

The deployed application produces:

- **Forecasted electricity consumption**
- **Selectable 1–48 hour forecast horizon**
- **Expected demand trend**
- **Change versus recent electricity usage**
- **Expected peak-demand value and timestamp**
- **Model error metrics**
- **Business interpretation**
- **Downloadable forecast CSV**

---

## Project Highlights

- End-to-end electricity time-series workflow from validation to deployment
- Genuine trainable Keras `SimpleRNN` model
- Previous 24 hourly observations used to predict the next hour
- Chronological 70% / 15% / 15% train, validation, and test split
- Training-only target scaling to reduce data leakage
- Calendar and cyclical time-feature engineering
- Naive, seasonal-naive, and moving-average baseline comparison
- Recursive 1–48 hour forecasting
- MAE, RMSE, MAPE, sMAPE, R², and residual analysis
- CSV upload and preloaded sample-data workflows
- Downloadable forecast output
- Modular source code, tests, CI workflow, and Streamlit deployment

---

## Application Preview

The README uses four focused application screenshots to avoid repeating the
model-output graphs already stored in `outputs/`.

### 1. Application overview

The main view combines forecast controls, data-quality checks, the prepared
electricity history, and the first forecast summary.

![Electricity Consumption Forecasting application overview](images/01_streamlit_application_overview.png)

### 2. Forecast results and business interpretation

The forecast view presents the selected horizon, average forecast, expected
trend, change versus recent usage, forecast trajectory, and operational
interpretation.

![Forecast results and business interpretation](images/02_forecast_results_and_interpretation.png)

### 3. Model-performance summary

The model-performance dashboard reports the held-out regression metrics and
compares the Simple RNN with transparent forecasting baselines.

![Model-performance summary](images/03_model_performance_summary.png)

### 4. CSV upload workflow

Users can upload compatible time-series data, select the timestamp and
consumption columns, preview the data, and generate a forecast.

<details>
<summary><strong>View the CSV upload workflow</strong></summary>

![CSV upload workflow](images/04_csv_upload_workflow.png)

</details>

---

## Project Status and Honest Scope

This is a complete, deployable portfolio prototype built around a deterministic
**synthetic hourly electricity-consumption dataset**. The synthetic dataset is
privacy-safe and allows the entire preprocessing, training, evaluation,
persistence, and deployment workflow to be reproduced without exposing company
or customer information.

The committed metrics demonstrate that the project works end to end, but they
should not be interpreted as production grid performance. Operational use would
require retraining and backtesting on governed real electricity-consumption
data, domain review, uncertainty estimation, monitoring, and business-specific
error tolerances.

---

## Dataset

The repository includes:

```text
data/sample_input.csv
```

The sample contains **2,880 hourly observations**.

| Column | Description |
|---|---|
| `timestamp` | Hourly observation timestamp |
| `consumption_kwh` | Electricity-consumption target |
| `temperature_c` | Demonstration weather context |
| `humidity_pct` | Demonstration weather context |
| `source_type` | Explicit synthetic-data provenance |

| Dataset detail | Value |
|---|---:|
| Total observations | 2,880 |
| Frequency | Hourly |
| Training observations | 2,015 |
| Validation observations | 433 |
| Test observations | 432 |
| Invalid timestamps removed | 0 |
| Duplicate timestamps aggregated | 0 |
| Missing target values filled | 0 |
| Outlier rows flagged | 0 |
| Private operational data | None |

The current saved RNN uses lagged consumption and calendar variables. The
weather columns are retained for future multivariate-model extensions.

A helper script is also included for a documented real-data extension:

```text
scripts/download_tetouan_data.py
```

See [`data/README_data.md`](data/README_data.md) for data placement, provenance,
and repository-safety guidance.

---

## Time-Series Preprocessing

The preprocessing workflow applies the following controls:

| Control | Implementation |
|---|---|
| Timestamp parsing | Invalid timestamps are coerced and reported |
| Chronological ordering | Data is sorted before sequence generation |
| Duplicate timestamps | Duplicate observations are aggregated by mean |
| Missing intervals | The series is reindexed to the configured frequency |
| Missing target values | Time interpolation followed by forward/backward fill |
| Invalid targets | Coerced to missing and handled through the documented fill logic |
| Outlier review | IQR-based flags are retained for audit rather than silently deleting rows |
| Training order | RNN training uses `shuffle=False` |
| Leakage prevention | The scaler is fitted only on training target values |

A structured quality report is saved in:

```text
outputs/data_quality_report.json
```

---

## Feature Engineering

Nine features are supplied at each time step:

| Feature | Purpose |
|---|---|
| `consumption_scaled` | Historical electricity-consumption signal |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day representation |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical weekday representation |
| `month_sin`, `month_cos` | Cyclical month representation |
| `weekend_flag` | Weekend-demand indicator |
| `peak_flag` | 17:00–21:00 peak-period indicator |

Cyclical encoding preserves relationships such as hour 23 being adjacent to
hour 0 rather than treating them as distant numeric endpoints.

---

## Technical Workflow

```text
Electricity CSV or privacy-safe sample
                │
                ▼
Schema validation and timestamp parsing
                │
                ▼
Chronological sorting and duplicate aggregation
                │
                ▼
Frequency regularization and missing-value treatment
                │
                ▼
Outlier audit and electricity-trend analysis
                │
                ▼
Calendar and cyclical feature engineering
                │
                ▼
Chronological 70% / 15% / 15% split
                │
                ▼
Training-only target scaling
                │
                ▼
24-step sequence generation
                │
                ▼
Simple RNN training with shuffling disabled
                │
                ▼
Held-out test evaluation and baseline comparison
                │
                ▼
Saved model, scaler, metadata, metrics, and plots
                │
                ▼
Recursive future forecast and Streamlit deployment
```

---

## Sequence Design

The supervised-learning setup is:

```text
Previous 24 hourly time steps
            ↓
Predict the next 1 hourly consumption value
```

Formally:

```text
X(t-23), X(t-22), ..., X(t)  →  y(t+1)
```

The Keras input shape is:

```text
(samples, 24 time steps, 9 features)
```

Validation and test sequences may use earlier observations as historical
context, while their target timestamps remain inside the correct chronological
partition.

---

## Simple RNN Architecture

```text
Input: 24 time steps × 9 features
                ↓
SimpleRNN(64, activation="tanh")
                ↓
Dropout(0.05)
                ↓
Dense(32, activation="relu")
                ↓
Dense(1, activation="linear")
                ↓
Next-hour electricity forecast
```

Training uses:

- Adam optimizer
- Mean squared error loss
- Mean absolute error tracking
- Early stopping
- `ReduceLROnPlateau`
- `shuffle=False`
- Best-weight restoration

Simple RNN remains the primary model because this project is the first case
study in the Simple RNN portfolio series. GRU and LSTM comparisons are planned
as future benchmarking extensions.

---

## Forecasting Approach

### Single-step training objective

The trained model predicts the next one-hour consumption value from the
previous 24 hourly observations.

### Recursive multi-hour application forecast

The Streamlit application supports a selectable **1–48 hour horizon**:

1. Predict the next hour.
2. Append the prediction to the active sequence.
3. Generate the next timestamp and its calendar features.
4. Repeat until the requested horizon is complete.

Recursive forecasting is practical for demonstration, but one-step errors and
uncertainty can accumulate as the requested horizon becomes longer.

---

## Supplied-Model Test Results

| Metric | Result | Interpretation |
|---|---:|---|
| MAE | **2.765 kWh** | Average absolute forecast error |
| RMSE | **3.429 kWh** | Penalizes larger forecast misses |
| MAPE | **2.36%** | Average percentage forecast error |
| sMAPE | **2.40%** | Symmetric percentage forecast error |
| R² | **0.938** | Share of held-out variation explained |
| Mean residual | **2.011 kWh** | Positive value indicates mild underprediction |

The results apply only to the committed synthetic demonstration split.

---

## Baseline Comparison

A recurrent model should outperform transparent forecasting rules before its
additional complexity is justified.

| Model | MAE (kWh) | RMSE (kWh) | MAPE | R² |
|---|---:|---:|---:|---:|
| **Simple RNN** | **2.765** | **3.429** | **2.36%** | **0.938** |
| Naive previous value | 3.690 | 4.488 | 3.17% | 0.893 |
| Seasonal naive — previous 24-hour value | 4.350 | 5.108 | 3.72% | 0.862 |
| Moving average — previous 24 hours | 11.656 | 13.152 | 10.08% | 0.085 |

The Simple RNN reduced RMSE by approximately **23.6%** compared with the
previous-value baseline on the included test split.

---

## Model Diagnostics

### Actual vs. predicted electricity consumption

The held-out comparison shows how closely the Simple RNN follows observed
electricity demand.

![Actual versus predicted electricity consumption](outputs/actual_vs_predicted.png)

### Next 24-hour forecast

The saved pipeline produces a practical future-horizon forecast.

![Next 24-hour electricity forecast](outputs/forecast_plot.png)

### Residual analysis

Residuals reveal systematic underprediction, overprediction, and changes in
error behavior across the prediction range.

![Residual analysis](outputs/residual_plot.png)

### Training and validation loss

The training history supports review of convergence and potential overfitting.

![Training and validation loss](outputs/training_curve.png)

---

## Business Interpretation

The inference pipeline compares the future average with the recent 24-hour
average and identifies the expected peak.

Example output:

```text
Forecast Horizon: Next 24 hours
Average Forecast: 122.41 kWh
Trend: Increasing
Change vs Recent: +1.8%
Business Interpretation:
Higher near-term demand is expected. Review available capacity,
peak-demand monitoring, and operational readiness.
```

Forecast error has different operational implications:

- **Underprediction** may lead to insufficient capacity preparation.
- **Overprediction** may contribute to unnecessary reserve allocation or cost.
- **Peak-hour error** may be more important than average error.
- **Long recursive horizons** should be interpreted with greater caution.

---

## Streamlit Application

The deployed application supports:

- Privacy-safe preloaded sample data
- Compatible CSV upload
- Timestamp-column selection
- Consumption-column selection
- Uploaded-data preview
- Time-series data-quality checks
- Historical electricity-trend visualization
- Selectable 1–48 hour forecast horizon
- Forecast summary cards
- Expected peak and trend interpretation
- Model-performance and baseline tables
- Actual-versus-predicted and residual charts
- Downloadable forecast CSV

**Live application:**  
[Open Electricity Consumption Forecasting](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

**Streamlit entrypoint:**

```text
01-electricity-consumption-forecasting/app/streamlit_app.py
```

---

## Model Artifacts

| Artifact | Purpose |
|---|---|
| `models/electricity_rnn_model.keras` | Trained Simple RNN used by the application |
| `models/scaler.pkl` | Target scaler fitted only on training data |
| `models/model_metadata.json` | Feature order, sequence design, split strategy, metrics, and provenance |
| `outputs/model_metrics.json` | Held-out Simple RNN metrics |
| `outputs/baseline_comparison.csv` | RNN and baseline comparison |
| `outputs/test_predictions.csv` | Timestamped actual values, predictions, residuals, and errors |
| `outputs/next_24_hour_forecast.csv` | Example future forecast |

---

## Project Structure

```text
simple-rnn-projects/
├── .github/
│   └── workflows/
│       └── electricity-rnn-ci.yml
│
└── 01-electricity-consumption-forecasting/
    ├── .streamlit/
    │   └── config.toml
    ├── app/
    │   ├── requirements.txt
    │   └── streamlit_app.py
    ├── archive/
    │   └── original_streamlit_app.py
    ├── data/
    │   ├── README_data.md
    │   └── sample_input.csv
    ├── images/
    │   ├── 01_streamlit_application_overview.png
    │   ├── 02_forecast_results_and_interpretation.png
    │   ├── 03_model_performance_summary.png
    │   └── 04_csv_upload_workflow.png
    ├── models/
    │   ├── electricity_rnn_model.keras
    │   ├── model_metadata.json
    │   └── scaler.pkl
    ├── notebooks/
    │   ├── electricity_consumption_forecasting.ipynb
    │   └── archive/
    │       └── Code_original.ipynb
    ├── outputs/
    │   ├── actual_vs_predicted.png
    │   ├── baseline_comparison.csv
    │   ├── consumption_trend.png
    │   ├── data_quality_report.json
    │   ├── error_distribution.png
    │   ├── forecast_plot.png
    │   ├── model_metrics.json
    │   ├── next_24_hour_forecast.csv
    │   ├── residual_plot.png
    │   ├── test_predictions.csv
    │   ├── training_curve.png
    │   └── training_history.csv
    ├── scripts/
    │   └── download_tetouan_data.py
    ├── src/
    │   ├── config.py
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   ├── forecasting_pipeline.py
    │   ├── model_evaluation.py
    │   ├── model_training.py
    │   ├── sequence_generation.py
    │   └── visualization.py
    ├── tests/
    │   ├── conftest.py
    │   ├── test_preprocessing.py
    │   └── test_sequences.py
    ├── .gitignore
    ├── Dockerfile
    ├── LICENSE
    ├── PROJECT_AUDIT.md
    ├── README.md
    ├── README_HOSTING.md
    ├── requirements-dev.txt
    ├── requirements.txt
    ├── run_local.bat
    ├── run_local.sh
    └── train_model.py
```

---

## Run Locally

Use Python 3.12 to match the tested local and deployment environments.

### Windows Command Prompt

Clone the repository and enter the project folder:

```bat
git clone https://github.com/unit-mole/simple-rnn-projects.git

cd simple-rnn-projects\01-electricity-consumption-forecasting
```

Create and activate the virtual environment:

```bat
python -m venv "%USERPROFILE%\venvs\simple-rnn"

call "%USERPROFILE%\venvs\simple-rnn\Scripts\activate.bat"
```

A short environment path is recommended on Windows to avoid long-path issues
with deeply nested dependency folders.

Install the project and development dependencies:

```bat
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements-dev.txt
```

Run the automated tests:

```bat
python -m pytest -q

python -m compileall src app scripts tests train_model.py
```

Launch the Streamlit application:

```bat
python -m streamlit run app\streamlit_app.py
```

Open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

### Future local runs

```bat
cd simple-rnn-projects\01-electricity-consumption-forecasting

call "%USERPROFILE%\venvs\simple-rnn\Scripts\activate.bat"

python -m streamlit run app\streamlit_app.py
```

---

## Optional Retraining

The included saved model runs without retraining.

Retrain on the included demonstration sample:

```bat
python train_model.py --epochs 60 --batch-size 64
```

Retrain on a compatible hourly CSV:

```bat
python train_model.py ^
  --data data\your_dataset.csv ^
  --timestamp-column timestamp ^
  --target-column consumption_kwh ^
  --frequency h
```

Retraining overwrites the model artifacts in `models/` and regenerates the
evaluation outputs in `outputs/`.

---

## Deployment

The application is deployed on Streamlit Community Cloud and connected directly
to the `main` branch of this GitHub repository.

**Live application:**  
[Open Electricity Consumption Forecasting](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

**Repository:**

```text
unit-mole/simple-rnn-projects
```

**Branch:**

```text
main
```

**Streamlit entrypoint:**

```text
01-electricity-consumption-forecasting/app/streamlit_app.py
```

Changes pushed to the relevant project files on the `main` branch automatically
trigger a Streamlit application update.

See [`README_HOSTING.md`](README_HOSTING.md) for deployment configuration,
maintenance instructions, and troubleshooting guidance.

---

## Data and Repository Safety

- The committed demonstration data is synthetic and privacy-safe.
- No company, customer, billing, or grid-operational data is included.
- Only a small sample dataset is committed for testing and demonstration.
- Virtual environments, caches, local logs, private data, and secrets are excluded through `.gitignore`.
- Streamlit secrets must not be committed to GitHub.
- The saved model, scaler, and metadata are required for inference and should remain under `models/`.

---

## Original-Code Improvements

The original supplied implementation is preserved under `archive/`. The
portfolio version improves it by:

- Replacing fixed random recurrent weights plus ridge regression with a
  trainable Keras `SimpleRNN`
- Removing the misleading treatment of a transformer oil-temperature target as
  electricity consumption
- Adding chronological train, validation, and test partitions
- Fitting target scaling only on the training partition
- Adding transparent timestamp, duplicate, missing-value, and outlier handling
- Adding known-at-forecast-time calendar features
- Adding naive, seasonal-naive, and moving-average baselines
- Adding MAE, RMSE, MAPE, sMAPE, R², and residual diagnostics
- Adding reusable model persistence and forecasting inference
- Adding a recruiter-friendly Streamlit application
- Adding tests, CI, hosting documentation, and portfolio screenshots

See [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) for the detailed technical audit.

---

## Known Limitations

- The committed training data is synthetic, which limits external validity.
- Recursive forecasts can accumulate one-step prediction errors.
- The application does not currently provide prediction intervals.
- Weather columns exist in the sample but are not yet used by the saved model.
- Holidays, tariffs, outages, and operating events are not modeled.
- A single chronological split is less robust than rolling-origin backtesting.
- The saved metrics should not be generalized to production electricity systems.
- Production use would require governed real data, domain validation, monitoring, and retraining.

---

## Future Improvements

- Retrain and validate on approved real electricity-consumption data
- Add forecasted weather, holidays, tariffs, outages, and operating events
- Add rolling-origin backtesting across multiple temporal folds
- Compare Simple RNN with GRU, LSTM, temporal CNN, XGBoost, and statistical models
- Implement direct multi-output forecasting rather than recursive prediction
- Add prediction intervals and uncertainty calibration
- Add peak-weighted business metrics
- Add model and data-drift monitoring
- Add scheduled retraining and model-version tracking
- Add a model-artifact smoke test to CI

---

## Skills Demonstrated

`Time-Series Forecasting` · `Simple RNN` · `Recurrent Neural Networks` ·
`Chronological Validation` · `Data-Leakage Prevention` · `Sequence Generation` ·
`Calendar Feature Engineering` · `Baseline Benchmarking` · `Regression Metrics` ·
`Residual Analysis` · `Recursive Forecasting` · `TensorFlow` · `Keras` ·
`scikit-learn` · `Streamlit` · `Model Persistence` · `Testing` · `CI/CD` ·
`Deployment` · `Business Interpretation`

---

## Portfolio Description

**One-line description**

> Built and deployed a Simple RNN electricity-demand forecasting pipeline with
> chronological validation, transparent baseline comparison, recursive
> multi-hour forecasts, and a Streamlit application.

**Pinned-repository description**

> Simple RNN portfolio featuring an end-to-end electricity forecasting project
> with time-series validation, training-only scaling, sequence generation,
> baseline benchmarking, residual diagnostics, CSV upload, testing, CI/CD, and
> Streamlit deployment.

This project supports a transition from Quality Data Scientist to broader Data
Science, Machine Learning, Applied AI, Analytics Engineering, Business
Intelligence, and Quality Analytics roles by demonstrating how sequential
operational data can be converted into a reproducible predictive workflow and
decision-oriented application.

---

## Responsible Use

This repository is a portfolio demonstration. The included model is not
validated for production grid operations, electricity billing, energy trading,
or safety-critical control. Forecasts should not be used for operational
decisions without real-data validation, uncertainty analysis, monitoring, and
domain review.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
