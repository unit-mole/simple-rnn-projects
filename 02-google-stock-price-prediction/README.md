# Google Stock Price Prediction using Simple RNN

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Keras](https://img.shields.io/badge/Keras-3.15-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b)
![Project Status](https://img.shields.io/badge/Status-Portfolio%20Ready-brightgreen)

A recruiter-friendly financial time-series project that uses a **Simple Recurrent Neural Network (Simple RNN)** to estimate the next Google (`GOOG`) closing price from recent daily return behavior.

> [!IMPORTANT]
> **Financial disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not financial advice. Stock-price predictions are uncertain and should not be used for real investment decisions. Please consult a qualified financial adviser before making investment decisions.

## Live application

**Streamlit demo:** `ADD-LIVE-STREAMLIT-URL-HERE`

The deployed app supports a bundled sample, uploaded CSV files, and optional live `GOOG` data through `yfinance`. The app always remains usable without live API access.

---

## Business question

> Given recent historical Google stock-price behavior, what closing price does a Simple RNN estimate for the next trading session?

The project produces:

- a predicted next-session closing price;
- an estimated percentage movement;
- an actual-versus-predicted test-period comparison;
- MAE, RMSE, MAPE, R², and directional accuracy;
- naive and moving-return baseline comparisons;
- forecast and residual visualizations; and
- downloadable forecast and backtest files through Streamlit.

## Portfolio highlights

- Proper chronological train, validation, and test periods
- Train-only feature and target scaling
- Ten-trading-day input sequences
- One-session forecast horizon
- Return-based modeling to reduce raw-price extrapolation problems
- TensorFlow/Keras Simple RNN as the primary model
- Previous-close and five-day mean-return baselines
- Saved `.keras` model and reusable scalers
- Modular Python package, notebook, tests, and CI workflow
- Streamlit app with sample, upload, and optional live-data modes
- Explicit financial-risk and responsible-use communication

---

## Why the original implementation needed improvement

The supplied project correctly demonstrated a TensorFlow `SimpleRNN`, generated synthetic and real forecasts, exported reports, and included optional `yfinance` support. However, the original real-data chart showed the forecast flattening near the training price range while actual prices continued materially higher.

The original real-data output reported approximately:

| Metric | Original real-data result |
|---|---:|
| MAE | 64.61 |
| RMSE | 79.73 |
| MAPE | 21.78% |
| R² | -0.765 |
| Naive RMSE | 4.56 |

The main technical issue was asking a tanh Simple RNN trained on MinMax-scaled **raw price levels** to extrapolate beyond the price range learned during training. Additional weaknesses included a single 80/20 split, Keras `validation_split` behavior inside the training block, only five default epochs, and synthetic data being positioned as the primary app flow.

### Portfolio-ready corrections

| Original approach | Updated approach |
|---|---|
| Predict scaled raw closing price | Predict next-day log return and reconstruct closing price |
| 30 raw-price observations | 10 daily log-return observations |
| 80/20 split | Chronological 70/15/15 train/validation/test split |
| Validation created inside `model.fit` | Explicit chronological validation period |
| Potential training shuffling | `shuffle=False` |
| Broad synthetic-first application | Real GOOG sample as the default portfolio experience |
| Model fallback could change the project type | Saved Simple RNN is always the primary deployed model |
| No saved deployable model/scalers | `.keras` model and both train-fitted scalers included |
| Weak real test performance | Honest baseline-level next-session forecasting results |

---

## Dataset

The bundled GitHub-safe sample contains **246 chronological observations** with:

| Column | Meaning |
|---|---|
| `Date` | Trading date |
| `Close` | Google closing price used as the source series |

Sample period:

```text
2025-05-07 to 2026-04-29
```

The original project downloaded `GOOG` through `yfinance` and standardized the series to `Date` and `Close`; therefore, `Close` remains the default target in this version. Uploaded datasets may also contain `Adj Close`, Open, High, Low, and Volume, but the saved model currently uses the selected close-price sequence only.

See [`data/README_data.md`](data/README_data.md) for schema and data-safety guidance.

---

## Forecasting design

### Target formulation

The model does not directly predict the next raw price. It predicts:

```text
Next-day log return = log(next close / current close)
```

The closing-price forecast is reconstructed as:

```text
Predicted next close = current close × exp(predicted log return)
```

This formulation is more stable for a rapidly changing price level and makes the input sequence closer to stationary than raw prices.

### Input window and horizon

```text
Input: previous 10 daily log returns
Forecast horizon: next trading session
Output: predicted next-day log return → predicted next closing price
```

### Chronological split

```text
Training:   70%
Validation: 15%
Testing:    15%
```

The data is never randomly shuffled. Feature and target scalers are fit only on the training period and then applied to validation, test, and inference data.

---

## Model architecture

```text
Input: (10 trading days, 1 return feature)
                ↓
SimpleRNN: 16 units, tanh
                ↓
Dropout: 0.05
                ↓
Dense: 8 units, ReLU
                ↓
Dense: 1 linear output
                ↓
Predicted scaled next-day log return
```

Training configuration:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Huber |
| Batch size | 16 |
| Maximum epochs | 80 |
| Early-stopping patience | 10 |
| Sequence shuffling | Disabled |
| Random seed | 42 |

Huber loss is used because it is less sensitive than pure MSE to unusually large daily moves while still penalizing meaningful errors.

---

## Test results

The following results come from the untouched final chronological test period in the bundled sample:

| Model | MAE | RMSE | MAPE | R² | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| **Simple RNN** | **4.324** | **5.586** | **1.40%** | **0.932** | **61.1%** |
| Naive previous close | 4.350 | 5.608 | 1.41% | 0.931 | 0.0%* |
| Five-day mean return | 4.357 | 5.626 | 1.41% | 0.931 | 52.8% |

\*The previous-close baseline always predicts zero movement, so its directional score is zero under the strict up/down comparison used here.

The Simple RNN reduces RMSE by approximately **0.40%** versus the previous-close baseline. This is a deliberately honest result: next-day stock forecasting is difficult, and a naive previous-close forecast is a strong benchmark.

### Metric interpretation

- **MAE:** average absolute closing-price error in dollars.
- **RMSE:** gives larger misses more weight than MAE.
- **MAPE:** average percentage error relative to actual closing prices.
- **R²:** describes how closely predicted price variation follows the test-period price variation.
- **Directional accuracy:** percentage of sessions where the predicted up/down direction matches the actual movement.

---

## Saved outputs

| File | Purpose |
|---|---|
| `outputs/stock_price_trend.png` | Historical closing-price trend |
| `outputs/actual_vs_predicted.png` | Test-period actual versus Simple RNN prediction |
| `outputs/forecast_plot.png` | Latest observed prices plus the next-session forecast |
| `outputs/baseline_comparison.png` | RMSE comparison across the RNN and baselines |
| `outputs/residual_plot.png` | Residual behavior over time |
| `outputs/training_curve.png` | Training and validation Huber loss |
| `outputs/model_metrics.json` | Test metrics |
| `outputs/model_comparison.csv` | Baseline comparison table |
| `outputs/test_predictions.csv` | Detailed test-period predictions |
| `outputs/next_day_forecast.json` | Demonstration next-session forecast |
| `models/google_stock_rnn_model.keras` | Saved Keras model |
| `models/feature_scaler.joblib` | Training-fitted return scaler |
| `models/target_scaler.joblib` | Training-fitted target-return scaler |
| `models/model_metadata.json` | Architecture, split, target, and run metadata |

### Demonstration forecast from the bundled run

```text
Latest bundled close:      $349.44
Predicted next close:      $350.35
Estimated movement:        +0.26%
Forecast horizon:          next trading session
```

This historical demonstration output is not a current market prediction and is not financial advice.

---

## Streamlit application

The app supports:

1. **Bundled GOOG sample** — reliable offline demonstration.
2. **CSV upload** — Date plus Close/Adj Close input.
3. **Optional live GOOG data** — requested through `yfinance` with graceful failure handling.

It displays:

- cleaned data and a quality report;
- closing-price trend;
- selected target column;
- fixed 10-day return window and next-session horizon;
- predicted closing price and expected movement;
- baseline comparison table;
- actual-versus-predicted chart;
- residual chart;
- forecast and backtest downloads; and
- financial disclaimers at the top and bottom.

### Recommended screenshots

Save these after deployment:

```text
images/01_application_overview.png
images/02_data_quality_and_price_trend.png
images/03_next_session_forecast.png
images/04_model_evaluation_and_baselines.png
images/05_csv_upload_workflow.png
```

---

## Project structure

```text
02-google-stock-price-prediction/
│
├── README.md
├── README_HOSTING.md
├── train_model.py
├── requirements.txt
├── .gitignore
├── .python-version
│
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── data/
│   ├── sample_google_stock_data.csv
│   └── README_data.md
│
├── notebooks/
│   └── google_stock_price_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── sequence_generation.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── forecasting_pipeline.py
│   └── visualization.py
│
├── models/
│   ├── google_stock_rnn_model.keras
│   ├── feature_scaler.joblib
│   ├── target_scaler.joblib
│   └── model_metadata.json
│
├── outputs/
│   ├── stock_price_trend.png
│   ├── actual_vs_predicted.png
│   ├── forecast_plot.png
│   ├── baseline_comparison.png
│   ├── residual_plot.png
│   ├── training_curve.png
│   ├── training_history.csv
│   ├── test_predictions.csv
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   └── next_day_forecast.json
│
├── images/
│   └── .gitkeep
│
└── tests/
    ├── test_data_preprocessing.py
    ├── test_sequence_generation.py
    └── test_model_evaluation.py
```

The repository-level package also includes:

```text
.github/workflows/google-stock-rnn-ci.yml
```

---

## Run locally on Windows CMD

From the repository root:

```bat
cd /d "%USERPROFILE%\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\simple-rnn-projects\02-google-stock-price-prediction"
python -m venv "%USERPROFILE%\venvs\simple-rnn-google-stock"
call "%USERPROFILE%\venvs\simple-rnn-google-stock\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

The terminal will provide a local browser address, normally:

```text
http://localhost:8501
```

### Optional retraining

```bat
python train_model.py
```

Retrain on another compatible CSV:

```bat
python train_model.py --data "data\your_google_stock_data.csv" --target "Close" --window 10
```

---

## Testing

Run lightweight unit tests:

```bat
python -m pytest -q
```

The CI workflow also compiles the source and app files to catch syntax errors.

---

## Deployment

Streamlit Community Cloud is the recommended host because this is a Python-native interactive ML demonstration and the platform can deploy directly from GitHub.

Use:

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
App file: 02-google-stock-price-prediction/app/streamlit_app.py
Python: 3.12
```

The app-specific `requirements.txt` is located beside the entry point so the deployment service can resolve the correct pinned dependencies.

See [`README_HOSTING.md`](README_HOSTING.md) for the complete deployment workflow.

---

## Limitations

- Historical prices alone do not capture news, earnings, interest rates, macroeconomic conditions, sentiment, or market microstructure.
- The bundled sample is small and intentionally GitHub-safe.
- The model produces a one-session point forecast, not a calibrated probability distribution.
- The estimated next date uses a business-day offset and does not account for exchange holidays.
- Baseline-level improvement should not be interpreted as evidence of a profitable trading strategy.
- Live `yfinance` access may occasionally fail because of internet, rate-limit, or upstream-service conditions.
- The model was trained on `GOOG` closing-price returns and should not be assumed to generalize to unrelated assets.

## Future improvements

- Train on a longer historical period with rolling-origin backtesting.
- Add adjusted-close, volume, volatility, and OHLC-derived features.
- Compare Simple RNN with LSTM, GRU, temporal convolution, and gradient-boosting baselines.
- Add prediction intervals using quantile loss, bootstrap methods, or conformal forecasting.
- Add walk-forward retraining instead of a single fixed split.
- Evaluate transaction costs and economic value separately from forecast error.
- Add holiday-aware exchange calendars.
- Monitor data and return-distribution drift after deployment.

---

## Skills demonstrated

- Financial time-series preprocessing
- Date parsing, sorting, deduplication, and quality validation
- Log-return transformation
- Chronological model validation
- Train-only scaling and leakage prevention
- Sequence/window generation
- Simple RNN regression with TensorFlow/Keras
- Huber-loss optimization and early stopping
- Baseline forecasting
- Regression and directional evaluation
- Residual analysis
- Model serialization and inference
- Streamlit deployment
- Unit testing and GitHub Actions CI
- Responsible financial-model communication

## Recruiter-friendly descriptions

### One-line project description

> Built and deployed a leakage-controlled Simple RNN that converts recent Google stock returns into a next-session closing-price forecast and benchmarks performance against naive time-series baselines.

### GitHub pinned-repository description

> End-to-end Google stock forecasting portfolio project featuring chronological validation, return-based Simple RNN modeling, baseline comparison, saved artifacts, tests, and an interactive Streamlit demo.

### Resume bullet

> Developed a modular financial time-series forecasting pipeline using a Keras Simple RNN, chronological train/validation/test splitting, train-only scaling, baseline evaluation, residual diagnostics, model persistence, and Streamlit deployment.

## Career positioning

This project supports a transition from Quality Data Scientist work into broader Data Science, ML, and Applied AI roles by demonstrating the ability to:

- diagnose a technically weak first model rather than presenting misleading results;
- redesign the target formulation based on time-series behavior;
- build reusable preprocessing, training, and evaluation modules;
- compare a neural network honestly against simple baselines; and
- convert experimentation into a tested, deployable portfolio application.

---

## Responsible use

This repository is a learning and portfolio artifact. It does not recommend buying, selling, or holding any security. Forecast error metrics do not establish profitability, and past price behavior does not guarantee future performance.

## Author

**Anmol Tripathi**  
GitHub: [`unit-mole`](https://github.com/unit-mole)
