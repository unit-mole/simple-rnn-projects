# Data Guide

## Bundled sample

`sample_google_stock_data.csv` contains 246 chronological Google (`GOOG`) closing-price observations from the original project output. It is intentionally small enough for GitHub and Streamlit Community Cloud.

Columns:

| Column | Description |
|---|---|
| `Date` | Trading date |
| `Close` | Google closing price used by this project |

The sample is provided only to make the portfolio demo reproducible. It is not intended to be a complete market-data archive.

## Supported uploaded schema

The Streamlit app accepts CSV files containing:

- a date field such as `Date`, `Datetime`, or `Timestamp`; and
- `Close`, `Adj Close`, or another close-like numeric field.

Optional OHLCV columns may remain in the file. The current trained model uses only the selected close-price series after converting it into daily log returns.

## Optional live data

The app can request recent `GOOG` history with `yfinance`. Live access is optional and protected by error handling; the bundled sample remains available when network access fails.

## Data safety

Do not commit licensed, private, employer-owned, or very large datasets. Place large local downloads under `data/raw/`; that location is excluded by `.gitignore`.

## Replacing the training sample

```bat
python train_model.py --data "data\your_google_stock_data.csv" --target "Close" --window 10
```

Retraining overwrites the saved model, scalers, metrics, predictions, and plots with artifacts based on the supplied data.

## Financial disclaimer

This dataset and project are for education and portfolio demonstration only. They are not financial advice and must not be used as the sole basis for investment decisions.
