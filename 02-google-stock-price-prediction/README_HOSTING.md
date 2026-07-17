# Hosting Guide — Google Stock Simple RNN App

## Deployment status

The application is deployed on Streamlit Community Cloud.

**Live application:** [Open the Google Stock Price Prediction application](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/)

## Streamlit configuration

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
Main file path: 02-google-stock-price-prediction/app/streamlit_app.py
Python: 3.12
```

No secrets are required for the bundled sample or the standard optional `yfinance` mode.

## Required repository files

```text
02-google-stock-price-prediction/app/streamlit_app.py
02-google-stock-price-prediction/app/requirements.txt
02-google-stock-price-prediction/data/sample_google_stock_data.csv
02-google-stock-price-prediction/models/google_stock_rnn_model.keras
02-google-stock-price-prediction/models/feature_scaler.joblib
02-google-stock-price-prediction/models/target_scaler.joblib
02-google-stock-price-prediction/models/model_metadata.json
02-google-stock-price-prediction/outputs/test_predictions.csv
02-google-stock-price-prediction/outputs/model_comparison.csv
02-google-stock-price-prediction/src/*.py
```

Do not change the relative folder structure because the app resolves model, data, output, and source paths from the project directory.

## Deployment dependencies

Streamlit reads:

```text
02-google-stock-price-prediction/app/requirements.txt
```

The file pins the deployment versions of TensorFlow, Keras, Streamlit, pandas, NumPy, scikit-learn, Matplotlib, joblib, and `yfinance`.

## Post-deployment checks

1. Open the [live application](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/).
2. Confirm the bundled GOOG sample loads.
3. Verify the historical trend and next-session forecast render.
4. Confirm the metrics and baseline comparison are visible.
5. Download the forecast and backtest CSV files.
6. Upload a compatible CSV containing `Date` and `Close` or `Adj Close`.
7. Test optional live-data mode. A live-data failure should show a controlled message instead of crashing the app.
8. Confirm the financial disclaimer appears in the interface.

## Updating the deployed application

Commits pushed to the relevant files on the `main` branch normally trigger a Streamlit redeployment.

From the repository root:

```bat
set "PATH=%USERPROFILE%\Tools\PortableGit\cmd;%PATH%"
cd /d "%USERPROFILE%\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\simple-rnn-projects"
git add "02-google-stock-price-prediction"
git commit -m "Update Google stock forecasting project"
git push origin main
```

## Troubleshooting

### App cannot find the model or sample

Confirm these paths exist on GitHub:

```text
02-google-stock-price-prediction/models/google_stock_rnn_model.keras
02-google-stock-price-prediction/data/sample_google_stock_data.csv
```

### Live market data fails

Return to the bundled GOOG sample or upload a CSV. The application is intentionally not dependent on live API availability.

### Dependency conflict

Confirm Streamlit is reading:

```text
02-google-stock-price-prediction/app/requirements.txt
```

### Python version mismatch

Redeploy using Python 3.12.

## Public link placement

Use the live URL in:

- the project README;
- the repository About section;
- LinkedIn;
- your resume project section; and
- your portfolio website.

## Security and data guidance

- Do not commit brokerage credentials, API keys, personal portfolio data, employer data, or `.streamlit/secrets.toml`.
- Keep large or licensed datasets outside GitHub.
- Treat optional `yfinance` data as educational and review applicable provider terms.

## Financial disclaimer

This application is for educational and portfolio demonstration purposes only. It is not financial advice and must not be used as the sole basis for an investment decision.
