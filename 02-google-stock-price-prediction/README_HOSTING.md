# Hosting Guide — Google Stock Simple RNN App

## Recommended platform

Use **Streamlit Community Cloud** for this project.

It is the best fit because:

- the application is already written in Streamlit;
- the app can deploy directly from the GitHub repository;
- the saved Keras model and scalers can be loaded without retraining on every visit;
- the bundled sample keeps the demonstration available when live market-data access fails; and
- the resulting public URL can be shared on GitHub, LinkedIn, a resume, or a portfolio website.

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

Do not remove the relative folder structure because the app resolves model, data, output, and source paths from its project directory.

## Deployment dependencies

The app-specific file is:

```text
02-google-stock-price-prediction/app/requirements.txt
```

It contains pinned versions for TensorFlow, Keras, Streamlit, pandas, NumPy, scikit-learn, Matplotlib, joblib, and optional `yfinance` access.

Only one dependency file should be used from the app entry-point directory. Streamlit Community Cloud searches beside the entry-point file before searching the repository root.

## Push the project to GitHub

From the `simple-rnn-projects` repository root:

```bat
set "PATH=%USERPROFILE%\Tools\PortableGit\cmd;%PATH%"
cd /d "%USERPROFILE%\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\simple-rnn-projects"
git add "02-google-stock-price-prediction" ".github\workflows\google-stock-rnn-ci.yml"
git commit -m "Add Google stock Simple RNN forecasting project"
git push origin main
```

Your configured Git identity should remain:

```text
Anmol Tripathi <antripat3@gmail.com>
```

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with the GitHub account that can access `unit-mole/simple-rnn-projects`.
2. Choose **Create app**.
3. Select **Deploy a public app from GitHub**.
4. Configure:

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
Main file path: 02-google-stock-price-prediction/app/streamlit_app.py
```

5. Open **Advanced settings** and select Python **3.12**.
6. No secrets are required for the bundled sample or standard `yfinance` mode.
7. Click **Deploy**.

## First deployment test

After the build completes:

1. Confirm the app opens without an import error.
2. Keep **Bundled GOOG sample** selected.
3. Verify the data overview and price trend render.
4. Confirm the predicted closing price appears.
5. Confirm the model-comparison table and both evaluation plots render.
6. Download the forecast CSV.
7. Upload a CSV containing `Date` and `Close` to test upload mode.
8. Test optional live mode. A live-data failure should show a controlled message rather than crash the app.

## Build troubleshooting

### TensorFlow installation is slow

TensorFlow is the largest dependency. A first build may take several minutes. Do not interrupt the deployment while dependencies are being installed.

### App cannot find the model or sample

Confirm that the complete project folder was pushed and that these paths exist on GitHub:

```text
02-google-stock-price-prediction/models/google_stock_rnn_model.keras
02-google-stock-price-prediction/data/sample_google_stock_data.csv
```

### Live data fails

The app is intentionally not dependent on live data. Return to **Bundled GOOG sample** or upload a CSV. Live access can fail because of network restrictions, upstream changes, or rate limits.

### Dependency conflict

Confirm Streamlit is reading:

```text
02-google-stock-price-prediction/app/requirements.txt
```

Do not add another dependency file beside the app entry point.

### Python version mismatch

Delete and redeploy the app using Python 3.12 if it was originally created with another Python version.

## Add the live link to GitHub

After deployment, replace this README placeholder:

```text
ADD-LIVE-STREAMLIT-URL-HERE
```

Also add the link under the GitHub repository **About** section.

Recommended repository topics:

```text
simple-rnn
recurrent-neural-network
time-series-forecasting
stock-price-prediction
financial-data
keras
tensorflow
streamlit
machine-learning
data-science
```

## Suggested public link placement

Use the final Streamlit URL in:

- the project README;
- the repository About section;
- a LinkedIn project post;
- your resume project section; and
- your portfolio website.

## Security and data guidance

- Do not place brokerage credentials, API keys, personal portfolio data, or employer data in the repository.
- Do not commit `.streamlit/secrets.toml`.
- Keep large or licensed data outside GitHub.
- Treat optional `yfinance` data as research/educational input and review the applicable data-provider terms.

## Financial disclaimer

The deployed application is for educational and portfolio demonstration purposes only. It is not financial advice and must not be used as the sole basis for an investment decision.
