# Hosting Guide — Electricity Consumption Forecasting

## Recommended Platform

Use **Streamlit Community Cloud** for the first public deployment.

The application is already built with Streamlit, the trained model is committed,
and the monorepo includes a dependency file beside the application entrypoint.
No Docker deployment is required for the first version.

## Deployment Files

```text
simple-rnn-projects/
└── 01-electricity-consumption-forecasting/
    ├── app/
    │   ├── requirements.txt
    │   └── streamlit_app.py
    ├── data/sample_input.csv
    ├── models/
    │   ├── electricity_rnn_model.keras
    │   ├── model_metadata.json
    │   └── scaler.pkl
    ├── outputs/
    │   ├── baseline_comparison.csv
    │   └── model_metrics.json
    ├── src/
    └── requirements.txt
```

## 1. Test Locally

From the project folder:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

Test:

- Demo sample loading
- CSV upload
- Timestamp and target selection
- Forecast-horizon control
- Forecast chart and business interpretation
- Performance tab
- Forecast CSV download

## 2. GitHub Repository Settings

Use:

```text
Repository: simple-rnn-projects
Branch: main
Project folder: 01-electricity-consumption-forecasting
```

The repository should be public for Streamlit Community Cloud unless your
Streamlit plan supports the selected private-repository workflow.

## 3. Streamlit Community Cloud Settings

Create a new app using:

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
Main file path:
01-electricity-consumption-forecasting/app/streamlit_app.py
Python version: 3.12
```

No secrets are required for the included demonstration.

The app-level dependency file is:

```text
01-electricity-consumption-forecasting/app/requirements.txt
```

## 4. Expected Public Link

After deployment, Streamlit provides a link similar to:

```text
https://electricity-rnn-forecasting.streamlit.app
```

The exact subdomain depends on availability.

## 5. Update the Portfolio After Deployment

Replace the deployment-pending text in:

```text
simple-rnn-projects/README.md
01-electricity-consumption-forecasting/README.md
```

Add the final URL to:

- The project badge
- The root project-roadmap table
- The GitHub repository About section
- Your resume project entry
- LinkedIn Featured
- Your portfolio website

## Troubleshooting

### Model file is missing

Run:

```bat
python train_model.py
```

Then verify:

```text
models/electricity_rnn_model.keras
models/scaler.pkl
models/model_metadata.json
```

### The app cannot import `src`

Keep the supplied folder structure unchanged and confirm that the selected
entrypoint is inside `01-electricity-consumption-forecasting/app/`.

### Dependency installation fails

Use Python 3.12 and confirm that `app/requirements.txt` is committed. Review the
Streamlit build log before changing versions.

### Uploaded data is rejected

The file needs:

- A parseable timestamp column
- A numeric consumption column
- At least 24 chronological observations
- A frequency compatible with the model or a retrained artifact

### Uploaded values differ greatly from training data

Retrain the model on that dataset. The demo app intentionally warns when the
input range differs materially from the training range.

### Hosted memory is limited

Keep training outside the hosted app. Commit the trained inference artifacts
and use the hosted app only for prediction.
