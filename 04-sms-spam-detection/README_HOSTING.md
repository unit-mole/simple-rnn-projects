# Hosting Guide — SMS Spam Simple RNN Application

## Deployment status

The application is deployed on Streamlit Community Cloud.

**Live application:**  
[Open the SMS Spam Detection application](https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/)

## Deployment configuration

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
Main file path: 04-sms-spam-detection/app/streamlit_app.py
Python: 3.12
```

No secrets are required.

## Required files

```text
04-sms-spam-detection/app/streamlit_app.py
04-sms-spam-detection/app/requirements.txt
04-sms-spam-detection/data/sample_sms_messages.csv
04-sms-spam-detection/models/sms_spam_simple_rnn_model.keras
04-sms-spam-detection/models/tokenizer.json
04-sms-spam-detection/models/model_metadata.json
04-sms-spam-detection/outputs/model_metrics.json
04-sms-spam-detection/outputs/model_comparison.csv
04-sms-spam-detection/outputs/error_analysis.csv
04-sms-spam-detection/outputs/*.png
04-sms-spam-detection/src/*.py
```

## Pre-deployment validation

Run:

```bat
python -m pytest -q
python validate_project.py
python runtime_smoke_test.py
```

The lightweight validator checks documentation, screenshots, model and tokenizer
presence, split integrity, metric consistency, output schemas, and privacy-safe
prediction artifacts. The runtime smoke test loads the saved Keras model and
tokenizer and performs real inference.

## Deployment steps

1. Push the project and workflow to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Choose **Create app**.
4. Select `unit-mole/simple-rnn-projects`.
5. Select branch `main`.
6. Enter `04-sms-spam-detection/app/streamlit_app.py`.
7. Select Python 3.12.
8. Leave secrets empty.
9. Deploy.

## Post-deployment tests

Test all four modes:

- Single Message
- Sample Messages
- CSV Upload
- Model Performance

Confirm model loading, obvious spam-like and legitimate examples, template
download, batch scoring, downloadable output, metrics, charts, controlled
invalid-file errors, and public access without sign-in.

## Screenshot plan

```text
01_streamlit_application_overview.png
02_single_message_prediction.png
03_batch_spam_detection_workflow.png
04_model_performance_and_error_analysis.png
```

## Limits

```text
Maximum upload: 5 MB
Maximum batch: 1,000 messages
Maximum message: 10,000 characters
```

## Updating the deployment

Changes pushed to relevant Project 04 files on the `main` branch normally trigger
a Streamlit redeployment. Confirm the application and the two GitHub Actions jobs
after each model or dependency update.

## Privacy

Do not upload private messages to the public demo. Do not commit user uploads,
credentials, secrets, personal identifiers, or `.streamlit/secrets.toml`.
