# Hosting Guide — SMS Spam Simple RNN Application

## Recommended host

Use **Streamlit Community Cloud**. The application loads saved model artifacts,
requires no database or paid API, and can deploy directly from the existing
GitHub monorepo.

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

Confirm model loading, positive and negative examples, template download,
batch scoring, downloadable output, metrics, charts, controlled invalid-file
errors, and public access without sign-in.

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

## Privacy

Do not upload private messages to the public demo. Do not commit user uploads,
credentials, secrets, personal identifiers, or `.streamlit/secrets.toml`.
