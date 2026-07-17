# Hosting Guide — IMDb Simple RNN Sentiment Application

## Recommended host

Use **Streamlit Community Cloud**. The application is already written in Streamlit,
loads saved artifacts from the repository, and requires no database or paid API.

## Deployment configuration

```text
Repository: unit-mole/simple-rnn-projects
Branch: main
Main file path: 03-imdb-data-analysis/app/streamlit_app.py
Python: 3.12
```

## Required files

```text
03-imdb-data-analysis/app/streamlit_app.py
03-imdb-data-analysis/app/requirements.txt
03-imdb-data-analysis/data/sample_reviews.csv
03-imdb-data-analysis/models/imdb_simple_rnn_model.keras
03-imdb-data-analysis/models/tokenizer.json
03-imdb-data-analysis/models/model_metadata.json
03-imdb-data-analysis/outputs/model_metrics.json
03-imdb-data-analysis/outputs/model_comparison.csv
03-imdb-data-analysis/outputs/error_analysis.csv
03-imdb-data-analysis/outputs/*.png
03-imdb-data-analysis/src/*.py
```

Do not change the relative folder structure because the application resolves its data,
model, output, and source files from the project directory.

## Deployment steps

1. Push the project folder and CI workflow to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app**.
4. Choose `unit-mole/simple-rnn-projects`.
5. Choose branch `main`.
6. Enter:
   `03-imdb-data-analysis/app/streamlit_app.py`
7. Select Python 3.12.
8. Deploy.
9. Test all four app modes.

## Post-deployment validation

Confirm:

- the saved Keras model loads;
- one manual review can be scored;
- sample reviews generate predictions;
- a compatible CSV can be uploaded;
- scored CSV output downloads;
- metrics and comparison tables appear;
- confusion matrix, ROC, PR, and training charts render;
- the app displays a controlled error for an invalid CSV;
- no sign-in is required to view the public application.

## Screenshot plan

Save these files under `images/`:

```text
01_streamlit_application_overview.png
02_single_review_prediction.png
03_batch_sentiment_workflow.png
04_model_performance_and_error_analysis.png
```

## Updating the app

Commits to the project files on `main` normally trigger a Streamlit redeployment.

## Troubleshooting

### Model-loading error

Confirm the model exists:

```text
03-imdb-data-analysis/models/imdb_simple_rnn_model.keras
```

### Tokenizer error

Confirm:

```text
03-imdb-data-analysis/models/tokenizer.json
```

### Dependency error

Confirm Streamlit uses:

```text
03-imdb-data-analysis/app/requirements.txt
```

### Memory or timeout issue

The app does not retrain. Batch uploads are intentionally limited to 1,000 reviews.

### Invalid CSV

Use a text column named `review`, `text`, `review_text`, `comment`, or `content`.

## Security

No secrets are required. Do not add user review uploads, account credentials, API keys,
or `.streamlit/secrets.toml` to the repository.
