# Hosting Guide — IMDb Simple RNN Sentiment Application

## Deployment status

The application is deployed on Streamlit Community Cloud.

**Live application:**  
[Open the IMDb Movie Review Sentiment Analysis application](https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/)

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

Do not change the relative folder structure because the app resolves its model,
data, output, and source files from the project directory.

## Deployment limits

```text
Maximum CSV upload: 5 MB
Maximum batch size: 1,000 reviews
Maximum review length: 50,000 characters
```

## Post-deployment validation

Confirm that:

1. The saved Keras model and tokenizer load.
2. One positive and one negative manual review can be scored.
3. Sample-review predictions display correctly.
4. The CSV template downloads.
5. A compatible CSV can be uploaded.
6. Batch predictions and the scored CSV download work.
7. Metrics and all evaluation charts render.
8. Invalid files produce a controlled error.
9. The app is publicly accessible without sign-in.

## Repository validation

Before pushing a release, run:

```bat
python -m pytest -q
python validate_project.py
```

The validator checks required artifacts, screenshot paths, metadata consistency,
the model-comparison contract, the sample schema, and the deployed URL.

## Updating the deployed application

Commits to the relevant project files on `main` normally trigger a Streamlit
redeployment.

## Troubleshooting

### Model-loading error

Confirm:

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

### Invalid CSV

Use a text column named `review`, `text`, `review_text`, `comment`, or
`content`. Optional labels must be binary.

### Memory or timeout issue

Use a smaller CSV. The application does not retrain during deployment.

## Security

No secrets are required. Do not commit user uploads, private feedback,
credentials, API keys, or `.streamlit/secrets.toml`.
