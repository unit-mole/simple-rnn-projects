# IMDb Data Guidance

## Project data source

The project uses the TensorFlow/Keras IMDb movie-review dataset. The standard
dataset contains positive and negative review labels and is loaded
programmatically during retraining.

The reviewed source run contains:

- 2,000 real IMDb reviews in the training pool;
- 1,600 reviews in the model-fitting partition;
- 400 reviews in the validation partition;
- 600 untouched real IMDb test reviews; and
- 600 synthetic reviews retained only as historical development context.

The complete decoded IMDb review text is **not committed to GitHub**. The
repository includes a small hand-written sample and a saved model for immediate
inference.

## Included sample

```text
data/sample_reviews.csv
```

| Column | Purpose |
|---|---|
| `review` | Hand-written movie-review text |
| `illustrative_tone` | Human-readable tone for demonstration only |

The application is a binary positive/negative classifier. Rows marked `mixed`
are intentional stress tests, not a third model class.

## Supported upload columns

The application searches for the first matching text column from:

```text
review
text
review_text
comment
content
```

A label column is optional. Supported binary labels include `0`/`1` and
`negative`/`positive`.

## Deployment limits

- Maximum batch size: 1,000 reviews
- Maximum uploaded CSV size: 5 MB
- Maximum review length: 50,000 characters

## Retraining data

`train_model.py` downloads the Keras IMDb dataset when it is not already
cached. Keep the downloaded dataset, complete decoded reviews, and temporary
training exports outside GitHub.

## Repository safety

Do not commit:

- complete raw or decoded IMDb datasets;
- private, employer, or customer feedback;
- licensed text corpora without redistribution permission;
- API keys or Streamlit secrets;
- virtual environments or local cache directories.

Saved model artifacts, aggregate metrics, de-identified predictions, and
portfolio visualizations under `models/` and `outputs/` are intentionally
included.
