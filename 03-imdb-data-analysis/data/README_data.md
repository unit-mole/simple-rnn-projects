# IMDb Data Guidance

## Project data source

The supplied project uses the TensorFlow/Keras IMDb movie-review dataset. The standard
dataset contains labeled positive and negative reviews and is loaded programmatically
through Keras during retraining.

The reviewed source run exported:

- 2,000 real IMDb training reviews;
- 600 untouched real IMDb test reviews; and
- 600 synthetic reviews used for pipeline validation.

The complete exported review text is **not included in this GitHub package**. The project
contains a small hand-written sample file for application testing and a saved model for
immediate inference.

## Included sample

```text
data/sample_reviews.csv
```

Columns:

| Column | Purpose |
|---|---|
| `review` | Hand-written movie-review text |
| `expected_sentiment` | Human-readable demonstration label |

The expected label is included only for sample interpretation. Uploaded files need only a
text column.

## Supported upload columns

The Streamlit application searches for the first matching column from:

```text
review
text
review_text
comment
content
```

## Retraining data

`train_model.py` downloads the Keras IMDb dataset when it is not already cached. Keep the
downloaded dataset, full decoded reviews, and temporary training files outside GitHub.

## Repository safety

Do not commit:

- the complete raw IMDb dataset;
- private or licensed review corpora;
- employer or customer feedback;
- API keys or Streamlit secrets;
- virtual environments or local cache directories.

The saved model, tokenizer, metadata, aggregate metrics, and de-identified prediction
artifacts under `models/` and `outputs/` are intentionally included.
