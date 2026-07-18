# Data Documentation

## Included corpus

`sample_text.csv` contains **1,020 privacy-safe synthetic sentences**
created specifically for this educational portfolio project. The sentences cover five
domains that were already present in the supplied notebook:

- artificial intelligence,
- finance,
- healthcare,
- energy, and
- quality/manufacturing.

The file has four columns:

| Column | Description |
|---|---|
| `sentence_id` | Stable sentence identifier |
| `topic` | Curated domain label used only for analysis |
| `text` | Synthetic training sentence |
| `source_type` | Data provenance marker |

`sample_sentences.txt` contains a small human-readable subset for demonstrations, and
`topic_lexicon.json` contains the curated vocabulary used for domain-purity evaluation.

## Why synthetic data is used

The included corpus is reproducible, contains no personal or employer information, and
keeps the deployed application independent of network downloads. It is intentionally
small and educational; it does not represent broad real-world language.

## Optional public corpus

The supplied notebook attempted to load the scikit-learn 20 Newsgroups dataset. The
cleaned project preserves this idea through `load_optional_20newsgroups()` in
`src/data_preprocessing.py`. It is an optional local retraining source and is **not**
downloaded by the Streamlit app or committed to GitHub.

Official reference:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html

## Safety rules

- Do not add private messages, employer documents, customer records, health records, or
  other sensitive text.
- Do not commit downloaded raw corpora unless redistribution is clearly permitted.
- Review external datasets for licensing, privacy, bias, and harmful-language concerns.
- Keep local downloaded datasets inside ignored folders such as `data/raw/`.
