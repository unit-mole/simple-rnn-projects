from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_DATA_PATH, TOPIC_LEXICON_PATH
from .text_preprocessing import tokenize


REQUIRED_COLUMNS = {"sentence_id", "topic", "text", "source_type"}


def load_included_corpus(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the privacy-safe corpus committed with the repository."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {data_path}")
    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["tokens"] = frame["text"].map(tokenize)
    frame = frame[frame["tokens"].map(len) >= 3].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No usable sentences remained after preprocessing.")
    return frame


def load_topic_lexicon(path: str | Path = TOPIC_LEXICON_PATH) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(topic): [str(word) for word in words] for topic, words in payload.items()}


def load_optional_20newsgroups(
    max_documents: int = 500,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Optionally fetch selected public 20 Newsgroups posts for local retraining.

    This function is never called by the deployed app. It requires internet access and
    keeps the downloaded corpus outside the Git repository in scikit-learn's cache.
    """
    from sklearn.datasets import fetch_20newsgroups

    selected_categories = list(categories) if categories else [
        "comp.graphics",
        "rec.autos",
        "rec.sport.baseball",
        "sci.med",
        "sci.space",
    ]
    dataset = fetch_20newsgroups(
        subset="train",
        categories=selected_categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    rows: list[dict[str, str]] = []
    for index, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        if not isinstance(text, str) or len(text.strip()) < 80:
            continue
        rows.append(
            {
                "sentence_id": f"20news_{index:05d}",
                "topic": dataset.target_names[int(target)],
                "text": text,
                "source_type": "20_newsgroups_optional",
            }
        )
        if len(rows) >= max_documents:
            break
    frame = pd.DataFrame(rows)
    frame["tokens"] = frame["text"].map(tokenize)
    return frame


def corpus_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.copy()
    summary["token_count"] = summary["tokens"].map(len)
    return (
        summary.groupby(["topic", "source_type"], as_index=False)
        .agg(sentences=("sentence_id", "count"), mean_tokens=("token_count", "mean"))
        .sort_values(["source_type", "topic"])
        .reset_index(drop=True)
    )
