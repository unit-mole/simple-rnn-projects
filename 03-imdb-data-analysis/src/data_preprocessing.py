"""Data loading and validation for IMDb sentiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    LABEL_COLUMN_CANDIDATES,
    MAX_REVIEW_CHARACTERS,
    TEXT_COLUMN_CANDIDATES,
)


def identify_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    """Return the first candidate matching a column name, case-insensitively."""
    lookup = {
        str(column).strip().lower(): str(column)
        for column in columns
    }
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def identify_text_column(frame: pd.DataFrame) -> str:
    """Identify a supported text column or raise a helpful error."""
    column = identify_column(frame.columns, TEXT_COLUMN_CANDIDATES)
    if column is None:
        raise ValueError(
            "No review-text column was found. Use one of: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
        )
    return column


def identify_label_column(frame: pd.DataFrame) -> str | None:
    """Identify an optional binary sentiment-label column."""
    return identify_column(frame.columns, LABEL_COLUMN_CANDIDATES)


def normalize_binary_label(value: object) -> int:
    """Map common binary sentiment labels to 0/1."""
    if pd.isna(value):
        raise ValueError("Sentiment label cannot be missing.")

    if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
        return int(value)

    text = str(value).strip().lower()
    if text in {"1", "positive", "pos", "good"}:
        return 1
    if text in {"0", "negative", "neg", "bad"}:
        return 0

    raise ValueError(
        f"Unsupported binary label: {value!r}. "
        "Use 0/1 or negative/positive."
    )


def clean_review_frame(
    frame: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
    drop_duplicates: bool = True,
    max_review_characters: int | None = MAX_REVIEW_CHARACTERS,
) -> tuple[pd.DataFrame, dict]:
    """Validate, clean, and standardize a review dataframe."""
    if frame is None or frame.empty:
        raise ValueError("The review dataset is empty.")

    text_column = text_column or identify_text_column(frame)
    label_column = label_column or identify_label_column(frame)

    selected = [text_column]
    if label_column:
        selected.append(label_column)

    cleaned = frame[selected].copy()
    rename_map = {text_column: "review"}
    if label_column:
        rename_map[label_column] = "label"
    cleaned = cleaned.rename(columns=rename_map)

    input_rows = len(cleaned)
    cleaned["review"] = (
        cleaned["review"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    missing_text_rows = int((cleaned["review"] == "").sum())
    cleaned = cleaned[cleaned["review"] != ""].copy()

    over_limit_rows = 0
    if max_review_characters is not None:
        over_limit_mask = (
            cleaned["review"].str.len() > int(max_review_characters)
        )
        over_limit_rows = int(over_limit_mask.sum())
        if over_limit_rows:
            raise ValueError(
                f"{over_limit_rows} review(s) exceed the "
                f"{int(max_review_characters):,}-character safety limit."
            )

    duplicate_rows = int(
        cleaned.duplicated(subset=["review"]).sum()
    )
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates(
            subset=["review"],
            keep="first",
        )

    if "label" in cleaned.columns:
        cleaned["label"] = cleaned["label"].map(
            normalize_binary_label
        )
        cleaned["sentiment"] = cleaned["label"].map(
            {0: "negative", 1: "positive"}
        )

    cleaned = cleaned.reset_index(drop=True)

    report = {
        "input_rows": int(input_rows),
        "output_rows": int(len(cleaned)),
        "missing_or_blank_text_rows_removed": missing_text_rows,
        "duplicate_reviews_found": duplicate_rows,
        "duplicates_removed": bool(drop_duplicates),
        "reviews_over_character_limit": over_limit_rows,
        "maximum_review_characters": max_review_characters,
        "text_column": text_column,
        "label_column": label_column,
    }
    return cleaned, report


def load_review_csv(
    path_or_buffer: str | Path | object,
) -> tuple[pd.DataFrame, dict]:
    """Load a CSV and return standardized reviews plus a quality report."""
    frame = pd.read_csv(path_or_buffer)
    return clean_review_frame(frame)
