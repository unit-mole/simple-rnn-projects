"""Data loading, schema normalization, and quality checks for stock data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATE_ALIASES = ("Date", "Datetime", "Timestamp", "date", "datetime", "timestamp")
TARGET_PREFERENCE = ("Close", "Adj Close", "Adj_Close", "Adjusted Close")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance-style MultiIndex columns without losing useful names."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        first_level = [str(col[0]) for col in out.columns]
        if len(set(first_level)) == len(first_level):
            out.columns = first_level
        else:
            out.columns = ["_".join(str(x) for x in col if str(x)) for col in out.columns]
    return out


def load_stock_csv(path_or_buffer) -> pd.DataFrame:
    """Load a CSV path, file-like object, or Streamlit uploaded file."""
    return pd.read_csv(path_or_buffer)


def choose_target_column(columns: Iterable[str], requested: str | None = None) -> str:
    """Choose a supported price target, preferring Close for project consistency."""
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    if requested:
        key = requested.strip().lower()
        if key in normalized:
            return normalized[key]
        raise ValueError(f"Requested target column '{requested}' was not found.")

    for candidate in TARGET_PREFERENCE:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    close_like = [str(col) for col in columns if "close" in str(col).lower()]
    if close_like:
        return close_like[0]
    raise ValueError("No Close or adjusted-close column was found in the dataset.")


def standardize_stock_data(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Standardize stock data while preserving available numeric fields.

    Returns
    -------
    clean_df, selected_target, quality_report
    """
    if df is None or df.empty:
        raise ValueError("The supplied stock dataset is empty.")

    out = _flatten_columns(df)
    if not any(str(c) in DATE_ALIASES for c in out.columns):
        if isinstance(out.index, pd.DatetimeIndex) or out.index.name:
            out = out.reset_index()

    date_column = next((c for c in out.columns if str(c) in DATE_ALIASES), None)
    if date_column is None:
        date_column = next((c for c in out.columns if "date" in str(c).lower()), None)
    if date_column is None:
        raise ValueError("A Date/Datetime column is required.")

    selected_target = choose_target_column(out.columns, target_column)
    original_rows = len(out)
    original_duplicate_dates = int(out[date_column].duplicated().sum())

    out = out.rename(columns={date_column: "Date"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    for column in out.columns:
        if column != "Date":
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["Date", selected_target])
    out = out[out[selected_target] > 0]
    out = out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    if len(out) < 40:
        raise ValueError("At least 40 valid chronological price rows are required.")

    quality_report = {
        "rows_received": int(original_rows),
        "rows_retained": int(len(out)),
        "rows_removed": int(original_rows - len(out)),
        "duplicate_dates_removed": original_duplicate_dates,
        "missing_target_after_cleaning": int(out[selected_target].isna().sum()),
        "date_min": out["Date"].min().isoformat(),
        "date_max": out["Date"].max().isoformat(),
        "target_column": selected_target,
    }
    return out, selected_target, quality_report


def load_and_standardize(path: str | Path, target_column: str | None = None):
    """Convenience wrapper for local training scripts."""
    return standardize_stock_data(load_stock_csv(path), target_column=target_column)
