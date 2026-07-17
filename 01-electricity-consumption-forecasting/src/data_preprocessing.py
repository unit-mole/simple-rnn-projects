"""Time-series loading, cleaning, validation, and regularization utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    input_rows: int
    output_rows: int
    invalid_timestamps_removed: int
    invalid_target_values_detected: int
    duplicate_timestamps_aggregated: int
    missing_target_values_filled: int
    inferred_frequency: str
    outlier_rows_flagged: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def load_csv(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load a CSV path or uploaded file-like object."""
    return pd.read_csv(source)


def _infer_frequency(timestamps: pd.Series, fallback: str = "h") -> str:
    clean = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values()
    if len(clean) < 3:
        return fallback
    inferred = pd.infer_freq(clean.head(min(len(clean), 500)))
    if inferred:
        return inferred
    median_delta = clean.diff().dropna().median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return fallback
    return pd.tseries.frequencies.to_offset(median_delta).freqstr


def prepare_time_series(
    raw_df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    frequency: str | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean a time series without introducing future information.

    Steps:
    1. Parse timestamps and numeric target values.
    2. Sort chronologically.
    3. Aggregate duplicate timestamps by mean.
    4. Reindex to a regular frequency.
    5. Fill target gaps by time interpolation, then forward/backward fill.
    6. Flag IQR outliers for audit; values are not silently deleted or clipped.
    """
    if timestamp_column not in raw_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' was not found.")
    if target_column not in raw_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    input_rows = len(raw_df)
    work = raw_df.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    invalid_timestamps = int(work[timestamp_column].isna().sum())
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    invalid_targets = int(work[target_column].isna().sum())
    work = work.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    duplicate_count = int(work.duplicated(timestamp_column).sum())
    numeric_columns = work.select_dtypes(include=[np.number]).columns.tolist()
    aggregation = {column: "mean" for column in numeric_columns}
    for column in work.columns:
        if column not in aggregation and column != timestamp_column:
            aggregation[column] = "first"
    work = work.groupby(timestamp_column, as_index=False).agg(aggregation)

    selected_frequency = frequency or _infer_frequency(work[timestamp_column])
    work = work.set_index(timestamp_column).sort_index()
    full_index = pd.date_range(work.index.min(), work.index.max(), freq=selected_frequency)
    work = work.reindex(full_index)
    work.index.name = "timestamp"

    missing_before = int(work[target_column].isna().sum())
    work[target_column] = work[target_column].interpolate(method="time")
    work[target_column] = work[target_column].ffill().bfill()
    if work[target_column].isna().any():
        raise ValueError("The target column still contains missing values after cleaning.")

    q1 = float(work[target_column].quantile(0.25))
    q3 = float(work[target_column].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    work["outlier_flag"] = ~work[target_column].between(lower, upper)

    cleaned = work.reset_index()
    report = DataQualityReport(
        input_rows=input_rows,
        output_rows=len(cleaned),
        invalid_timestamps_removed=invalid_timestamps,
        invalid_target_values_detected=invalid_targets,
        duplicate_timestamps_aggregated=duplicate_count,
        missing_target_values_filled=missing_before,
        inferred_frequency=selected_frequency,
        outlier_rows_flagged=int(cleaned["outlier_flag"].sum()),
    )
    return cleaned, report
