"""Calendar feature engineering for sequential electricity forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CALENDAR_FEATURE_COLUMNS


def add_calendar_features(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Add bounded cyclical calendar features known at forecast time."""
    frame = df.copy()
    ts = pd.to_datetime(frame[timestamp_column], errors="raise")
    hour = ts.dt.hour
    day_of_week = ts.dt.dayofweek
    month = ts.dt.month - 1

    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["weekend_flag"] = (day_of_week >= 5).astype(float)
    frame["peak_flag"] = hour.between(17, 21).astype(float)
    return frame


def calendar_vector(timestamp: pd.Timestamp) -> np.ndarray:
    """Return calendar features for one future timestamp."""
    row = add_calendar_features(pd.DataFrame({"timestamp": [pd.Timestamp(timestamp)]}))
    return row[CALENDAR_FEATURE_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
