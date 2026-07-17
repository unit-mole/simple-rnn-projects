"""Feature engineering for a return-based next-day stock forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_diagnostic_features(df: pd.DataFrame, target_column: str = "Close") -> pd.DataFrame:
    """Add interpretable financial diagnostics without changing the model target."""
    out = df.copy()
    price = out[target_column].astype(float)
    out["Daily_Return"] = price.pct_change()
    out["Log_Return"] = np.log(price / price.shift(1))
    out["Moving_Average_5"] = price.rolling(5).mean()
    out["Moving_Average_20"] = price.rolling(20).mean()
    out["Rolling_Volatility_20"] = out["Log_Return"].rolling(20).std()
    out["Momentum_5"] = price / price.shift(5) - 1.0
    out["Momentum_20"] = price / price.shift(20) - 1.0
    return out


def build_return_forecast_frame(
    df: pd.DataFrame,
    target_column: str = "Close",
) -> pd.DataFrame:
    """
    Build the supervised frame used by the Simple RNN.

    The network learns from the previous log-return sequence instead of raw prices.
    Its output is the next-day log return, which is converted back into a closing
    price. This avoids the raw-price saturation seen in the original project.
    """
    out = df[["Date", target_column]].copy()
    out = out.rename(columns={target_column: "Current_Close"})
    out["Log_Return"] = np.log(out["Current_Close"] / out["Current_Close"].shift(1))
    out["Target_Return"] = np.log(
        out["Current_Close"].shift(-1) / out["Current_Close"]
    )
    out["Target_Close"] = out["Current_Close"].shift(-1)
    out["Target_Date"] = out["Date"].shift(-1)
    return out.dropna().reset_index(drop=True)
