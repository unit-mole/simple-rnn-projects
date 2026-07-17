import numpy as np
import pandas as pd

from src.feature_engineering import build_return_forecast_frame
from src.sequence_generation import create_return_sequences, chronological_split_and_scale


def test_sequence_shape_and_chronological_split():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = 100 * np.exp(np.linspace(0, 0.20, len(dates)))
    frame = build_return_forecast_frame(pd.DataFrame({"Date": dates, "Close": prices}))
    sequences = create_return_sequences(frame, window_size=10)
    split = chronological_split_and_scale(sequences)

    assert sequences.X.shape[1:] == (10, 1)
    assert len(split.X_train) > len(split.X_validation) > 0
    assert len(split.X_test) > 0
    assert split.target_dates_test[0] > sequences.target_dates[split.split_indices["train_end"] - 1]
    assert abs(float(split.X_train.mean())) < 1e-6
