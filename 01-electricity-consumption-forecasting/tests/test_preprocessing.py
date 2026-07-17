import pandas as pd

from src.data_preprocessing import prepare_time_series


def test_preprocessing_sorts_deduplicates_and_fills_gaps():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 00:00"],
            "load": [30.0, 10.0, 14.0],
        }
    )
    cleaned, report = prepare_time_series(frame, "time", "load", frequency="h")
    assert cleaned["timestamp"].is_monotonic_increasing
    assert len(cleaned) == 3
    assert cleaned.loc[0, "load"] == 12.0
    assert cleaned["load"].isna().sum() == 0
    assert report.duplicate_timestamps_aggregated == 1
