import pandas as pd

from src.data_preprocessing import standardize_stock_data


def test_standardize_sorts_deduplicates_and_drops_invalid_rows():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02", "bad-date"] * 12,
            "Close": [103.0, 101.0, 102.0, 99.0] * 12,
        }
    )
    # Add enough unique valid dates for the minimum-row guard.
    extra = pd.DataFrame(
        {
            "Date": pd.date_range("2024-02-01", periods=45, freq="B"),
            "Close": range(110, 155),
        }
    )
    clean, target, report = standardize_stock_data(pd.concat([raw, extra], ignore_index=True))
    assert target == "Close"
    assert clean["Date"].is_monotonic_increasing
    assert not clean["Date"].duplicated().any()
    assert clean["Close"].notna().all()
    assert report["rows_retained"] == len(clean)
