import pandas as pd
import pytest

from src.data_preprocessing import clean_review_frame, normalize_binary_label


def test_clean_review_frame_removes_blanks_and_duplicates():
    frame = pd.DataFrame(
        {
            "text": ["Great movie", "Great movie", "", None, "Bad movie"],
            "sentiment": ["positive", "positive", "negative", "negative", "negative"],
        }
    )
    cleaned, report = clean_review_frame(frame)
    assert len(cleaned) == 2
    assert report["duplicate_reviews_found"] == 1
    assert report["missing_or_blank_text_rows_removed"] == 2
    assert cleaned["label"].tolist() == [1, 0]


def test_invalid_label_is_rejected():
    with pytest.raises(ValueError):
        normalize_binary_label("neutral")
