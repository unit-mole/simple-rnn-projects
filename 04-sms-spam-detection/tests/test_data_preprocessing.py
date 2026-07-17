import pandas as pd
import pytest
from src.data_preprocessing import clean_sms_frame,normalize_binary_label

def test_v1_v2_schema_and_duplicate_removal():
    frame=pd.DataFrame({
        "v1":["ham","ham","spam","spam"],
        "v2":["See you soon","See   you soon","WIN £500 now",""],
    })
    cleaned,report=clean_sms_frame(frame)
    assert len(cleaned)==2
    assert report["duplicate_messages_found"]==1
    assert report["blank_messages_removed"]==1
    assert cleaned["label"].tolist()==[0,1]

def test_invalid_label_fails():
    with pytest.raises(ValueError):
        normalize_binary_label("maybe")
