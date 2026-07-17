from pathlib import Path
import json
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]

def test_model_metadata_split_contract():
    metadata=json.loads((PROJECT_ROOT/"models"/"model_metadata.json").read_text())
    assert metadata["train_rows"]+metadata["validation_rows"]+metadata["test_rows"]==metadata["modeling_rows"]
    assert metadata["train_test_overlap"]==0

def test_model_comparison_contract():
    comparison=pd.read_csv(PROJECT_ROOT/"outputs"/"model_comparison.csv")
    assert set(comparison["model"])=={
        "Majority Class","Multinomial Naive Bayes",
        "TF-IDF + Logistic Regression","Simple RNN",
    }

def test_privacy_safe_sample_schema():
    sample=pd.read_csv(PROJECT_ROOT/"data"/"sample_sms_messages.csv")
    assert {"message","illustrative_category"}.issubset(sample.columns)
    assert sample["message"].str.strip().ne("").all()
