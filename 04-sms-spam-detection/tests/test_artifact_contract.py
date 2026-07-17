from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_APP_URL = "https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/"


def test_model_metadata_split_contract():
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "models"
            / "model_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        metadata["train_rows"]
        + metadata["validation_rows"]
        + metadata["test_rows"]
        == metadata["modeling_rows"]
    )
    assert metadata["train_validation_overlap"] == 0
    assert metadata["train_test_overlap"] == 0
    assert metadata["validation_test_overlap"] == 0
    assert 0.0 < metadata["decision_threshold"] < 1.0


def test_model_comparison_contract():
    comparison = pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "model_comparison.csv"
    )
    assert set(comparison["model"]) == {
        "Majority Class",
        "Multinomial Naive Bayes",
        "TF-IDF + Logistic Regression",
        "Simple RNN",
    }
    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        assert comparison[column].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_privacy_safe_prediction_contract():
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "models"
            / "model_metadata.json"
        ).read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "test_predictions.csv"
    )
    assert len(predictions) == metadata["test_rows"]
    assert predictions["message_id"].is_unique
    assert "message" not in predictions.columns
    assert predictions[
        "spam_probability"
    ].between(0.0, 1.0).all()


def test_privacy_safe_sample_schema():
    sample = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "sample_sms_messages.csv"
    )
    assert {
        "message",
        "illustrative_category",
    }.issubset(sample.columns)
    assert sample["message"].str.strip().ne("").all()


def test_documentation_contains_deployed_url():
    for relative in [
        "README.md",
        "README_HOSTING.md",
    ]:
        content = (
            PROJECT_ROOT / relative
        ).read_text(encoding="utf-8")
        assert LIVE_APP_URL in content
        assert "ADD-LIVE-STREAMLIT-URL-HERE" not in content
