from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_metadata_split_contract():
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "models"
            / "model_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        metadata["model_fit_reviews"]
        + metadata["validation_reviews"]
        == metadata["training_pool_reviews"]
    )
    assert metadata["post_selection_refit"] is False
    assert metadata["test_reviews"] == 600


def test_model_comparison_has_fair_scope():
    comparison = pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "model_comparison.csv"
    )
    assert set(comparison["model"]) == {
        "Majority Class",
        "Simple RNN",
        "TF-IDF + Logistic Regression",
    }
    baseline = comparison.loc[
        comparison["model"]
        == "TF-IDF + Logistic Regression"
    ].iloc[0]
    assert 0.0 <= baseline["accuracy"] <= 1.0
    assert 0.0 <= baseline["roc_auc"] <= 1.0


def test_sample_schema_is_binary_model_safe():
    sample = pd.read_csv(
        PROJECT_ROOT / "data" / "sample_reviews.csv"
    )
    assert {"review", "illustrative_tone"}.issubset(
        sample.columns
    )
    assert sample["review"].str.strip().ne("").all()
    assert set(sample["illustrative_tone"]).issubset(
        {"positive", "negative", "mixed"}
    )
