"""Validate repository artifacts without loading TensorFlow."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parent
REQUIRED_FILES=[
    "README.md","README_HOSTING.md","PROJECT_REVIEW.md",
    "app/streamlit_app.py","app/requirements.txt","data/sample_sms_messages.csv",
    "models/sms_spam_simple_rnn_model.keras","models/tokenizer.json",
    "models/model_metadata.json","models/MODEL_CARD.md",
    "outputs/model_metrics.json","outputs/model_comparison.csv",
    "outputs/test_predictions.csv","outputs/error_analysis.csv",
    "outputs/confusion_matrix.png","outputs/roc_curve.png",
    "outputs/precision_recall_curve.png","outputs/training_accuracy.png",
    "outputs/training_loss.png",
]
METRIC_KEYS=["accuracy","precision","recall","f1","roc_auc","pr_auc"]

def validate_project(root:Path=PROJECT_ROOT):
    errors=[]
    for relative in REQUIRED_FILES:
        path=root/relative
        if not path.exists():
            errors.append(f"Missing required file: {relative}")
        elif path.is_file() and path.stat().st_size==0:
            errors.append(f"Empty required file: {relative}")
    metadata_path=root/"models"/"model_metadata.json"
    metrics_path=root/"outputs"/"model_metrics.json"
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        total=int(metadata["train_rows"])+int(metadata["validation_rows"])+int(metadata["test_rows"])
        if total!=int(metadata["modeling_rows"]):
            errors.append("Split rows do not equal modeling rows.")
        if int(metadata.get("train_test_overlap",-1))!=0:
            errors.append("Training/test overlap is not zero.")
    if metadata_path.exists() and metrics_path.exists():
        metadata=json.loads(metadata_path.read_text())
        metrics=json.loads(metrics_path.read_text())
        for key in METRIC_KEYS:
            if abs(float(metadata["test_metrics"].get(key,-1))-float(metrics.get(key,-2)))>1e-10:
                errors.append(f"Metric mismatch for {key}.")
    comparison_path=root/"outputs"/"model_comparison.csv"
    if comparison_path.exists():
        expected={
            "Majority Class","Multinomial Naive Bayes",
            "TF-IDF + Logistic Regression","Simple RNN",
        }
        if set(pd.read_csv(comparison_path)["model"])!=expected:
            errors.append("Unexpected model comparison rows.")
    return errors

def main():
    errors=validate_project()
    if errors:
        print("Project validation failed:")
        for error in errors: print(f"- {error}")
        return 1
    print("Project validation passed.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
