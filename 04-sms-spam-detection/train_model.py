#!/usr/bin/env python
"""Retrain the SMS Simple RNN and refresh core evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from src.config import DATA_SOURCE_URL,MAX_SEQUENCE_LENGTH,MODEL_DIR,OUTPUT_DIR
from src.data_preprocessing import clean_sms_frame,load_tab_separated_sms
from src.model_evaluation import classification_report_frame,classify_probabilities
from src.model_training import split_sms_dataset,train_sms_models

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="",help="Optional local CSV/TSV path.")
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--batch-size",type=int,default=64)
    return parser.parse_args()

def load_frame(path):
    if not path:
        return load_tab_separated_sms(DATA_SOURCE_URL)
    input_path=Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() in {".tsv",".txt"}:
        return load_tab_separated_sms(input_path)
    return clean_sms_frame(pd.read_csv(input_path))

def main():
    args=parse_args()
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    frame,quality_report=load_frame(args.data)
    result=train_sms_models(frame,epochs=args.epochs,batch_size=args.batch_size)
    train,validation,test=split_sms_dataset(frame)
    y_test=test["label"].to_numpy(dtype=int)
    predictions=classify_probabilities(result.test_probabilities,result.threshold)

    result.model.save(MODEL_DIR/"sms_spam_simple_rnn_model.keras")
    result.tokenizer.save(
        MODEL_DIR/"tokenizer.json",
        max_sequence_length=MAX_SEQUENCE_LENGTH,padding="post",truncation="post",
    )
    metadata={
        "project_name":"SMS Spam Detection using Simple RNN",
        "modeling_rows":len(frame),"train_rows":result.train_rows,
        "validation_rows":result.validation_rows,"test_rows":result.test_rows,
        "decision_threshold":result.threshold,"class_weights":result.class_weights,
        "test_metrics":result.test_metrics,"quality_report":quality_report,
    }
    (MODEL_DIR/"model_metadata.json").write_text(
        json.dumps(metadata,indent=2),encoding="utf-8"
    )
    (OUTPUT_DIR/"model_metrics.json").write_text(
        json.dumps({"model":"Simple RNN",**result.test_metrics},indent=2),
        encoding="utf-8",
    )
    result.history.to_csv(OUTPUT_DIR/"training_history.csv",index=False)
    result.threshold_table.to_csv(OUTPUT_DIR/"threshold_analysis.csv",index=False)
    pd.DataFrame({
        "message_id":[f"test_{i+1:05d}" for i in range(len(test))],
        "true_label":y_test,"spam_probability":result.test_probabilities,
        "predicted_label":predictions,
    }).to_csv(OUTPUT_DIR/"test_predictions.csv",index=False)
    classification_report_frame(y_test,predictions).to_csv(
        OUTPUT_DIR/"classification_report.csv",index=False
    )
    pd.DataFrame([
        {"model":"Majority Class",**result.majority_metrics},
        {"model":"Multinomial Naive Bayes",**result.naive_bayes_metrics},
        {"model":"TF-IDF + Logistic Regression",**result.logistic_metrics},
        {"model":"Simple RNN",**result.test_metrics},
    ]).to_csv(OUTPUT_DIR/"model_comparison.csv",index=False)
    print(json.dumps(metadata,indent=2))
    print("Training complete. Review and regenerate portfolio charts before publishing.")

if __name__=="__main__":
    main()
