"""Saved-artifact inference for manual and batch SMS scoring."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

from .config import (
    DECISION_THRESHOLD, MAX_SEQUENCE_LENGTH, METADATA_PATH, MODEL_PATH, TOKENIZER_PATH
)
from .sequence_generation import sequence_report, texts_to_padded_sequences
from .text_preprocessing import VocabularyTokenizer, clean_text

@dataclass
class SpamArtifacts:
    model: object
    tokenizer: VocabularyTokenizer
    tokenizer_metadata: dict
    model_metadata: dict

def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> SpamArtifacts:
    import keras
    paths=[Path(model_path),Path(tokenizer_path),Path(metadata_path)]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact was not found: {path}")
    model=keras.models.load_model(paths[0],compile=False)
    tokenizer,tokenizer_metadata=VocabularyTokenizer.load(paths[1])
    model_metadata=json.loads(paths[2].read_text(encoding="utf-8"))
    return SpamArtifacts(model,tokenizer,tokenizer_metadata,model_metadata)

def decision_confidence(spam_probability: float, threshold: float) -> float:
    if spam_probability >= threshold:
        normalized=(spam_probability-threshold)/max(1e-9,1-threshold)
    else:
        normalized=(threshold-spam_probability)/max(1e-9,threshold)
    return float(.5+.5*min(1.0,normalized))

def confidence_band(confidence: float) -> str:
    if confidence >= .85: return "High"
    if confidence >= .70: return "Moderate"
    return "Low"

def surface_cues(text: object) -> list[str]:
    cleaned=clean_text(text); cues=[]
    if "<url>" in cleaned: cues.append("contains a URL")
    if "<phone>" in cleaned: cues.append("contains a phone-number pattern")
    if "<currency>" in cleaned: cues.append("contains a currency symbol")
    if "!" in cleaned: cues.append("uses exclamation marks")
    groups={
        "promotional language":{"free","offer","deal","discount","voucher","bonus"},
        "prize or reward language":{"win","won","prize","reward","claim","cash"},
        "urgency language":{"urgent","now","today","immediately","expires"},
        "response instructions":{"reply","call","text","click","stop"},
    }
    tokens=set(re.findall(r"[a-z]+",cleaned))
    for description,words in groups.items():
        if tokens.intersection(words): cues.append(description)
    return cues or ["no predefined surface cue was detected"]

def interpretation_text(
    predicted_class: str,
    spam_probability: float,
    threshold: float,
    confidence: float,
) -> str:
    relation="above" if predicted_class=="spam" else "below"
    text=(
        f"The model score is {relation} the validation-selected spam threshold. "
        f"Spam probability is {spam_probability:.1%}, using a threshold of {threshold:.3f}."
    )
    if confidence < .70:
        text += " The score is close to the decision boundary, so manual review is appropriate."
    return text

def predict_messages(
    messages: list[object],
    artifacts: SpamArtifacts,
    threshold: float | None = None,
) -> tuple[pd.DataFrame,list[dict]]:
    if not messages:
        raise ValueError("At least one SMS message is required.")
    threshold=float(
        threshold if threshold is not None else
        artifacts.model_metadata.get("decision_threshold",DECISION_THRESHOLD)
    )
    max_length=int(
        artifacts.tokenizer_metadata.get("max_sequence_length",MAX_SEQUENCE_LENGTH)
    )
    sequences=texts_to_padded_sequences(
        artifacts.tokenizer,messages,max_length=max_length
    )
    probabilities=artifacts.model.predict(
        sequences,batch_size=256,verbose=0
    ).reshape(-1)
    labels=(probabilities>=threshold).astype(int)
    classes=np.where(labels==1,"spam","ham")
    confidences=[decision_confidence(float(p),threshold) for p in probabilities]
    result=pd.DataFrame({
        "message":[str(message) for message in messages],
        "clean_message":[clean_text(message) for message in messages],
        "spam_probability":probabilities,
        "predicted_label":labels,
        "predicted_class":classes,
        "decision_confidence":confidences,
        "confidence_band":[confidence_band(v) for v in confidences],
        "surface_cues":["; ".join(surface_cues(m)) for m in messages],
    })
    result["interpretation"]=[
        interpretation_text(str(c),float(p),threshold,float(conf))
        for c,p,conf in zip(classes,probabilities,confidences)
    ]
    reports=[
        sequence_report(artifacts.tokenizer,message,max_length)
        for message in messages
    ]
    return result,reports
