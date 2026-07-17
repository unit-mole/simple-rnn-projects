#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple RNN IMDB Sentiment App
Synthetic validation first, real IMDB data second, same preprocessing + Simple RNN pipeline.
Run:
    streamlit run simple_rnn_imdb_streamlit_FINAL_DOWNLOADABLE.py
"""

import os
import io
import json
import time
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


APP_TITLE = "Simple RNN IMDB Sentiment Classifier"
MAX_WORDS = 8000
MAX_LEN = 120
SEED = 42


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def build_synthetic_reviews(n: int = 200) -> pd.DataFrame:
    positive_templates = [
        "This movie was excellent with strong acting and a beautiful story",
        "I loved the film because it was emotional funny and memorable",
        "A wonderful experience with great direction and brilliant performances",
        "The plot was engaging and the characters were very likeable",
        "This is a fantastic movie that I would happily watch again",
    ]
    negative_templates = [
        "This movie was terrible with weak acting and a boring story",
        "I disliked the film because it was slow dull and forgettable",
        "A poor experience with bad direction and disappointing performances",
        "The plot was confusing and the characters were very annoying",
        "This is a frustrating movie that I would not watch again",
    ]
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"text": positive_templates[i % len(positive_templates)], "label": 1, "source": "synthetic"})
        else:
            rows.append({"text": negative_templates[i % len(negative_templates)], "label": 0, "source": "synthetic"})
    return pd.DataFrame(rows)


def load_real_imdb(max_train: int = 1200, max_test: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not TF_AVAILABLE:
        synth = build_synthetic_reviews(max_train + max_test)
        return synth.iloc[:max_train].copy(), synth.iloc[max_train:].copy(), "fallback_synthetic_tf_unavailable"

    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAX_WORDS)
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = {v + 3: k for k, v in word_index.items()}
        reverse_word_index[0] = "<PAD>"
        reverse_word_index[1] = "<START>"
        reverse_word_index[2] = "<UNK>"
        reverse_word_index[3] = "<UNUSED>"

        def decode_review(encoded):
            return " ".join(reverse_word_index.get(int(i), "<UNK>") for i in encoded)

        train_df = pd.DataFrame({
            "text": [decode_review(x) for x in x_train[:max_train]],
            "label": y_train[:max_train],
            "source": "real_imdb",
        })
        test_df = pd.DataFrame({
            "text": [decode_review(x) for x in x_test[:max_test]],
            "label": y_test[:max_test],
            "source": "real_imdb",
        })
        return train_df, test_df, "tensorflow_keras_imdb"
    except Exception as exc:
        synth = build_synthetic_reviews(max_train + max_test)
        train_df = synth.iloc[:max_train].copy()
        test_df = synth.iloc[max_train:].copy()
        train_df["source"] = "fallback_synthetic_after_imdb_error"
        test_df["source"] = "fallback_synthetic_after_imdb_error"
        return train_df, test_df, f"fallback_after_error: {exc}"


def train_tensorflow_rnn(train_df: pd.DataFrame, epochs: int = 2):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"].astype(str).tolist())
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_df["text"].astype(str)), maxlen=MAX_LEN)
    y_train = train_df["label"].astype(int).values

    model = models.Sequential([
        layers.Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        layers.SimpleRNN(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model, tokenizer, history


def train_fallback_classifier(train_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["text"].astype(str))
    y_train = train_df["label"].astype(int).values
    clf = LogisticRegression(max_iter=500)
    clf.fit(x_train, y_train)
    return clf, vectorizer


def predict_texts(texts, model_bundle):
    mode = model_bundle["mode"]
    if mode == "tensorflow_rnn":
        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]
        x = pad_sequences(tokenizer.texts_to_sequences([str(t) for t in texts]), maxlen=MAX_LEN)
        probs = model.predict(x, verbose=0).reshape(-1)
    else:
        vectorizer = model_bundle["vectorizer"]
        clf = model_bundle["model"]
        probs = clf.predict_proba(vectorizer.transform([str(t) for t in texts]))[:, 1]
    labels = (probs >= 0.5).astype(int)
    return pd.DataFrame({
        "text": texts,
        "positive_probability": probs,
        "predicted_label": labels,
        "predicted_sentiment": ["positive" if x == 1 else "negative" for x in labels],
    })


@st.cache_resource
def build_model(max_train: int, use_real: bool, epochs: int):
    set_seed()
    synthetic_df = build_synthetic_reviews(300)

    if use_real:
        real_train_df, real_test_df, source_name = load_real_imdb(max_train=max_train, max_test=300)
        train_df = pd.concat([synthetic_df, real_train_df], ignore_index=True)
    else:
        real_train_df, real_test_df, source_name = pd.DataFrame(), synthetic_df.copy(), "synthetic_only"
        train_df = synthetic_df.copy()

    if TF_AVAILABLE:
        try:
            model, tokenizer, history = train_tensorflow_rnn(train_df, epochs=epochs)
            bundle = {"mode": "tensorflow_rnn", "model": model, "tokenizer": tokenizer, "history": history.history}
        except Exception as exc:
            clf, vectorizer = train_fallback_classifier(train_df)
            bundle = {"mode": "fallback_tfidf_logreg", "model": clf, "vectorizer": vectorizer, "history": {}, "error": str(exc)}
    else:
        clf, vectorizer = train_fallback_classifier(train_df)
        bundle = {"mode": "fallback_tfidf_logreg", "model": clf, "vectorizer": vectorizer, "history": {}}

    eval_df = predict_texts(real_test_df["text"].astype(str).head(200).tolist(), bundle)
    eval_df["true_label"] = real_test_df["label"].astype(int).head(len(eval_df)).values
    acc = accuracy_score(eval_df["true_label"], eval_df["predicted_label"]) if len(eval_df) else np.nan

    metadata = {
        "synthetic_rows": int(len(synthetic_df)),
        "real_train_rows": int(len(real_train_df)),
        "real_test_rows": int(len(real_test_df)),
        "train_rows_total": int(len(train_df)),
        "data_source": source_name,
        "model_mode": bundle["mode"],
        "accuracy": None if pd.isna(acc) else float(acc),
    }
    return bundle, metadata, eval_df


def make_download_bundle(eval_df: pd.DataFrame, metadata: dict):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.csv", eval_df.to_csv(index=False))
        zf.writestr("manifest.json", json.dumps(metadata, indent=2))
    mem_zip.seek(0)
    return mem_zip


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Simple RNN sentiment classification with synthetic validation first, then real IMDB data through the same pipeline.")

    with st.sidebar:
        st.header("Settings")
        use_real = st.checkbox("Use real IMDB data after synthetic validation", value=True)
        max_train = st.slider("Real IMDB training rows", 200, 3000, 1000, step=200)
        epochs = st.slider("Training epochs", 1, 5, 2)

    bundle, metadata, eval_df = build_model(max_train=max_train, use_real=use_real, epochs=epochs)

    st.subheader("Pipeline Status")
    st.json(metadata)

    st.subheader("Try Your Own Review")
    review = st.text_area("Enter a movie review", value="This movie was surprisingly emotional and well acted.")
    if st.button("Predict Sentiment"):
        pred = predict_texts([review], bundle)
        st.dataframe(pred)

    st.subheader("Evaluation Sample")
    st.dataframe(eval_df.head(50))

    zip_file = make_download_bundle(eval_df, metadata)
    st.download_button(
        "Download predictions + manifest ZIP",
        data=zip_file,
        file_name="simple_rnn_imdb_outputs.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()
