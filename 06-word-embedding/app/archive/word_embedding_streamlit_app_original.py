# -*- coding: utf-8 -*-
import os
import re
import json
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

APP_TITLE = "Word Embedding Lab — Synthetic to Real Text"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("outputs") / f"word_embedding_streamlit_{RUN_ID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_synthetic_corpus():
    rows = [
        ("synthetic_ai_1", "neural networks learn word embeddings from text sequences", "synthetic"),
        ("synthetic_ai_2", "deep learning models create semantic vectors for language", "synthetic"),
        ("synthetic_ai_3", "transformer models understand context using token representations", "synthetic"),
        ("synthetic_fin_1", "stocks bonds portfolio market risk return investment", "synthetic"),
        ("synthetic_fin_2", "trading systems forecast price movement and volatility", "synthetic"),
        ("synthetic_health_1", "patients doctors hospitals treatment diagnosis medicine", "synthetic"),
        ("synthetic_health_2", "clinical notes contain symptoms diseases and prescriptions", "synthetic"),
        ("synthetic_energy_1", "electricity demand load forecast grid power consumption", "synthetic"),
        ("synthetic_energy_2", "renewable energy solar wind battery storage grid", "synthetic"),
        ("synthetic_quality_1", "manufacturing defects inspection quality process control", "synthetic"),
        ("synthetic_quality_2", "root cause analysis reduces failures and improves yield", "synthetic"),
    ]
    return pd.DataFrame(rows, columns=["doc_id", "text", "source_type"])


def load_real_corpus(max_docs=300):
    try:
        from sklearn.datasets import fetch_20newsgroups
        ds = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        texts = [t for t in ds.data if isinstance(t, str) and len(t.strip()) > 100][:max_docs]
        df = pd.DataFrame({
            "doc_id": [f"real_20news_{i:04d}" for i in range(len(texts))],
            "text": texts,
            "source_type": "real_20newsgroups",
        })
        return df, "real_20newsgroups"
    except Exception as e:
        fallback = [
            "Public news articles discuss technology markets medicine energy and science in real-world language.",
            "A computer graphics discussion describes images rendering hardware software and visual systems.",
            "A sports article describes teams players scores tournaments and season performance.",
            "A medical article describes health diagnosis treatment hospitals patients and research studies.",
            "A science article describes space missions astronomy physics experiments and discovery.",
        ]
        df = pd.DataFrame({
            "doc_id": [f"fallback_real_{i:04d}" for i in range(len(fallback))],
            "text": fallback,
            "source_type": "fallback_public_style_real_text",
        })
        return df, f"fallback_used_because: {e}"


def build_term_document_embeddings(texts, max_features=1500, n_components=50):
    clean_texts = [normalize_text(t) for t in texts]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=1)
    X = vectorizer.fit_transform(clean_texts)
    n_components = min(n_components, max(2, min(X.shape) - 1)) if min(X.shape) > 2 else 2
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    doc_embeddings = svd.fit_transform(X)
    terms = vectorizer.get_feature_names_out()
    # term embeddings from V^T columns projected by singular values
    term_embeddings = svd.components_.T
    return vectorizer, svd, terms, term_embeddings, doc_embeddings


def nearest_words(query_word, terms, term_embeddings, top_k=10):
    q = normalize_text(query_word)
    if q not in set(terms):
        return pd.DataFrame({"message": [f"'{query_word}' is not in vocabulary."]})
    idx = list(terms).index(q)
    sims = cosine_similarity(term_embeddings[idx:idx+1], term_embeddings)[0]
    order = np.argsort(-sims)
    rows = []
    for j in order[:top_k+1]:
        if j == idx:
            continue
        rows.append({"word": terms[j], "similarity": float(sims[j])})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def search_documents(query, corpus_df, vectorizer, svd, doc_embeddings, top_k=5):
    q_vec = svd.transform(vectorizer.transform([normalize_text(query)]))
    sims = cosine_similarity(q_vec, doc_embeddings)[0]
    order = np.argsort(-sims)[:top_k]
    out = corpus_df.iloc[order][["doc_id", "source_type", "text"]].copy()
    out["similarity"] = sims[order]
    return out


def save_outputs(corpus_df, terms, term_embeddings, doc_embeddings, search_df, nearest_df, data_source):
    term_df = pd.DataFrame(term_embeddings, index=terms).reset_index().rename(columns={"index": "word"})
    doc_emb_df = pd.DataFrame(doc_embeddings)
    doc_emb_df.insert(0, "doc_id", corpus_df["doc_id"].values)
    summary_df = pd.DataFrame([{
        "run_id": RUN_ID,
        "documents": len(corpus_df),
        "synthetic_docs": int((corpus_df["source_type"] == "synthetic").sum()),
        "real_docs": int((corpus_df["source_type"] != "synthetic").sum()),
        "vocab_size": len(terms),
        "real_data_source": data_source,
    }])
    corpus_df.to_csv(OUTPUT_DIR / "corpus.csv", index=False)
    term_df.to_csv(OUTPUT_DIR / "word_embeddings.csv", index=False)
    doc_emb_df.to_csv(OUTPUT_DIR / "document_embeddings.csv", index=False)
    search_df.to_csv(OUTPUT_DIR / "search_results.csv", index=False)
    nearest_df.to_csv(OUTPUT_DIR / "nearest_words.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "word_embedding_report.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        corpus_df.head(200).to_excel(writer, index=False, sheet_name="corpus_sample")
        term_df.head(200).to_excel(writer, index=False, sheet_name="word_embeddings")
        search_df.to_excel(writer, index=False, sheet_name="search_results")
        nearest_df.to_excel(writer, index=False, sheet_name="nearest_words")
    manifest = {"run_id": RUN_ID, "output_dir": str(OUTPUT_DIR), "real_data_source": data_source}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    zip_path = OUTPUT_DIR.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in OUTPUT_DIR.rglob("*"):
            zf.write(fp, fp.relative_to(OUTPUT_DIR.parent))
    return OUTPUT_DIR / "word_embedding_report.xlsx", zip_path


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Synthetic validation first, then real public text through the same word-embedding pipeline.")

with st.sidebar:
    st.header("Settings")
    max_real_docs = st.slider("Max real documents", 20, 500, 150, 10)
    max_features = st.slider("Vocabulary size", 100, 3000, 1200, 100)
    n_components = st.slider("Embedding dimensions", 2, 100, 50, 1)
    query_word = st.text_input("Nearest-word query", value="learning")
    search_query = st.text_input("Document search query", value="neural language models")
    run_button = st.button("Build embedding system")

if run_button:
    synthetic_df = make_synthetic_corpus()
    real_df, data_source = load_real_corpus(max_real_docs)
    corpus_df = pd.concat([synthetic_df, real_df], ignore_index=True)
    corpus_df["clean_text"] = corpus_df["text"].map(normalize_text)

    vectorizer, svd, terms, term_embeddings, doc_embeddings = build_term_document_embeddings(
        corpus_df["clean_text"].tolist(), max_features=max_features, n_components=n_components
    )
    nearest_df = nearest_words(query_word, terms, term_embeddings, top_k=10)
    search_df = search_documents(search_query, corpus_df, vectorizer, svd, doc_embeddings, top_k=10)

    st.subheader("Corpus Source Counts")
    st.dataframe(corpus_df["source_type"].value_counts().reset_index().rename(columns={"index": "source_type", "source_type": "count"}))
    st.write("Real data source:", data_source)

    st.subheader("Nearest Words")
    st.dataframe(nearest_df)

    st.subheader("Semantic Document Search")
    st.dataframe(search_df)

    excel_path, zip_path = save_outputs(corpus_df, terms, term_embeddings, doc_embeddings, search_df, nearest_df, data_source)
    st.success(f"Outputs saved to {OUTPUT_DIR}")
    st.download_button("Download Excel report", data=excel_path.read_bytes(), file_name=excel_path.name)
    st.download_button("Download output ZIP", data=zip_path.read_bytes(), file_name=zip_path.name)
else:
    st.info("Choose settings and click 'Build embedding system'.")
