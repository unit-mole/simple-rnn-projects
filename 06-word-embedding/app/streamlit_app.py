from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.data_preprocessing import load_included_corpus, load_topic_lexicon
from src.embedding_analysis import (
    build_sentence_embedding_table,
    pca_projection,
    semantic_search,
)
from src.embedding_pipeline import load_embedding_bundle
from src.text_preprocessing import clean_text


APP_TITLE = "Word Embedding and NLP Representation Learning"
GITHUB_URL = (
    "https://github.com/unit-mole/simple-rnn-projects/tree/main/06-word-embedding"
)
LIVE_DEMO_PLACEHOLDER = "#"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_bundle():
    return load_embedding_bundle()


@st.cache_data
def get_corpus():
    return load_included_corpus()


@st.cache_data
def get_search_index():
    bundle = get_bundle()
    corpus = get_corpus()
    return build_sentence_embedding_table(
        corpus,
        bundle.word_to_index,
        bundle.embedding_matrix,
    )


@st.cache_data
def get_representation_comparison():
    path = OUTPUT_DIR / "representation_comparison.csv"
    return pd.read_csv(path)


def plot_projection(frame: pd.DataFrame, title: str):
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(frame["x"], frame["y"], s=60)
    for _, row in frame.iterrows():
        axis.annotate(
            str(row["word"]),
            (float(row["x"]), float(row["y"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_title(title)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def dataframe_download(frame: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


bundle = get_bundle()
corpus = get_corpus()
metadata = bundle.metadata
metrics = metadata["metrics"]

st.title(APP_TITLE)
st.caption(
    "Explore learned dense word vectors, nearest-neighbor relationships, sentence "
    "representations, and semantic search using a saved neural skip-gram model."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Word Explorer",
            "Sentence Embedding",
            "Semantic Search",
            "Model Performance",
            "Project Overview",
        ],
    )
    st.divider()
    st.markdown(f"[View the GitHub project]({GITHUB_URL})")
    st.caption(
        "The deployed app loads saved artifacts. It does not download external data "
        "or retrain the model during startup."
    )


if page == "Word Explorer":
    st.subheader("Explore a learned word vector")
    vocabulary = [
        word
        for word in bundle.word_to_index
        if word not in {"<PAD>", "<UNK>"}
    ]
    sample_words = [
        word
        for word in [
            "model",
            "embedding",
            "market",
            "doctor",
            "electricity",
            "quality",
            "defect",
        ]
        if word in bundle.word_to_index
    ]
    col1, col2 = st.columns([2, 1])
    with col1:
        input_mode = st.radio(
            "Input method",
            ["Choose a sample", "Enter a word"],
            horizontal=True,
        )
        if input_mode == "Choose a sample":
            query_word = st.selectbox("Sample word", sample_words)
        else:
            query_word = st.text_input("Word", value="model").strip().lower()
    with col2:
        top_k = st.slider("Nearest words", min_value=3, max_value=15, value=8)

    if query_word:
        nearest = bundle.nearest(query_word, top_k=top_k)
        if nearest["status"].eq("out_of_vocabulary").all():
            st.warning(
                f"`{query_word}` is not in the saved vocabulary. Try one of the "
                f"{len(vocabulary)} available words."
            )
        else:
            query_index = bundle.word_to_index[query_word]
            vector = bundle.embedding_matrix[query_index]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vocabulary index", query_index)
            m2.metric("Embedding dimensions", len(vector))
            m3.metric("Vector L2 norm", f"{np.linalg.norm(vector):.3f}")
            m4.metric("Nearest-word score", f"{nearest.iloc[0]['similarity']:.3f}")

            st.markdown("#### Embedding-vector preview")
            preview = pd.DataFrame(
                {
                    "dimension": [f"dim_{i:02d}" for i in range(min(12, len(vector)))],
                    "value": vector[:12],
                }
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)

            st.markdown("#### Nearest words by cosine similarity")
            display_nearest = nearest[["word", "similarity"]].copy()
            display_nearest["similarity"] = display_nearest["similarity"].round(4)
            st.dataframe(display_nearest, use_container_width=True, hide_index=True)
            dataframe_download(
                display_nearest,
                "Download nearest words",
                f"{query_word}_nearest_words.csv",
            )

            projection_words = [query_word] + display_nearest["word"].tolist()
            projection = pca_projection(
                projection_words,
                bundle.word_to_index,
                bundle.embedding_matrix,
            )
            st.markdown("#### Local two-dimensional projection")
            st.pyplot(
                plot_projection(
                    projection,
                    f"PCA view of {query_word} and its nearest neighbors",
                ),
                use_container_width=True,
            )
            st.info(
                "Cosine similarity measures vector direction, while PCA compresses the "
                "32-dimensional vectors into two dimensions for visualization. The 2D "
                "plot is illustrative and cannot preserve every high-dimensional distance."
            )


elif page == "Sentence Embedding":
    st.subheader("Convert a sentence into token and embedding representations")
    samples = [
        "neural language model learns semantic vectors",
        "market risk affects investment return",
        "doctor diagnosis supports patient treatment",
        "electricity demand changes grid load",
        "quality inspection finds manufacturing defect",
    ]
    input_mode = st.radio(
        "Input method",
        ["Choose a sample sentence", "Enter a sentence"],
        horizontal=True,
    )
    if input_mode == "Choose a sample sentence":
        sentence = st.selectbox("Sample sentence", samples)
    else:
        sentence = st.text_area(
            "Sentence",
            value="quality inspection helps identify a manufacturing defect",
            height=100,
        )

    result = bundle.encode_sentence(sentence)
    st.markdown("#### Preprocessing and tokenization")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Cleaned text", result["cleaned_text"], disabled=True)
        st.write("**Tokens**")
        st.code(result["tokens"])
    with c2:
        st.write("**Token indices**")
        st.code(result["indices"])
        st.write(
            f"**Known tokens:** {len(result['known_tokens'])}  \n"
            f"**Out-of-vocabulary tokens:** "
            f"{', '.join(result['oov_tokens']) or 'None'}"
        )

    token_matrix = np.asarray(result["token_embedding_matrix"])
    sentence_vector = np.asarray(result["sentence_vector"])
    c3, c4, c5 = st.columns(3)
    c3.metric(
        "Token embedding matrix",
        f"{token_matrix.shape[0]} × {token_matrix.shape[1] if token_matrix.ndim == 2 else 0}",
    )
    c4.metric("Sentence vector dimensions", len(sentence_vector))
    c5.metric("Sentence-vector norm", f"{np.linalg.norm(sentence_vector):.3f}")

    if token_matrix.size:
        matrix_preview = pd.DataFrame(
            token_matrix[:, : min(8, token_matrix.shape[1])],
            index=result["known_tokens"],
            columns=[
                f"dim_{index:02d}"
                for index in range(min(8, token_matrix.shape[1]))
            ],
        )
        st.markdown("#### Token embedding matrix preview")
        st.dataframe(matrix_preview, use_container_width=True)

        vector_frame = pd.DataFrame(
            {
                "dimension": [
                    f"dim_{index:02d}"
                    for index in range(len(sentence_vector))
                ],
                "value": sentence_vector,
            }
        )
        st.markdown("#### Mean-pooled sentence vector")
        st.dataframe(vector_frame.head(12), use_container_width=True, hide_index=True)
        dataframe_download(
            vector_frame,
            "Download sentence vector",
            "sentence_embedding.csv",
        )
    else:
        st.warning("The sentence contains no words from the saved vocabulary.")

    st.info(
        "Each known word is mapped to a dense 32-dimensional vector. The demo creates "
        "a lightweight sentence representation by averaging the known token vectors. "
        "A Simple RNN would instead consume the token vectors in sequence order."
    )


elif page == "Semantic Search":
    st.subheader("Search the included sentence corpus with learned embeddings")
    searchable_corpus, corpus_vectors = get_search_index()
    query = st.text_input(
        "Search query",
        value="root cause quality failure",
    )
    top_k = st.slider("Number of results", 3, 12, 5)
    if query:
        results = semantic_search(
            query,
            searchable_corpus,
            corpus_vectors,
            bundle.word_to_index,
            bundle.embedding_matrix,
            top_k=top_k,
        )
        if results["status"].eq("no_known_tokens").all():
            st.warning("The query does not contain any in-vocabulary words.")
        else:
            display_results = results[
                ["sentence_id", "topic", "similarity", "text"]
            ].copy()
            display_results["similarity"] = display_results["similarity"].round(4)
            st.dataframe(display_results, use_container_width=True, hide_index=True)
            dataframe_download(
                display_results,
                "Download search results",
                "semantic_search_results.csv",
            )
            st.caption(
                "Search uses cosine similarity between mean-pooled word embeddings. "
                "It is an educational semantic-search baseline, not a production search engine."
            )


elif page == "Model Performance":
    st.subheader("Saved-model evaluation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation loss", f"{metrics['validation_loss']:.3f}")
    c2.metric("Top-5 context accuracy", f"{metrics['validation_top5_accuracy']:.1%}")
    c3.metric("Domain purity@5", f"{metrics['domain_purity_at_5']:.1%}")
    c4.metric(
        "LSA baseline purity@5",
        f"{metrics['lsa_baseline_domain_purity_at_5']:.1%}",
    )

    st.markdown("#### Metric interpretation")
    st.markdown(
        """
- **Validation loss** evaluates next-context-word prediction on held-out skip-gram pairs.
- **Top-5 context accuracy** checks whether the actual context word appears among the five highest-scoring predictions.
- **Domain purity@5** measures how often a word's nearest neighbors come from the same curated domain.
- **LSA baseline purity@5** evaluates the supplied TF-IDF + TruncatedSVD approach using the same domain-neighborhood criterion.
"""
    )

    comparison = get_representation_comparison()
    st.markdown("#### Representation comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    image_col1, image_col2 = st.columns(2)
    with image_col1:
        st.image(
            str(OUTPUT_DIR / "training_curve.png"),
            caption="Training and validation loss",
            use_container_width=True,
        )
    with image_col2:
        st.image(
            str(OUTPUT_DIR / "embedding_norm_distribution.png"),
            caption="Learned embedding-vector norms",
            use_container_width=True,
        )

    st.image(
        str(OUTPUT_DIR / "embedding_visualization_2d.png"),
        caption="PCA projection of selected vocabulary words",
        use_container_width=True,
    )

    st.warning(
        "The evaluation corpus is deliberately small and synthetic. Strong domain "
        "purity shows that the model learned the curated topic structure; it does not "
        "prove broad real-world language understanding."
    )


else:
    st.subheader("Project overview")
    st.markdown(
        """
### Problem statement

> How can words be represented as dense numerical vectors so that a neural model can
> learn contextual similarity and sequence-ready representations from text?

### Primary workflow

```text
Privacy-safe sentence corpus
        ↓
Deterministic text cleaning and tokenization
        ↓
Training-only vocabulary with PAD and UNK handling
        ↓
Sentence-bounded skip-gram center/context pairs
        ↓
PyTorch Embedding layer
        ↓
Context-word prediction with cross-entropy loss
        ↓
Saved embedding matrix and vocabulary
        ↓
Nearest-word, sentence-vector, PCA, and semantic-search analysis
```

### Why this belongs in a Simple RNN portfolio

An embedding layer converts integer word indices into dense vectors before sequence
models process them. In a typical NLP Simple RNN pipeline, the learned vector sequence
becomes the recurrent network's input. This project isolates and explains that
representation-learning stage rather than repeating another classification task.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vocabulary size", metadata["vocabulary_size"])
    c2.metric("Embedding dimensions", metadata["embedding_dimension"])
    c3.metric("Context window", metadata["window_size"])
    c4.metric("Corpus sentences", metadata["corpus_sentences"])

    st.markdown("### Responsible-use note")
    st.info(
        "This educational model learns from a small curated corpus. Its nearest-word "
        "relationships are incomplete and may reflect corpus design choices. Do not "
        "treat the vectors as authoritative definitions, fairness evidence, or a "
        "production semantic system."
    )

    st.markdown("### Skills demonstrated")
    st.write(
        "`NLP preprocessing` · `Vocabulary management` · `PyTorch Embedding` · "
        "`Skip-gram training` · `Cosine similarity` · `PCA` · `Semantic search` · "
        "`Artifact management` · `Testing` · `Streamlit deployment`"
    )
