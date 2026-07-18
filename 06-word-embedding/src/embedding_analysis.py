from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .config import PAD_TOKEN, UNK_TOKEN
from .text_preprocessing import encode_tokens, tokenize


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def nearest_words(
    query_word: str,
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    embedding_matrix: np.ndarray,
    top_k: int = 8,
) -> pd.DataFrame:
    query = tokenize(query_word)
    if len(query) != 1 or query[0] not in word_to_index:
        return pd.DataFrame(
            [{"word": query_word, "similarity": np.nan, "status": "out_of_vocabulary"}]
        )
    normalized = l2_normalize_rows(embedding_matrix)
    query_index = word_to_index[query[0]]
    similarities = normalized @ normalized[query_index]
    order = np.argsort(-similarities)
    rows: list[dict[str, object]] = []
    excluded = {query_index, word_to_index.get(PAD_TOKEN), word_to_index.get(UNK_TOKEN)}
    for index in order:
        if int(index) in excluded:
            continue
        rows.append(
            {
                "word": index_to_word[int(index)],
                "similarity": float(similarities[int(index)]),
                "status": "in_vocabulary",
            }
        )
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def sentence_embedding(
    text: str,
    word_to_index: dict[str, int],
    embedding_matrix: np.ndarray,
) -> dict[str, object]:
    tokens = tokenize(text)
    indices = encode_tokens(tokens, word_to_index)
    known_mask = [
        token in word_to_index and token not in {PAD_TOKEN, UNK_TOKEN}
        for token in tokens
    ]
    known_indices = [index for index, is_known in zip(indices, known_mask) if is_known]
    if known_indices:
        token_matrix = embedding_matrix[np.asarray(known_indices, dtype=np.int64)]
        vector = token_matrix.mean(axis=0)
    else:
        token_matrix = np.empty((0, embedding_matrix.shape[1]), dtype=np.float32)
        vector = np.zeros(embedding_matrix.shape[1], dtype=np.float32)
    return {
        "cleaned_text": " ".join(tokens),
        "tokens": tokens,
        "indices": indices,
        "known_tokens": [t for t, flag in zip(tokens, known_mask) if flag],
        "oov_tokens": [t for t, flag in zip(tokens, known_mask) if not flag],
        "token_embedding_matrix": token_matrix,
        "sentence_vector": vector,
    }


def build_sentence_embedding_table(
    corpus: pd.DataFrame,
    word_to_index: dict[str, int],
    embedding_matrix: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    vectors = []
    keep_rows = []
    for _, row in corpus.iterrows():
        result = sentence_embedding(str(row["text"]), word_to_index, embedding_matrix)
        if result["known_tokens"]:
            vectors.append(result["sentence_vector"])
            keep_rows.append(row)
    return pd.DataFrame(keep_rows).reset_index(drop=True), np.asarray(vectors)


def semantic_search(
    query: str,
    corpus: pd.DataFrame,
    corpus_vectors: np.ndarray,
    word_to_index: dict[str, int],
    embedding_matrix: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    query_result = sentence_embedding(query, word_to_index, embedding_matrix)
    if not query_result["known_tokens"]:
        return pd.DataFrame(
            [{"text": query, "similarity": np.nan, "status": "no_known_tokens"}]
        )
    similarities = cosine_similarity(
        np.asarray(query_result["sentence_vector"]).reshape(1, -1),
        corpus_vectors,
    )[0]
    order = np.argsort(-similarities)[:top_k]
    output = corpus.iloc[order][["sentence_id", "topic", "text", "source_type"]].copy()
    output["similarity"] = similarities[order]
    output["status"] = "matched"
    return output.reset_index(drop=True)


def pca_projection(
    words: Iterable[str],
    word_to_index: dict[str, int],
    embedding_matrix: np.ndarray,
) -> pd.DataFrame:
    unique_words = [
        word for word in dict.fromkeys(words)
        if word in word_to_index and word not in {PAD_TOKEN, UNK_TOKEN}
    ]
    if len(unique_words) < 2:
        raise ValueError("At least two in-vocabulary words are required for PCA.")
    matrix = np.vstack([embedding_matrix[word_to_index[word]] for word in unique_words])
    coordinates = PCA(n_components=2).fit_transform(matrix)
    return pd.DataFrame(
        {"word": unique_words, "x": coordinates[:, 0], "y": coordinates[:, 1]}
    )


def domain_purity_at_k(
    topic_lexicon: dict[str, list[str]],
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    embedding_matrix: np.ndarray,
    top_k: int = 5,
) -> tuple[float, pd.DataFrame]:
    word_to_topic = {
        word: topic
        for topic, words in topic_lexicon.items()
        for word in words
        if word in word_to_index
    }
    rows: list[dict[str, object]] = []
    for word, expected_topic in word_to_topic.items():
        nearest = nearest_words(
            word, word_to_index, index_to_word, embedding_matrix, top_k=top_k
        )
        neighbors = nearest.loc[
            nearest["status"].eq("in_vocabulary"), "word"
        ].astype(str).tolist()
        matches = sum(word_to_topic.get(neighbor) == expected_topic for neighbor in neighbors)
        purity = matches / max(len(neighbors), 1)
        rows.append(
            {
                "word": word,
                "expected_topic": expected_topic,
                "same_topic_neighbors": matches,
                "neighbors_evaluated": len(neighbors),
                "purity_at_k": purity,
            }
        )
    frame = pd.DataFrame(rows)
    return float(frame["purity_at_k"].mean()), frame
