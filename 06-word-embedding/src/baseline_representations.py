from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def build_lsa_term_embeddings(
    texts: list[str],
    max_features: int = 500,
    embedding_dimension: int = 32,
    random_seed: int = 42,
):
    """Reproduce the supplied project's TF-IDF + TruncatedSVD term-vector baseline."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        max_features=max_features,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    safe_dimension = max(2, min(embedding_dimension, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=safe_dimension, random_state=random_seed)
    document_embeddings = svd.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    term_embeddings = svd.components_.T.astype(np.float32)
    return vectorizer, svd, terms, term_embeddings, document_embeddings
