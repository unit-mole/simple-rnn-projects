import numpy as np

from src.embedding_analysis import pca_projection
from src.embedding_pipeline import load_embedding_bundle


def test_nearest_words_excludes_query():
    bundle = load_embedding_bundle()
    result = bundle.nearest("quality", top_k=5)
    assert len(result) == 5
    assert "quality" not in result["word"].tolist()
    assert result["similarity"].between(-1, 1).all()


def test_sentence_embedding_reports_oov_tokens():
    bundle = load_embedding_bundle()
    result = bundle.encode_sentence("quality completelyunknownword defect")
    assert "completelyunknownword" in result["oov_tokens"]
    assert result["token_embedding_matrix"].shape[1] == bundle.metadata["embedding_dimension"]
    assert np.isfinite(result["sentence_vector"]).all()


def test_pca_projection_has_two_coordinates():
    bundle = load_embedding_bundle()
    projection = pca_projection(
        ["quality", "defect", "failure"],
        bundle.word_to_index,
        bundle.embedding_matrix,
    )
    assert list(projection.columns) == ["word", "x", "y"]
    assert projection.shape == (3, 3)
