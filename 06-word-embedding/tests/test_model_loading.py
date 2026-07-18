import numpy as np

from src.embedding_pipeline import load_embedding_bundle


def test_saved_model_and_embedding_matrix_load():
    bundle = load_embedding_bundle()
    assert bundle.embedding_matrix.shape == (
        bundle.metadata["vocabulary_size"],
        bundle.metadata["embedding_dimension"],
    )
    assert np.isfinite(bundle.embedding_matrix).all()
    assert bundle.model.embedding.weight.shape[0] == bundle.metadata["vocabulary_size"]
