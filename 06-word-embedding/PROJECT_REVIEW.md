# Project Review and Transformation Summary

## What the supplied project was doing

The supplied notebook and Streamlit app implemented **TF-IDF + Truncated SVD**. TF-IDF
created sparse document features, while SVD projected those features into dense latent
vectors for documents and terms. The app then used cosine similarity for nearest-word
lookup and semantic document search.

That is a valid **latent semantic analysis (LSA)** workflow, but it is not a learned
neural word-embedding layer in the same sense as Word2Vec, GloVe, or a trainable
`Embedding` layer.

## Main improvements

1. Retained TF-IDF + SVD as an explicit baseline rather than presenting it as the main
   neural embedding model.
2. Added a PyTorch skip-gram model with a trainable `Embedding` layer.
3. Added deterministic cleaning, tokenization, PAD/UNK handling, and vocabulary saving.
4. Created sentence-bounded center/context pairs so windows do not cross sentence boundaries.
5. Added held-out validation pairs and early stopping.
6. Saved the model, vocabulary, embedding matrix, metadata, configuration, and checksums.
7. Added nearest-word analysis, sentence-vector inspection, PCA visualization, and
   embedding-based semantic search.
8. Added domain purity@5 to compare the neural model with the supplied LSA baseline.
9. Rebuilt the Streamlit app so it loads saved artifacts and never retrains at startup.
10. Added tests, validation, CI, hosting instructions, data documentation, and a model card.

## Scope decision

A second sentiment or topic classifier was intentionally not forced into the project.
The portfolio already contains multiple Simple RNN classification projects. This project
isolates the representation-learning stage that supplies dense vectors to recurrent
networks and therefore adds a distinct skill to the portfolio.
