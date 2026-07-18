from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_representations import build_lsa_term_embeddings
from src.config import (
    EMBEDDING_MATRIX_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    TRAINING_CONFIG_PATH,
    VOCAB_PATH,
)
from src.data_preprocessing import load_included_corpus, load_topic_lexicon
from src.embedding_analysis import (
    build_sentence_embedding_table,
    domain_purity_at_k,
    nearest_words,
    pca_projection,
    semantic_search,
    sentence_embedding,
)
from src.embedding_training import extract_embedding_matrix, train_skipgram_model
from src.model_evaluation import evaluate_context_prediction
from src.sequence_generation import create_skipgram_pairs, split_pairs
from src.text_preprocessing import build_vocabulary
from src.visualization import (
    save_embedding_norm_distribution,
    save_embedding_projection,
    save_training_curve,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the educational skip-gram word-embedding model."
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_included_corpus()
    tokenized = corpus["tokens"].tolist()
    word_to_index, index_to_word, token_counts = build_vocabulary(
        tokenized,
        min_frequency=args.min_frequency,
        max_vocabulary_size=args.max_vocab,
    )
    centers, contexts = create_skipgram_pairs(
        tokenized, word_to_index, window_size=args.window_size
    )
    split = split_pairs(
        centers,
        contexts,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
    )
    training_result = train_skipgram_model(
        split.train_centers,
        split.train_contexts,
        split.validation_centers,
        split.validation_contexts,
        vocabulary_size=len(word_to_index),
        embedding_dimension=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        random_seed=args.seed,
    )
    model = training_result.model
    embedding_matrix = extract_embedding_matrix(model)

    torch.save(model.state_dict(), MODEL_PATH)
    np.save(EMBEDDING_MATRIX_PATH, embedding_matrix)
    VOCAB_PATH.write_text(
        json.dumps(
            {
                "word_to_index": word_to_index,
                "token_frequency": {
                    token: int(token_counts[token])
                    for token in word_to_index
                    if token not in {"<PAD>", "<UNK>"}
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    training_config = vars(args)
    TRAINING_CONFIG_PATH.write_text(
        json.dumps(training_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metrics = evaluate_context_prediction(
        model,
        split.validation_centers,
        split.validation_contexts,
    )
    topic_lexicon = load_topic_lexicon()
    domain_purity, domain_detail = domain_purity_at_k(
        topic_lexicon,
        word_to_index,
        index_to_word,
        embedding_matrix,
        top_k=5,
    )
    metrics["domain_purity_at_5"] = domain_purity

    # Reproduce the supplied TF-IDF + TruncatedSVD approach as a documented baseline.
    _, _, lsa_terms, lsa_embeddings, _ = build_lsa_term_embeddings(
        corpus["text"].astype(str).tolist(),
        max_features=args.max_vocab,
        embedding_dimension=args.embedding_dim,
        random_seed=args.seed,
    )
    lsa_word_to_index = {str(word): index for index, word in enumerate(lsa_terms)}
    lsa_index_to_word = {index: word for word, index in lsa_word_to_index.items()}
    lsa_domain_purity, lsa_domain_detail = domain_purity_at_k(
        topic_lexicon,
        lsa_word_to_index,
        lsa_index_to_word,
        lsa_embeddings,
        top_k=5,
    )
    metrics["lsa_baseline_domain_purity_at_5"] = lsa_domain_purity
    metrics["training_pairs"] = int(len(split.train_centers))
    metrics["validation_pairs"] = int(len(split.validation_centers))
    metrics["vocabulary_size"] = int(len(word_to_index))
    metrics["embedding_dimension"] = int(args.embedding_dim)
    metrics["epochs_completed"] = int(training_result.epochs_completed)
    metrics["corpus_sentences"] = int(len(corpus))
    metrics["corpus_tokens"] = int(sum(map(len, tokenized)))

    history_path = OUTPUT_DIR / "training_history.csv"
    training_result.history.to_csv(history_path, index=False)
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    domain_detail.to_csv(OUTPUT_DIR / "domain_purity_details.csv", index=False)
    lsa_domain_detail.to_csv(OUTPUT_DIR / "lsa_domain_purity_details.csv", index=False)

    example_words = [
        "model",
        "embedding",
        "market",
        "doctor",
        "electricity",
        "quality",
        "defect",
    ]
    nearest_frames = []
    for word in example_words:
        frame = nearest_words(
            word,
            word_to_index,
            index_to_word,
            embedding_matrix,
            top_k=8,
        )
        frame.insert(0, "query", word)
        nearest_frames.append(frame)
    nearest_examples = pd.concat(nearest_frames, ignore_index=True)
    nearest_examples.to_csv(OUTPUT_DIR / "word_similarity_examples.csv", index=False)

    sentence_examples = [
        "neural language model learns semantic vectors",
        "market risk affects investment return",
        "doctor diagnosis supports patient treatment",
        "electricity demand changes grid load",
        "quality inspection finds manufacturing defect",
    ]
    sentence_rows = []
    for text in sentence_examples:
        result = sentence_embedding(text, word_to_index, embedding_matrix)
        sentence_rows.append(
            {
                "sentence": text,
                "cleaned_text": result["cleaned_text"],
                "tokens": " | ".join(result["tokens"]),
                "indices": " | ".join(map(str, result["indices"])),
                "known_tokens": len(result["known_tokens"]),
                "oov_tokens": " | ".join(result["oov_tokens"]),
                "embedding_shape": f"{len(result['known_tokens'])} x {args.embedding_dim}",
                "sentence_vector_norm": float(
                    np.linalg.norm(result["sentence_vector"])
                ),
            }
        )
    pd.DataFrame(sentence_rows).to_csv(
        OUTPUT_DIR / "sample_sentence_analysis.csv", index=False
    )

    searchable_corpus, corpus_vectors = build_sentence_embedding_table(
        corpus,
        word_to_index,
        embedding_matrix,
    )
    search_examples = []
    for query in [
        "language model training",
        "portfolio market risk",
        "patient medical treatment",
        "renewable electricity grid",
        "root cause quality failure",
    ]:
        result = semantic_search(
            query,
            searchable_corpus,
            corpus_vectors,
            word_to_index,
            embedding_matrix,
            top_k=3,
        )
        result.insert(0, "query", query)
        search_examples.append(result)
    pd.concat(search_examples, ignore_index=True).to_csv(
        OUTPUT_DIR / "semantic_search_examples.csv", index=False
    )

    representation_comparison = pd.DataFrame(
        [
            {
                "representation": "One-hot encoding",
                "vector_type": "Sparse",
                "captures_similarity": "No",
                "preserves_word_order": "No",
                "portfolio_role": "Conceptual baseline",
            },
            {
                "representation": "Bag-of-Words",
                "vector_type": "Sparse",
                "captures_similarity": "Limited",
                "preserves_word_order": "No",
                "portfolio_role": "Frequency baseline",
            },
            {
                "representation": "TF-IDF",
                "vector_type": "Sparse",
                "captures_similarity": "Document-level only",
                "preserves_word_order": "No",
                "portfolio_role": "Weighted baseline",
            },
            {
                "representation": "TF-IDF + Truncated SVD",
                "vector_type": "Dense",
                "captures_similarity": "Latent co-occurrence",
                "preserves_word_order": "No",
                "domain_purity_at_5": lsa_domain_purity,
                "portfolio_role": "Original-project LSA baseline",
            },
            {
                "representation": "Neural skip-gram embedding",
                "vector_type": "Dense",
                "captures_similarity": "Learned local context",
                "preserves_word_order": "Through context windows",
                "domain_purity_at_5": domain_purity,
                "portfolio_role": "Primary model",
            },
        ]
    )
    representation_comparison.to_csv(
        OUTPUT_DIR / "representation_comparison.csv", index=False
    )

    matrix_summary = {
        "shape": list(map(int, embedding_matrix.shape)),
        "dtype": str(embedding_matrix.dtype),
        "mean": float(embedding_matrix.mean()),
        "standard_deviation": float(embedding_matrix.std()),
        "minimum": float(embedding_matrix.min()),
        "maximum": float(embedding_matrix.max()),
        "mean_l2_norm_excluding_special_tokens": float(
            np.linalg.norm(embedding_matrix[2:], axis=1).mean()
        ),
    }
    (OUTPUT_DIR / "embedding_matrix_summary.json").write_text(
        json.dumps(matrix_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_training_curve(
        training_result.history,
        OUTPUT_DIR / "training_curve.png",
    )
    selected_words = [
        word
        for words in topic_lexicon.values()
        for word in words[:10]
        if word in word_to_index
    ]
    projection = pca_projection(selected_words, word_to_index, embedding_matrix)
    word_topic_lookup = {
        word: topic
        for topic, words in topic_lexicon.items()
        for word in words
    }
    projection.to_csv(OUTPUT_DIR / "embedding_projection_2d.csv", index=False)
    save_embedding_projection(
        projection,
        OUTPUT_DIR / "embedding_visualization_2d.png",
        topic_lookup=word_topic_lookup,
    )
    save_embedding_norm_distribution(
        embedding_matrix,
        OUTPUT_DIR / "embedding_norm_distribution.png",
    )

    metadata = {
        "project_name": "Word Embedding and NLP Representation Learning",
        "model_type": "neural_skipgram_full_softmax",
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vocabulary_size": int(len(word_to_index)),
        "embedding_dimension": int(args.embedding_dim),
        "window_size": int(args.window_size),
        "padding_index": 0,
        "unknown_index": 1,
        "corpus_sentences": int(len(corpus)),
        "corpus_tokens": int(sum(map(len, tokenized))),
        "training_pairs": int(len(split.train_centers)),
        "validation_pairs": int(len(split.validation_centers)),
        "metrics": metrics,
        "artifacts": {
            "model": MODEL_PATH.name,
            "vocabulary": VOCAB_PATH.name,
            "embedding_matrix": EMBEDDING_MATRIX_PATH.name,
            "training_config": TRAINING_CONFIG_PATH.name,
        },
        "data_scope": (
            "Privacy-safe synthetic educational corpus based on the domains present "
            "in the supplied notebook. Optional 20 Newsgroups loading is available "
            "for local experimentation but is not required by the deployed app."
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    checksums = {
        path.name: sha256(path)
        for path in [
            MODEL_PATH,
            VOCAB_PATH,
            EMBEDDING_MATRIX_PATH,
            METADATA_PATH,
            TRAINING_CONFIG_PATH,
            PROJECT_ROOT / "data" / "sample_text.csv",
        ]
    }
    (MODEL_DIR / "artifact_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
