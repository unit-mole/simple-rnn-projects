from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_curve(history: pd.DataFrame, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["epoch"], history["training_loss"], label="Training loss")
    axis.plot(history["epoch"], history["validation_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Skip-gram training and validation loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_embedding_projection(
    projection: pd.DataFrame,
    output_path: str | Path,
    topic_lookup: dict[str, str] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 8))
    if topic_lookup:
        topics = [topic_lookup.get(word, "other") for word in projection["word"]]
        topic_values = {topic: index for index, topic in enumerate(sorted(set(topics)))}
        values = [topic_values[topic] for topic in topics]
        scatter = axis.scatter(projection["x"], projection["y"], c=values, s=55)
        handles, _ = scatter.legend_elements()
        axis.legend(handles, list(topic_values.keys()), title="Topic", loc="best")
    else:
        axis.scatter(projection["x"], projection["y"], s=55)
    for _, row in projection.iterrows():
        axis.annotate(
            str(row["word"]),
            (float(row["x"]), float(row["y"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Two-dimensional PCA view of learned word embeddings")
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_embedding_norm_distribution(
    embedding_matrix: np.ndarray,
    output_path: str | Path,
) -> None:
    norms = np.linalg.norm(embedding_matrix[2:], axis=1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(norms, bins=20)
    axis.set_title("Distribution of learned embedding-vector norms")
    axis.set_xlabel("L2 norm")
    axis.set_ylabel("Number of words")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
