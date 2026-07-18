"""Train, evaluate, and export the character-level Simple RNN portfolio model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from src.data_preprocessing import load_text_corpus
from src.model_evaluation import (
    build_markov_model,
    generate_markov_text,
    generated_text_metrics,
    perplexity_from_loss,
)
from src.model_training import TrainingConfig, evaluate_model, train_model
from src.sequence_generation import build_train_validation_sequences
from src.text_generator import TextGenerationBundle, generate_text
from src.visualization import (
    plot_sequence_summary,
    plot_temperature_comparison,
    plot_training_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "sample_text.txt")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--max-characters", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    config = TrainingConfig(
        sequence_length=args.sequence_length,
        step=args.step,
        epochs=args.epochs,
    )
    corpus = load_text_corpus(args.corpus, max_characters=args.max_characters)
    dataset = build_train_validation_sequences(
        corpus,
        sequence_length=config.sequence_length,
        step=config.step,
        validation_fraction=config.validation_fraction,
    )

    model, history_df, model_path = train_model(dataset, config=config, model_dir=models_dir)
    history_df.to_csv(outputs_dir / "training_history.csv", index=False)

    validation_loss, validation_accuracy = evaluate_model(
        model, dataset, batch_size=config.batch_size
    )

    metadata = {
        "project_name": "Character-Level Text Generation using Simple RNN",
        "model_type": "PyTorch nn.RNN (Simple RNN)",
        "task": "next-character prediction",
        "corpus_file": str(args.corpus.name),
        "corpus_characters": len(corpus),
        "train_characters": dataset.train_characters,
        "validation_characters": dataset.validation_characters,
        "training_sequences": int(len(dataset.X_train)),
        "validation_sequences": int(len(dataset.X_validation)),
        "sequence_length": config.sequence_length,
        "step": config.step,
        "vocabulary_size": dataset.vocabulary.size,
        "embedding_dim": config.embedding_dim,
        "rnn_units": config.rnn_units,
        "dropout_rate": config.dropout_rate,
        "validation_loss": float(validation_loss),
        "validation_accuracy": float(validation_accuracy),
        "validation_perplexity": perplexity_from_loss(validation_loss),
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_size_bytes": int(model_path.stat().st_size),
        "model_sha256": sha256_file(model_path),
        "corpus_sha256": sha256_file(args.corpus),
        "default_generation": {
            "length": 300,
            "temperature": 0.7,
            "top_k": 20,
            "random_seed": 42,
        },
        "responsible_use": "Educational portfolio demonstration only; human review is required.",
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (outputs_dir / "model_metrics.json").write_text(
        json.dumps(
            {
                "validation_loss": metadata["validation_loss"],
                "validation_accuracy": metadata["validation_accuracy"],
                "validation_perplexity": metadata["validation_perplexity"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = TextGenerationBundle(model=model, vocabulary=dataset.vocabulary, metadata=metadata)
    seed = "Alice was beginning"
    temperatures = [0.3, 0.7, 1.0, 1.2]
    generated_rows = []
    text_sections = []

    for temperature in temperatures:
        generated = generate_text(
            bundle,
            seed_text=seed,
            generation_length=400,
            temperature=temperature,
            top_k=20,
            random_seed=42,
        )
        metrics = generated_text_metrics(generated)
        generated_rows.append({"temperature": temperature, **metrics, "generated_text": generated})
        text_sections.append(
            f"TEMPERATURE: {temperature}\nSEED: {seed}\n\n{generated}\n\n{'=' * 80}\n"
        )

    generated_df = pd.DataFrame(generated_rows)
    generated_df.to_csv(outputs_dir / "temperature_generation_metrics.csv", index=False)
    (outputs_dir / "generated_text_samples.txt").write_text(
        "\n".join(text_sections), encoding="utf-8"
    )

    markov = build_markov_model(corpus, order=3)
    baseline_text = generate_markov_text(
        markov,
        seed_text=seed,
        generation_length=400,
        random_seed=42,
    )
    rnn_text = generated_rows[1]["generated_text"]
    baseline_comparison = pd.DataFrame(
        [
            {
                "model": "3-character Markov baseline",
                "summary": "Captures short local transitions but has no recurrent hidden state.",
                **generated_text_metrics(baseline_text),
                "sample": baseline_text,
            },
            {
                "model": "Simple RNN",
                "summary": "Learns a recurrent representation of the preceding character sequence.",
                **generated_text_metrics(rnn_text),
                "sample": rnn_text,
            },
        ]
    )
    baseline_comparison.to_csv(outputs_dir / "baseline_comparison.csv", index=False)

    plot_training_history(history_df, outputs_dir / "training_curve.png")
    plot_temperature_comparison(generated_df, outputs_dir / "temperature_comparison.png")
    plot_sequence_summary(metadata, outputs_dir / "sequence_length_summary.png")

    print(f"Model saved to: {model_path}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Validation perplexity: {metadata['validation_perplexity']:.2f}")


if __name__ == "__main__":
    main()
