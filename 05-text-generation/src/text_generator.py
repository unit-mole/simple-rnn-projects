"""Load model artifacts and generate text with temperature and top-k sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Small CPU models run faster and more predictably with limited thread fan-out.
torch.set_num_threads(1)

from .data_preprocessing import normalize_text
from .model_training import CharacterSimpleRNN
from .text_preprocessing import CharacterVocabulary


@dataclass
class TextGenerationBundle:
    model: CharacterSimpleRNN
    vocabulary: CharacterVocabulary
    metadata: dict
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = next(self.model.parameters()).device


def load_generation_bundle(model_dir: str | Path) -> TextGenerationBundle:
    model_dir = Path(model_dir)
    model_path = model_dir / "text_generation_simple_rnn_model.pt"
    vocabulary_path = model_dir / "vocabulary.json"
    metadata_path = model_dir / "model_metadata.json"

    missing = [path.name for path in (model_path, vocabulary_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = CharacterSimpleRNN(
        vocabulary_size=int(checkpoint["vocabulary_size"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        rnn_units=int(checkpoint["rnn_units"]),
        dropout_rate=float(checkpoint["dropout_rate"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    vocabulary = CharacterVocabulary.load(vocabulary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return TextGenerationBundle(model=model, vocabulary=vocabulary, metadata=metadata, device=device)


def sample_next_index(
    logits: np.ndarray,
    *,
    temperature: float = 0.7,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample from model logits after temperature and top-k filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    rng = rng or np.random.default_rng()

    adjusted_logits = np.asarray(logits, dtype=np.float64) / temperature
    if top_k is not None and 0 < top_k < adjusted_logits.size:
        keep = np.argpartition(adjusted_logits, -top_k)[-top_k:]
        filtered = np.full_like(adjusted_logits, -np.inf)
        filtered[keep] = adjusted_logits[keep]
        adjusted_logits = filtered

    finite = adjusted_logits[np.isfinite(adjusted_logits)]
    adjusted_logits = adjusted_logits - np.max(finite)
    probabilities = np.exp(adjusted_logits)
    probabilities[~np.isfinite(probabilities)] = 0.0
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def prepare_seed(seed_text: str, sequence_length: int, vocabulary: CharacterVocabulary) -> str:
    seed_text = normalize_text(seed_text, preserve_newlines=True)
    if not seed_text:
        raise ValueError("Seed text cannot be empty")
    padding_character = " " if " " in vocabulary.char_to_index else vocabulary.characters[0]
    return (padding_character * sequence_length + seed_text)[-sequence_length:]


def generate_text(
    bundle: TextGenerationBundle,
    *,
    seed_text: str,
    generation_length: int = 300,
    temperature: float = 0.7,
    top_k: int | None = 20,
    random_seed: int | None = 42,
    include_seed: bool = True,
) -> str:
    """Autoregressively predict one character at a time."""
    if not 1 <= generation_length <= 5000:
        raise ValueError("generation_length must be between 1 and 5000")

    sequence_length = int(bundle.metadata["sequence_length"])
    context = prepare_seed(seed_text, sequence_length, bundle.vocabulary)
    output = list(seed_text if include_seed else "")
    rng = np.random.default_rng(random_seed)

    with torch.no_grad():
        for _ in range(generation_length):
            encoded = bundle.vocabulary.encode(context).reshape(1, sequence_length)
            inputs = torch.from_numpy(encoded).long().to(bundle.device)
            logits = bundle.model(inputs)[0].detach().cpu().numpy()
            next_index = sample_next_index(
                logits,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
            next_character = bundle.vocabulary.index_to_char[next_index]
            if next_character == "�":
                next_character = " "
            output.append(next_character)
            context = context[1:] + next_character

    return "".join(output)
