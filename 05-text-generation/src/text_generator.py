"""Load saved model artifacts and generate text with controlled sampling."""

from __future__ import annotations

import hashlib
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


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest without loading the full artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TextGenerationBundle:
    model: CharacterSimpleRNN
    vocabulary: CharacterVocabulary
    metadata: dict
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = next(self.model.parameters()).device


def _validate_artifact_consistency(
    *,
    checkpoint: dict,
    vocabulary: CharacterVocabulary,
    metadata: dict,
    model_path: Path,
) -> None:
    """Fail early when committed model artifacts do not describe the same model."""
    checkpoint_vocabulary_size = int(checkpoint["vocabulary_size"])
    metadata_vocabulary_size = int(metadata["vocabulary_size"])
    if not (
        checkpoint_vocabulary_size == vocabulary.size == metadata_vocabulary_size
    ):
        raise ValueError(
            "Vocabulary-size mismatch across checkpoint, vocabulary.json, and metadata."
        )

    checkpoint_sequence_length = int(checkpoint["sequence_length"])
    metadata_sequence_length = int(metadata["sequence_length"])
    if checkpoint_sequence_length != metadata_sequence_length:
        raise ValueError("Sequence-length mismatch between checkpoint and metadata.")

    expected_sha256 = metadata.get("model_sha256")
    if expected_sha256 and sha256_file(model_path) != str(expected_sha256):
        raise ValueError(
            "Model checksum does not match model_metadata.json. "
            "Retrain or restore the committed artifact set."
        )


def load_generation_bundle(model_dir: str | Path) -> TextGenerationBundle:
    """Load and validate the committed model, vocabulary, and metadata artifacts."""
    model_dir = Path(model_dir)
    model_path = model_dir / "text_generation_simple_rnn_model.pt"
    vocabulary_path = model_dir / "vocabulary.json"
    metadata_path = model_dir / "model_metadata.json"

    required_paths = (model_path, vocabulary_path, metadata_path)
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    vocabulary = CharacterVocabulary.load(vocabulary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    _validate_artifact_consistency(
        checkpoint=checkpoint,
        vocabulary=vocabulary,
        metadata=metadata,
        model_path=model_path,
    )

    model = CharacterSimpleRNN(
        vocabulary_size=int(checkpoint["vocabulary_size"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        rnn_units=int(checkpoint["rnn_units"]),
        dropout_rate=float(checkpoint["dropout_rate"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = metadata.get("model_parameters")
    if expected_parameter_count is not None and parameter_count != int(expected_parameter_count):
        raise ValueError("Model parameter count does not match model_metadata.json.")

    return TextGenerationBundle(
        model=model,
        vocabulary=vocabulary,
        metadata=metadata,
        device=device,
    )


def sample_next_index(
    logits: np.ndarray,
    *,
    temperature: float = 0.7,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample from one-dimensional logits after temperature and top-k filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    adjusted_logits = np.asarray(logits, dtype=np.float64)
    if adjusted_logits.ndim != 1 or adjusted_logits.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    if not np.isfinite(adjusted_logits).any():
        raise ValueError("logits must contain at least one finite value")

    rng = rng or np.random.default_rng()
    adjusted_logits = adjusted_logits / float(temperature)

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        if top_k < adjusted_logits.size:
            keep = np.argpartition(adjusted_logits, -top_k)[-top_k:]
            filtered = np.full_like(adjusted_logits, -np.inf)
            filtered[keep] = adjusted_logits[keep]
            adjusted_logits = filtered

    finite = adjusted_logits[np.isfinite(adjusted_logits)]
    adjusted_logits = adjusted_logits - np.max(finite)
    probabilities = np.exp(adjusted_logits)
    probabilities[~np.isfinite(probabilities)] = 0.0

    probability_sum = probabilities.sum()
    if not np.isfinite(probability_sum) or probability_sum <= 0:
        return int(np.nanargmax(logits))

    probabilities = probabilities / probability_sum
    return int(rng.choice(len(probabilities), p=probabilities))


def prepare_seed(
    seed_text: str,
    sequence_length: int,
    vocabulary: CharacterVocabulary,
) -> str:
    """Normalize and left-pad a prompt to the model context length."""
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

    normalized_seed = normalize_text(seed_text, preserve_newlines=True)
    sequence_length = int(bundle.metadata["sequence_length"])
    context = prepare_seed(normalized_seed, sequence_length, bundle.vocabulary)
    output = list(normalized_seed if include_seed else "")
    rng = np.random.default_rng(random_seed)

    with torch.inference_mode():
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
