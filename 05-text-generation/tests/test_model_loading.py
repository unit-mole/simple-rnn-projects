"""Smoke test for the committed Simple RNN inference artifacts."""

from __future__ import annotations

from pathlib import Path

from src.text_generator import generate_text, load_generation_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_loads_and_generates_text() -> None:
    bundle = load_generation_bundle(PROJECT_ROOT / "models")
    generated = generate_text(
        bundle,
        seed_text="The White Rabbit",
        generation_length=20,
        temperature=0.7,
        top_k=20,
        random_seed=7,
    )
    assert generated.startswith("The White Rabbit")
    assert len(generated) == len("The White Rabbit") + 20
