"""Quantitative, qualitative, and baseline evaluation helpers."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def perplexity_from_loss(loss: float) -> float:
    """Convert next-character cross-entropy loss to perplexity."""
    return float(math.exp(min(float(loss), 20.0)))


def generated_text_metrics(text: str) -> dict[str, float | int]:
    """Simple diagnostics for diversity and repetition."""
    text = str(text)
    trigrams = [text[index : index + 3] for index in range(max(0, len(text) - 2))]
    unique_trigram_ratio = len(set(trigrams)) / max(len(trigrams), 1)
    return {
        "characters": len(text),
        "unique_characters": len(set(text)),
        "unique_character_ratio": len(set(text)) / max(len(text), 1),
        "unique_trigram_ratio": unique_trigram_ratio,
        "repeated_trigram_ratio": 1.0 - unique_trigram_ratio,
        "alphabetic_ratio": sum(character.isalpha() for character in text) / max(len(text), 1),
        "whitespace_ratio": sum(character.isspace() for character in text) / max(len(text), 1),
    }


def build_markov_model(text: str, order: int = 3) -> dict[str, list[str]]:
    """Build an n-gram/Markov baseline from character transitions."""
    if order < 1:
        raise ValueError("order must be at least 1")
    transitions: dict[str, list[str]] = defaultdict(list)
    for index in range(len(text) - order):
        context = text[index : index + order]
        transitions[context].append(text[index + order])
    return dict(transitions)


def generate_markov_text(
    transitions: dict[str, list[str]],
    *,
    seed_text: str,
    generation_length: int = 300,
    random_seed: int = 42,
) -> str:
    """Generate a baseline sample that only captures short local patterns."""
    if not transitions:
        raise ValueError("Markov model is empty")
    order = len(next(iter(transitions)))
    rng = np.random.default_rng(random_seed)
    context = (" " * order + seed_text)[-order:]
    output = list(seed_text)
    keys = list(transitions)

    for _ in range(generation_length):
        choices = transitions.get(context)
        if not choices:
            context = keys[int(rng.integers(0, len(keys)))]
            choices = transitions[context]
        next_character = choices[int(rng.integers(0, len(choices)))]
        output.append(next_character)
        context = (context + next_character)[-order:]
    return "".join(output)
