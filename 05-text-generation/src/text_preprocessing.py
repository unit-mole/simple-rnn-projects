"""Character vocabulary creation and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

UNK_TOKEN = "�"


@dataclass
class CharacterVocabulary:
    """Bidirectional character-index mapping used by training and inference."""

    characters: list[str]

    def __post_init__(self) -> None:
        if not self.characters:
            raise ValueError("Vocabulary cannot be empty")
        if len(self.characters) != len(set(self.characters)):
            raise ValueError("Vocabulary characters must be unique")
        if UNK_TOKEN not in self.characters:
            self.characters.append(UNK_TOKEN)
        self.char_to_index = {character: index for index, character in enumerate(self.characters)}
        self.index_to_char = {index: character for character, index in self.char_to_index.items()}

    @classmethod
    def fit(cls, text: str) -> "CharacterVocabulary":
        if not text:
            raise ValueError("Cannot build vocabulary from empty text")
        characters = sorted(set(text))
        if UNK_TOKEN not in characters:
            characters.append(UNK_TOKEN)
        return cls(characters)

    @property
    def size(self) -> int:
        return len(self.characters)

    @property
    def unknown_index(self) -> int:
        return self.char_to_index[UNK_TOKEN]

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(
            [self.char_to_index.get(character, self.unknown_index) for character in text],
            dtype=np.int32,
        )

    def decode(self, indices: Iterable[int], *, hide_unknown: bool = True) -> str:
        decoded = "".join(self.index_to_char.get(int(index), UNK_TOKEN) for index in indices)
        return decoded.replace(UNK_TOKEN, "") if hide_unknown else decoded

    def to_dict(self) -> dict:
        return {
            "characters": self.characters,
            "vocabulary_size": self.size,
            "unknown_token": UNK_TOKEN,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["characters"]))
