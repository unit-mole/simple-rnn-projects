"""Build and train the portfolio Simple RNN with PyTorch."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Small CPU models run faster and more predictably with limited thread fan-out.
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sequence_generation import SequenceDataset


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 50
    step: int = 3
    validation_fraction: float = 0.10
    embedding_dim: int = 32
    rnn_units: int = 64
    dropout_rate: float = 0.20
    batch_size: int = 256
    epochs: int = 18
    learning_rate: float = 0.002
    patience: int = 3
    random_seed: int = 42


class CharacterSimpleRNN(nn.Module):
    """Embedding -> Simple RNN -> Dropout -> vocabulary logits."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 32,
        rnn_units: int = 64,
        dropout_rate: float = 0.20,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(rnn_units, vocabulary_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        recurrent_output, _ = self.rnn(embedded)
        final_timestep = recurrent_output[:, -1, :]
        return self.output(self.dropout(final_timestep))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(targets.size(0))

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    dataset: SequenceDataset,
    *,
    config: TrainingConfig,
    model_dir: str | Path,
) -> tuple[CharacterSimpleRNN, pd.DataFrame, Path]:
    """Train with early stopping and export the best state dictionary."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterSimpleRNN(
        vocabulary_size=dataset.vocabulary.size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units,
        dropout_rate=config.dropout_rate,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.X_train).long(),
            torch.from_numpy(dataset.y_train).long(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.X_validation).long(),
            torch.from_numpy(dataset.y_validation).long(),
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model_path = model_dir / "text_generation_simple_rnn_model.pt"

    history_rows: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, optimizer=None, device=device
        )
        history_rows.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "val_loss": validation_loss,
                "val_accuracy": validation_accuracy,
            }
        )
        print(
            f"Epoch {epoch}/{config.epochs} - loss: {train_loss:.4f} - "
            f"accuracy: {train_accuracy:.4f} - val_loss: {validation_loss:.4f} - "
            f"val_accuracy: {validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary_size": dataset.vocabulary.size,
                    "embedding_dim": config.embedding_dim,
                    "rnn_units": config.rnn_units,
                    "dropout_rate": config.dropout_rate,
                    "sequence_length": config.sequence_length,
                },
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset.vocabulary.save(model_dir / "vocabulary.json")
    (model_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return model, pd.DataFrame(history_rows), model_path


def evaluate_model(
    model: CharacterSimpleRNN,
    dataset: SequenceDataset,
    *,
    batch_size: int = 256,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.X_validation).long(),
            torch.from_numpy(dataset.y_validation).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return _run_epoch(model, loader, nn.CrossEntropyLoss(), optimizer=None, device=device)
