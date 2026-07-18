from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SkipGramEmbeddingModel(nn.Module):
    """Predict surrounding context words from a center word."""

    def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dimension = int(embedding_dimension)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=0)
        self.output = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, center_indices: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(center_indices))


@dataclass
class TrainingResult:
    model: SkipGramEmbeddingModel
    history: pd.DataFrame
    best_validation_loss: float
    epochs_completed: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _evaluate_loader(
    model: SkipGramEmbeddingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for centers, contexts in loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            logits = model(centers)
            loss = criterion(logits, contexts)
            losses.append(float(loss.item()) * len(centers))
            top1_correct += int((logits.argmax(dim=1) == contexts).sum().item())
            k = min(5, logits.shape[1])
            top5 = logits.topk(k=k, dim=1).indices
            top5_correct += int((top5 == contexts.unsqueeze(1)).any(dim=1).sum().item())
            total += len(centers)
    return (
        sum(losses) / max(total, 1),
        top1_correct / max(total, 1),
        top5_correct / max(total, 1),
    )


def train_skipgram_model(
    train_centers: np.ndarray,
    train_contexts: np.ndarray,
    validation_centers: np.ndarray,
    validation_contexts: np.ndarray,
    vocabulary_size: int,
    embedding_dimension: int = 32,
    epochs: int = 60,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    patience: int = 10,
    random_seed: int = 42,
    device_name: str = "cpu",
) -> TrainingResult:
    set_global_seed(random_seed)
    device = torch.device(device_name)
    model = SkipGramEmbeddingModel(vocabulary_size, embedding_dimension).to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(train_centers), torch.from_numpy(train_contexts)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_centers), torch.from_numpy(validation_contexts)
    )
    generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        for centers, contexts in train_loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(centers)
            loss = criterion(logits, contexts)
            loss.backward()
            optimizer.step()
            training_loss_sum += float(loss.item()) * len(centers)
            training_count += len(centers)

        training_loss = training_loss_sum / max(training_count, 1)
        validation_loss, validation_top1, validation_top5 = _evaluate_loader(
            model, validation_loader, criterion, device
        )
        rows.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_top1_accuracy": validation_top1,
                "validation_top5_accuracy": validation_top5,
            }
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    model.to("cpu")
    history = pd.DataFrame(rows)
    return TrainingResult(
        model=model,
        history=history,
        best_validation_loss=float(best_validation_loss),
        epochs_completed=int(len(history)),
    )


def extract_embedding_matrix(model: SkipGramEmbeddingModel) -> np.ndarray:
    return model.embedding.weight.detach().cpu().numpy().astype(np.float32)
