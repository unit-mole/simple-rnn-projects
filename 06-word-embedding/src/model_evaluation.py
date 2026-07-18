from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .embedding_training import SkipGramEmbeddingModel


def evaluate_context_prediction(
    model: SkipGramEmbeddingModel,
    centers: np.ndarray,
    contexts: np.ndarray,
    batch_size: int = 1024,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = len(centers)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_centers = torch.from_numpy(centers[start : start + batch_size])
            batch_contexts = torch.from_numpy(contexts[start : start + batch_size])
            logits = model(batch_centers)
            total_loss += float(criterion(logits, batch_contexts).item())
            top1_correct += int((logits.argmax(dim=1) == batch_contexts).sum().item())
            k = min(5, logits.shape[1])
            top5 = logits.topk(k=k, dim=1).indices
            top5_correct += int(
                (top5 == batch_contexts.unsqueeze(1)).any(dim=1).sum().item()
            )
    mean_loss = total_loss / max(total, 1)
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": float(math.exp(min(mean_loss, 20))),
        "validation_top1_accuracy": top1_correct / max(total, 1),
        "validation_top5_accuracy": top5_correct / max(total, 1),
    }
