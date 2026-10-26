"""pFedGraph Client Local Trainer.

Executes local SGD optimization initialized from the client's personalized aggregated model.
Reference: Chen et al., 'pFedGraph: Personalized Federated Learning via Graph'.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.utils.serialization import load_bytes_into_model, model_to_bytes


class PFedGraphClientTrainer(ClientTrainer):
    """Local client trainer for pFedGraph personalized federated learning."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes pFedGraph client trainer."""

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: bytes | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes local gradient descent starting from personalized model weights.

        Args:
            model: Client local model.
            dataloader: Client local training data loader.
            epochs: Number of local training epochs.
            criterion: Loss function.
            optimizer: Local optimizer.
            device: Computing device (CPU or CUDA).
            global_weights: Personalized model weights dispatched from the server.

        Returns:
            Tuple of (updated_serialized_weights, total_samples, metrics).
        """
        # Ensure local model is synchronized with received personalized weights
        if global_weights is not None and len(global_weights) > 0:
            load_bytes_into_model(global_weights, model)

        model.train()
        total_samples = 0
        total_loss = 0.0
        batches = 0

        for _ in range(epochs):
            for batch in dataloader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = targets.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                batches += 1

        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        updated_weights = model_to_bytes(model)
        metrics = {"loss": avg_loss}

        return updated_weights, total_samples, metrics


# Factory alias
PFedGraphTrainer = PFedGraphClientTrainer

__all__ = [
    "PFedGraphClientTrainer",
    "PFedGraphTrainer",
]
