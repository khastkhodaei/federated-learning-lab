"""Local-Only client trainer performing purely private local SGD."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.utils.serialization import model_to_bytes


class LocalOnlyClientTrainer(ClientTrainer):
    """Reference trainer optimizing strictly on private local data without global sync."""

    ignore_global_weights: bool = True

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
        """Executes standard local SGD epochs while ignoring any incoming global weights."""
        model.train()
        total_samples = 0
        total_loss = 0.0

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

        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        updated_weights = model_to_bytes(model)
        return updated_weights, total_samples, {"loss": avg_loss}


LocalOnlyTrainer = LocalOnlyClientTrainer
