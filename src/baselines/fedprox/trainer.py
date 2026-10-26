"""Federated Proximal (FedProx) Client Local Trainer.

Implements local SGD training with a proximal regularization term (Li et al., 2020)
penalizing parameter divergence from the global model during non-IID local updates.
"""

from collections.abc import Iterable
import copy
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.utils.serialization import load_bytes_into_model, model_to_bytes


class FedProxClientTrainer(ClientTrainer):
    """Local client trainer incorporating FedProx proximal regularization."""

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = float(mu)

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: bytes | None = None,
        mu: float | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes local gradient descent with proximal penalty against frozen global weights."""
        model.train()
        effective_mu = float(mu if mu is not None else self.mu)

        # 1. Capture detached, frozen copy of global parameters on target device
        frozen_global_params: list[torch.Tensor] = []
        if global_weights is not None and len(global_weights) > 0:
            ref_model = copy.deepcopy(model)
            load_bytes_into_model(global_weights, ref_model)
            frozen_global_params = [
                param.detach().clone().to(device) for param in ref_model.parameters()
            ]
        else:
            frozen_global_params = [
                param.detach().clone().to(device) for param in model.parameters()
            ]

        for p in frozen_global_params:
            p.requires_grad = False

        total_samples = 0
        total_loss = 0.0
        total_base_loss = 0.0
        total_prox_loss = 0.0
        batches = 0

        # 2. Local SGD training loop
        for _ in range(epochs):
            for batch in dataloader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                base_loss = criterion(outputs, targets)

                # 3. Proximal regularization term: (mu / 2) * ||w - w^t||^2
                proximal_term = torch.tensor(0.0, device=device)
                if effective_mu > 0.0:
                    for local_param, global_param in zip(model.parameters(), frozen_global_params):
                        proximal_term += ((local_param - global_param.to(device)) ** 2).sum()
                    total_loss_batch = base_loss + (effective_mu / 2.0) * proximal_term
                else:
                    total_loss_batch = base_loss

                total_loss_batch.backward()
                optimizer.step()

                batch_size = targets.size(0)
                total_samples += batch_size
                total_loss += total_loss_batch.item() * batch_size
                total_base_loss += base_loss.item() * batch_size
                total_prox_loss += proximal_term.item() * batch_size
                batches += 1

        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        avg_base_loss = (total_base_loss / total_samples) if total_samples > 0 else 0.0
        avg_prox_loss = (total_prox_loss / total_samples) if total_samples > 0 else 0.0

        updated_weights = model_to_bytes(model)
        metrics = {
            "loss": avg_loss,
            "base_loss": avg_base_loss,
            "proximal_loss": avg_prox_loss,
            "mu": effective_mu,
        }

        return updated_weights, total_samples, metrics


# Factory / Architect Alias
FedProxTrainer = FedProxClientTrainer

__all__ = [
    "FedProxClientTrainer",
    "FedProxTrainer",
]
