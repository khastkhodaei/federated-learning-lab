"""Ditto Client Local Trainer.

Implements the client-side dual-model optimization algorithm for Ditto
(Li et al., 2021). Each client maintains:
1. A global model updated locally via standard SGD and communicated to the server.
2. A private personalized model optimized locally with a proximal penalty pulling it
   toward the updated global model: min_v F_k(v) + (lambda / 2) * ||v - w*||^2.
"""

from collections.abc import Iterable
import copy
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.utils.serialization import model_to_bytes


class DittoClientTrainer(ClientTrainer):
    """Local client trainer for Ditto Personalized Federated Learning."""

    def __init__(
        self,
        lambda_penalty: float = 0.1,
        personalized_epochs: int = 1,
    ) -> None:
        self.lambda_penalty = float(lambda_penalty)
        self.personalized_epochs = int(personalized_epochs)
        self.personalized_model: nn.Module | None = None
        self.personalized_optimizer: Optimizer | None = None

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: bytes | None = None,
        lambda_penalty: float | None = None,
        personalized_epochs: int | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes dual-model training: optimizes the global model first, then the personalized model."""
        effective_lambda = float(
            lambda_penalty if lambda_penalty is not None else self.lambda_penalty
        )
        effective_pers_epochs = int(
            personalized_epochs if personalized_epochs is not None else self.personalized_epochs
        )

        # 1. Initialize personalized model if first round
        if self.personalized_model is None:
            self.personalized_model = copy.deepcopy(model).to(device)
            lr = optimizer.param_groups[0].get("lr", 0.01)
            momentum = optimizer.param_groups[0].get("momentum", 0.9)
            weight_decay = optimizer.param_groups[0].get("weight_decay", 0.0001)
            self.personalized_optimizer = torch.optim.SGD(
                self.personalized_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            self.personalized_model.to(device)

        # 2. Phase 1: Train global model w on private data (Standard FedAvg SGD)
        model.train()
        total_samples = 0
        total_global_loss = 0.0

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
                total_global_loss += loss.item() * batch_size

        avg_global_loss = (total_global_loss / total_samples) if total_samples > 0 else 0.0

        # 3. Snapshot newly updated global model parameters w* as frozen reference targets
        frozen_updated_global: list[torch.Tensor] = [
            param.detach().clone().to(device) for param in model.parameters()
        ]
        for p in frozen_updated_global:
            p.requires_grad = False

        # 4. Phase 2: Train personalized model v with proximal regularizer towards w*
        self.personalized_model.train()
        total_pers_loss = 0.0
        total_prox_penalty = 0.0
        pers_samples = 0

        for _ in range(effective_pers_epochs):
            for batch in dataloader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                if self.personalized_optimizer is not None:
                    self.personalized_optimizer.zero_grad()

                outputs = self.personalized_model(inputs)
                base_loss = criterion(outputs, targets)

                # Proximal penalty: (lambda / 2) * ||v - w*||^2
                proximal_term = torch.tensor(0.0, device=device)
                if effective_lambda > 0.0:
                    for v_param, w_param in zip(
                        self.personalized_model.parameters(), frozen_updated_global
                    ):
                        proximal_term += ((v_param - w_param.to(device)) ** 2).sum()
                    total_loss_batch = base_loss + (effective_lambda / 2.0) * proximal_term
                else:
                    total_loss_batch = base_loss

                total_loss_batch.backward()
                if self.personalized_optimizer is not None:
                    self.personalized_optimizer.step()

                batch_size = targets.size(0)
                pers_samples += batch_size
                total_pers_loss += total_loss_batch.item() * batch_size
                total_prox_penalty += proximal_term.item() * batch_size

        avg_pers_loss = (total_pers_loss / pers_samples) if pers_samples > 0 else 0.0
        avg_prox_loss = (total_prox_penalty / pers_samples) if pers_samples > 0 else 0.0

        # 5. Only the global model weights are transmitted back to the server
        updated_global_weights = model_to_bytes(model)

        metrics = {
            "loss": avg_global_loss,
            "personalized_loss": avg_pers_loss,
            "proximal_penalty": avg_prox_loss,
            "lambda": effective_lambda,
        }

        return updated_global_weights, total_samples, metrics


# Factory alias
DittoTrainer = DittoClientTrainer

__all__ = [
    "DittoClientTrainer",
    "DittoTrainer",
]
