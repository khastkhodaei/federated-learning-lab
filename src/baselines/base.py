"""Abstract base classes for Federated Learning algorithm strategies.

Defines the core seams for Server aggregation (ServerStrategy) and Client local
training (ClientTrainer) following the Strategy Pattern.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class ServerStrategy(ABC):
    """Abstract strategy governing server-side aggregation and round coordination."""

    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module | None,
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        **kwargs: Any,
    ) -> bytes:
        """Aggregates client model update payloads into updated global weights.

        Args:
            global_model: The server's current global model (modified in-place if provided).
            client_updates: Sequence of tuples (serialized_weights, num_samples).
            round_id: Current federated round number.

        Returns:
            Serialized aggregated parameter byte payload.
        """


class ClientTrainer(ABC):
    """Abstract strategy governing client-side local model training."""

    @abstractmethod
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
        """Executes local training on client private data.

        Args:
            model: Client local neural network model.
            dataloader: DataLoader delivering local client data batches.
            epochs: Number of local training epochs.
            criterion: Loss function.
            optimizer: Local optimizer.
            device: Computing device (CPU or GPU).
            global_weights: Optional reference global weights from server.

        Returns:
            Tuple of (serialized_updated_weights, total_samples_trained, metrics_dict).
        """
