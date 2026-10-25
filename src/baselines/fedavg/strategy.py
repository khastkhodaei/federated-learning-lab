"""Federated Averaging (FedAvg) Server Strategy.

Implements the standard FedAvg parameter aggregation algorithm (McMahan et al., 2017)
weighting client model vectors proportionally to their local sample volumes.
"""

from collections.abc import Sequence
from typing import Any

from torch import nn

from src.baselines.base import ServerStrategy
from src.server.aggregation import fedavg_bytes
from src.utils.serialization import load_bytes_into_model


class FedAvgServerStrategy(ServerStrategy):
    """Canonical Federated Averaging server aggregation strategy."""

    def aggregate(
        self,
        global_model: nn.Module | None,
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        **kwargs: Any,
    ) -> bytes:
        """Computes the sample-weighted average of client updates and updates global_model."""
        if not client_updates:
            raise ValueError("Cannot aggregate empty updates.")

        aggregated_bytes = fedavg_bytes(client_updates)

        if global_model is not None:
            load_bytes_into_model(aggregated_bytes, global_model)

        return aggregated_bytes
