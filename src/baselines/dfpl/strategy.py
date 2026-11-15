"""DFPL server aggregation and prototype coordination strategy."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import nn

from src.baselines.base import ServerStrategy
from src.baselines.dfpl.prototype import (
    aggregate_prototypes,
    bytes_to_prototypes,
    prototypes_to_bytes,
)
from src.server.aggregation import fedavg_bytes
from src.utils.serialization import load_bytes_into_model


class DFPLServerStrategy(ServerStrategy):
    """Server-side aggregation strategy for Decentralized Federated Prototype Learning (DFPL).

    Coordinates parameter aggregation using sample-weighted FedAvg and aggregates class
    prototypes communicated by participating clients, dispatching consensus prototypes
    back to clients for feature alignment regularization.
    """

    def __init__(
        self,
        normalize_prototypes: bool = False,
        aggregate_weights: bool = True,
    ) -> None:
        """Initializes DFPL server strategy hyperparameters and internal state.

        Args:
            normalize_prototypes: If True, normalizes aggregated prototypes to unit L2 norm.
            aggregate_weights: Whether to aggregate model weights via sample-weighted FedAvg.
        """
        self.normalize_prototypes: bool = bool(normalize_prototypes)
        self.aggregate_weights: bool = bool(aggregate_weights)
        self.global_prototypes: dict[int, torch.Tensor] = {}
        self.global_prototypes_bytes: bytes = b""

    def aggregate(
        self,
        global_model: Optional[nn.Module],
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        client_prototypes: Optional[
            Sequence[Union[dict[int, torch.Tensor], bytes]]
        ] = None,
        client_sample_counts: Optional[Sequence[dict[int, int]]] = None,
        **kwargs: Any,
    ) -> bytes:
        """Aggregates client model weights and class prototype embeddings.

        Args:
            global_model: Reference global model updated in-place with aggregated parameters.
            client_updates: Sequence of tuples (serialized_weights, num_samples).
            round_id: Current federated communication round number.
            client_prototypes: Sequence of client prototype dictionaries or serialized bytes.
            client_sample_counts: Optional sequence of class-wise sample count dictionaries.
            **kwargs: Additional baseline-specific arguments.

        Returns:
            Serialized aggregated model parameter byte payload.

        Raises:
            ValueError: If client_updates is empty.
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty updates.")

        if self.aggregate_weights:
            aggregated_bytes = fedavg_bytes(client_updates)
        else:
            aggregated_bytes = client_updates[0][0]

        if global_model is not None:
            load_bytes_into_model(aggregated_bytes, global_model)

        if client_prototypes is not None:
            deserialized_protos: list[dict[int, torch.Tensor]] = []
            for p in client_prototypes:
                if isinstance(p, bytes):
                    deserialized_protos.append(bytes_to_prototypes(p))
                elif isinstance(p, dict):
                    deserialized_protos.append(p)
                else:
                    raise TypeError(f"Unsupported prototype format: {type(p)}")

            parsed_counts = (
                list(client_sample_counts)
                if client_sample_counts is not None
                else None
            )
            aggregated_protos = aggregate_prototypes(
                deserialized_protos,
                client_sample_counts=parsed_counts,
                normalize=self.normalize_prototypes,
            )
            self.global_prototypes = aggregated_protos
            self.global_prototypes_bytes = prototypes_to_bytes(aggregated_protos)

        return aggregated_bytes

    def get_client_payload(self, client_id: Optional[str] = None) -> bytes:
        """Retrieves serialized consensus prototype payload for client distribution.

        Args:
            client_id: Optional client identifier for client-specific payloads.

        Returns:
            Serialized bytes containing consensus class prototypes.
        """
        return self.global_prototypes_bytes

    def get_global_prototypes(self) -> dict[int, torch.Tensor]:
        """Returns the current consensus class prototypes dictionary."""
        return self.global_prototypes

    def get_global_prototypes_bytes(self) -> bytes:
        """Returns the serialized consensus class prototypes bytes."""
        return self.global_prototypes_bytes


DFPLStrategy = DFPLServerStrategy

__all__ = [
    "DFPLServerStrategy",
    "DFPLStrategy",
]
