"""DFedPGP server strategy for directed Push-Sum partial model aggregation."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from src.baselines.base import ServerStrategy
from src.baselines.dfedpgp.partition import DEFAULT_HEAD_PREFIXES, partition_model
from src.baselines.dfedpgp.topology import build_directed_mixing_matrix
from src.server.aggregation import fedavg_vectors
from src.utils.serialization import (
    bytes_to_vector,
    load_bytes_into_model,
    vector_to_bytes,
)


def _push_sum_step(
    states: torch.Tensor,
    mixing_matrix: torch.Tensor,
    mu_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs one column-stochastic Push-Sum mixing and de-biasing step."""
    u_biased = mu_prev.unsqueeze(1) * states
    u_next = mixing_matrix @ u_biased
    mu_next = mixing_matrix @ mu_prev
    z_next = u_next / mu_next.unsqueeze(1).clamp(min=1e-12)
    return z_next, mu_next


class DFedPGPServerStrategy(ServerStrategy):
    """Server-side coordination strategy for DFedPGP.

    Maintains a directed Push-Sum state vector ``push_sum_weights`` and generates
    client-specific payloads combining de-biased consensus ``shared_backbone``
    parameters with each client's private ``personal_head``.
    """

    def __init__(
        self,
        out_degree: int = 1,
        seed: int = 42,
        time_varying: bool = True,
        head_prefixes: tuple[str, ...] = DEFAULT_HEAD_PREFIXES,
    ) -> None:
        """Initializes DFedPGP strategy parameters.

        Args:
            out_degree: Number of non-self outgoing peers per client (must be ``>= 0``).
            seed: Deterministic base seed for directed topology generation.
            time_varying: Whether topology changes each round using ``round_id``.
            head_prefixes: Parameter/module name prefixes kept local as ``personal_head``.

        Raises:
            ValueError: If ``out_degree < 0``.
        """
        if out_degree < 0:
            raise ValueError(f"out_degree must be >= 0, got {out_degree}.")

        self.out_degree: int = int(out_degree)
        self.seed: int = int(seed)
        self.time_varying: bool = bool(time_varying)
        self.head_prefixes: tuple[str, ...] = tuple(head_prefixes)
        self.client_ids: list[str] = []
        self.client_weights: dict[str, bytes] = {}
        self.push_sum_weights: torch.Tensor | None = None
        self.mixing_matrix: torch.Tensor | None = None

    def aggregate(
        self,
        global_model: nn.Module | None,
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        client_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Aggregates shared backbones via directed Push-Sum while preserving personal heads.

        Args:
            global_model: Optional reference model defining the parameter partition; when
                provided, updated in-place with the sample-weighted FedAvg parameters.
            client_updates: Sequence of tuples ``(serialized_weights, num_samples)``.
            round_id: Current federated communication round index.
            client_ids: Optional client identifiers aligned with ``client_updates``.

        Returns:
            Serialized sample-weighted FedAvg global model parameters.

        Raises:
            ValueError: If ``client_updates`` is empty, ``client_ids`` length mismatches
                ``client_updates``, ``round_id < 0``, or ``global_model`` partitioning/splitting fails.
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty updates.")

        k = len(client_updates)
        if client_ids is not None and len(client_ids) != k:
            raise ValueError(
                f"client_ids length ({len(client_ids)}) must match client_updates length ({k})."
            )

        raw_ids = (
            [f"client_{i}" for i in range(k)]
            if client_ids is None
            else [str(cid) for cid in client_ids]
        )
        paired = sorted(
            zip(raw_ids, client_updates, strict=True),
            key=lambda item: item[0],
        )
        sorted_client_ids = [cid for cid, _ in paired]
        ordered_updates = [update for _, update in paired]

        if self.push_sum_weights is None or sorted_client_ids != self.client_ids:
            mu_prev = torch.ones(k, dtype=torch.float32)
        else:
            mu_prev = self.push_sum_weights

        p_matrix = build_directed_mixing_matrix(
            num_clients=k,
            round_id=round_id,
            out_degree=self.out_degree,
            seed=self.seed,
            time_varying=self.time_varying,
        )

        vectors = [bytes_to_vector(raw_bytes) for raw_bytes, _ in ordered_updates]
        vector_updates = [
            (vec, num_samples)
            for vec, (_, num_samples) in zip(vectors, ordered_updates, strict=True)
        ]

        if global_model is not None:
            partition = partition_model(global_model, head_prefixes=self.head_prefixes)
            split_pairs = [partition.split_vector(vec) for vec in vectors]
            shared_states = torch.stack([backbone for backbone, _ in split_pairs], dim=0)
            personal_heads: list[torch.Tensor] | None = [head for _, head in split_pairs]
        else:
            partition = None
            shared_states = torch.stack(vectors, dim=0)
            personal_heads = None

        z_next, mu_next = _push_sum_step(shared_states, p_matrix, mu_prev)
        global_vec = fedavg_vectors(vector_updates)
        global_bytes = vector_to_bytes(global_vec)
        client_weights = {
            cid: vector_to_bytes(
                partition.merge_vectors(z_next[i], personal_heads[i])
                if partition is not None and personal_heads is not None
                else z_next[i]
            )
            for i, cid in enumerate(sorted_client_ids)
        }

        if global_model is not None:
            load_bytes_into_model(global_bytes, global_model)
        self.client_ids = sorted_client_ids
        self.mixing_matrix = p_matrix
        self.push_sum_weights = mu_next
        self.client_weights = client_weights
        return global_bytes
