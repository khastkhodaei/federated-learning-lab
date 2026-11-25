"""Local-Only server strategy performing zero cross-client parameter aggregation."""

from collections.abc import Sequence
from typing import Any

from torch import nn

from src.baselines.base import ServerStrategy
from src.utils.serialization import model_to_bytes


class LocalOnlyServerStrategy(ServerStrategy):
    """Reference strategy where clients never share or aggregate weights.

    Preserves ``global_model`` unchanged and maps each client back to its own
    submitted parameter bytes without cross-client averaging.
    """

    def __init__(self) -> None:
        self.client_weights: dict[str, bytes] = {}

    def aggregate(
        self,
        global_model: nn.Module | None,
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        client_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Retains per-client weights independently without mutating global_model."""
        k = len(client_updates)
        effective_ids = (
            [str(cid) for cid in client_ids]
            if client_ids is not None and len(client_ids) == k
            else [f"client_{i}" for i in range(k)]
        )
        self.client_weights = {
            cid: weights_bytes
            for cid, (weights_bytes, _) in zip(effective_ids, client_updates, strict=True)
        }
        return model_to_bytes(global_model) if global_model is not None else b""
