from typing import Any, Optional

from src.transport.client import FederatedClient
from src.transport.federated_pb2 import RegisterResponse, UpdateResponse


class Node:
    """Represents a client node in a federated learning network."""

    def __init__(
        self,
        node_id: str | int,
        data: Optional[Any] = None,
        model: Optional[Any] = None,
        transport: Optional[FederatedClient] = None,
    ) -> None:
        self.node_id = str(node_id)
        self.data = data
        self.model = model
        self.transport = transport

    def register(self) -> Optional[RegisterResponse]:
        """Registers this node with the central server if transport is configured."""
        if self.transport is not None:
            return self.transport.register_node(self.node_id)
        return None

    def send_update(
        self, weights: bytes, num_samples: int
    ) -> Optional[UpdateResponse]:
        """Submits model update to the central server if transport is configured."""
        if self.transport is not None:
            return self.transport.submit_update(
                self.node_id, num_samples=num_samples, weights=weights
            )
        return None
