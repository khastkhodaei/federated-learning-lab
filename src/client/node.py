from typing import Any, Optional


class Node:
    """Represents a client node in a federated learning network."""

    def __init__(
        self,
        node_id: str | int,
        data: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.node_id = node_id
        self.data = data
        self.model = model
