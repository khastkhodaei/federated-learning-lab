import logging
import os
import sys
from typing import Any, Optional

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.transport.client import FederatedClient
from src.transport.federated_pb2 import RegisterResponse, UpdateResponse

logger = logging.getLogger(__name__)


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
            logger.info("Node %s registering with server...", self.node_id)
            response = self.transport.register_node(self.node_id)
            logger.info("Node %s registration response: %s", self.node_id, response.message)
            return response
        return None

    def send_update(
        self, weights: bytes, num_samples: int
    ) -> Optional[UpdateResponse]:
        """Submits model update to the central server if transport is configured."""
        if self.transport is not None:
            logger.info(
                "Node %s sending update (%d samples, %d bytes) to server...",
                self.node_id,
                num_samples,
                len(weights),
            )
            response = self.transport.submit_update(
                self.node_id, num_samples=num_samples, weights=weights
            )
            logger.info("Node %s update response: %s", self.node_id, response.message)
            return response
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [CLIENT] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    node_id = sys.argv[1] if len(sys.argv) > 1 else "client_1"
    server_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:50051"

    print(f"\n>>> Connecting Node '{node_id}' to server at '{server_address}'...\n")
    transport = FederatedClient(target=server_address)
    node = Node(node_id=node_id, transport=transport)

    # 1. Register with server
    reg_res = node.register()
    print(f"Server response: {reg_res.message if reg_res else 'Failed'}")

    # 2. Send sample update
    dummy_weights = b"sample_model_weights_tensor"
    update_res = node.send_update(weights=dummy_weights, num_samples=32)
    print(f"Server response: {update_res.message if update_res else 'Failed'}\n")

    transport.close()
    print("Done.")
