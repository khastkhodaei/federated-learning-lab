from concurrent import futures
import logging
import os
import sys
from typing import Any, Optional

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import grpc
from src.client.service import ClientNodeServicer
from src.transport.client import FederatedClient
from src.transport.federated_pb2 import (
    RegisterResponse,
    TrainRequest,
    TrainResponse,
    UpdateResponse,
)
from src.transport.federated_pb2_grpc import add_NodeServiceServicer_to_server
from src.utils.serialization import load_bytes_into_model, model_to_bytes

logger = logging.getLogger(__name__)


class Node:
    """Represents a client node in a federated learning network.

    Can act as a client talking to a central server, and optionally run its
    own gRPC listener to receive commands/rounds from the coordinator or peers.
    """

    def __init__(
        self,
        node_id: str | int,
        data: Optional[Any] = None,
        model: Optional[Any] = None,
        transport: Optional[FederatedClient] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        max_workers: int = 5,
    ) -> None:
        self.node_id = str(node_id)
        self.data = data
        self.model = model
        self.transport = transport
        self.host = host
        self.port = port

        self.server: Optional[grpc.Server] = None
        self.listening_port: Optional[int] = None
        self.address: str = ""

        if self.port is not None:
            self._start_server(max_workers=max_workers)

    def _start_server(self, max_workers: int = 5) -> None:
        """Starts a local gRPC server to listen for TrainRound requests."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        servicer = ClientNodeServicer(self)
        add_NodeServiceServicer_to_server(servicer, self.server)
        self.listening_port = self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()
        self.address = f"{self.host}:{self.listening_port}"
        logger.info("Node %s listening for RPCs on %s", self.node_id, self.address)

    def stop(self, grace: int = 0) -> None:
        """Stops the local gRPC listener if running."""
        if self.server is not None:
            self.server.stop(grace)
            logger.info("Node %s listener stopped.", self.node_id)

    def register(self) -> Optional[RegisterResponse]:
        """Registers this node with the central server if transport is configured."""
        if self.transport is not None:
            logger.info("Node %s registering with server...", self.node_id)
            if self.address:
                response = self.transport.register_node(self.node_id, address=self.address)
            else:
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

    def on_train_round(self, request: TrainRequest) -> TrainResponse:
        """Handles an incoming gRPC TrainRound request from the server.

        Unpacks the received model parameters into self.model at the start of the round.
        """
        logger.info(
            "Node %s received model parameters for round %d over gRPC (%d bytes)",
            self.node_id,
            request.round_id,
            len(request.weights),
        )

        # Unpack received weights into local model
        if self.model is not None and request.weights:
            load_bytes_into_model(request.weights, self.model)
            logger.info(
                "Node %s successfully synchronized local model with round %d global parameters",
                self.node_id,
                request.round_id,
            )

        weights_payload = (
            model_to_bytes(self.model) if self.model is not None else request.weights
        )

        return TrainResponse(
            node_id=str(self.node_id),
            num_samples=0,
            weights=weights_payload,
            success=True,
            message=f"Node {self.node_id} synchronized parameters for round {request.round_id}.",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [CLIENT] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    node_id = sys.argv[1] if len(sys.argv) > 1 else "client_1"
    server_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:50051"
    node_port = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"\n>>> Starting Node '{node_id}' (port: {node_port or 'no listener'})...\n")
    transport = FederatedClient(target=server_address)
    node = Node(node_id=node_id, transport=transport, port=node_port)

    # 1. Register with server
    reg_res = node.register()
    print(f"Server response: {reg_res.message if reg_res else 'Failed'}")

    # 2. Send sample update
    dummy_weights = b"sample_model_weights_tensor"
    update_res = node.send_update(weights=dummy_weights, num_samples=32)
    print(f"Server response: {update_res.message if update_res else 'Failed'}\n")

    if node.server is not None:
        print(f">>> Node {node_id} is listening on {node.address}. Press Ctrl+C to stop.\n")
        try:
            node.server.wait_for_termination()
        except KeyboardInterrupt:
            print("\nStopping node...")
            node.stop(grace=1)

    transport.close()
    print("Done.")
