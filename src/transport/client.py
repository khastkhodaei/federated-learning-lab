from typing import Optional
import grpc

from src.transport.federated_pb2 import (
    RegisterRequest,
    RegisterResponse,
    UpdateRequest,
    UpdateResponse,
)
from src.transport.federated_pb2_grpc import FederatedServiceStub


class FederatedClient:
    """Client for communicating with the central federated learning server via gRPC."""

    def __init__(
        self, target: str = "localhost:50051", channel: Optional[grpc.Channel] = None
    ) -> None:
        self.target = target
        self.channel = channel if channel is not None else grpc.insecure_channel(self.target)
        self.stub = FederatedServiceStub(self.channel)

    def register_node(self, node_id: str, address: str = "") -> RegisterResponse:
        request = RegisterRequest(node_id=node_id, address=address)
        return self.stub.RegisterNode(request)

    def submit_update(
        self, node_id: str, num_samples: int, weights: bytes
    ) -> UpdateResponse:
        request = UpdateRequest(
            node_id=node_id, num_samples=num_samples, weights=weights
        )
        return self.stub.SubmitUpdate(request)

    def close(self) -> None:
        self.channel.close()
