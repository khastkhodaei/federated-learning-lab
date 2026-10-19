from typing import Dict, Set
import grpc

from src.transport.federated_pb2 import (
    RegisterRequest,
    RegisterResponse,
    UpdateRequest,
    UpdateResponse,
)
from src.transport.federated_pb2_grpc import FederatedServiceServicer


class FederatedServerServicer(FederatedServiceServicer):
    """gRPC servicer handling client node registration and model updates."""

    def __init__(self) -> None:
        self.registered_nodes: Set[str] = set()
        self.updates: Dict[str, UpdateRequest] = {}

    def RegisterNode(
        self, request: RegisterRequest, context: grpc.ServicerContext
    ) -> RegisterResponse:
        self.registered_nodes.add(request.node_id)
        return RegisterResponse(
            success=True, message=f"Node {request.node_id} registered."
        )

    def SubmitUpdate(
        self, request: UpdateRequest, context: grpc.ServicerContext
    ) -> UpdateResponse:
        self.updates[request.node_id] = request
        return UpdateResponse(
            success=True, message=f"Update received from {request.node_id}."
        )
