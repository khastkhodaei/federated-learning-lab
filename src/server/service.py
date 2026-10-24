import logging
from typing import Dict
import grpc

from src.transport.federated_pb2 import (
    RegisterRequest,
    RegisterResponse,
    UpdateRequest,
    UpdateResponse,
)
from src.transport.federated_pb2_grpc import FederatedServiceServicer

logger = logging.getLogger(__name__)


class FederatedServerServicer(FederatedServiceServicer):
    """gRPC servicer handling client node registration and model updates."""

    def __init__(self) -> None:
        self.registered_nodes: Dict[str, str] = {}
        self.updates: Dict[str, UpdateRequest] = {}

    def RegisterNode(
        self, request: RegisterRequest, context: grpc.ServicerContext
    ) -> RegisterResponse:
        self.registered_nodes[request.node_id] = request.address
        logger.info(
            "Registered node: %s at address: '%s'",
            request.node_id,
            request.address or "none",
        )
        return RegisterResponse(
            success=True, message=f"Node {request.node_id} registered."
        )

    def SubmitUpdate(
        self, request: UpdateRequest, context: grpc.ServicerContext
    ) -> UpdateResponse:
        self.updates[request.node_id] = request
        logger.info(
            "Received update from %s (samples: %d, weights: %d bytes)",
            request.node_id,
            request.num_samples,
            len(request.weights),
        )
        return UpdateResponse(
            success=True, message=f"Update received from {request.node_id}."
        )
