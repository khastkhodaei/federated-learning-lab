import logging
from typing import Any
import grpc

from src.transport.federated_pb2 import TrainRequest, TrainResponse
from src.transport.federated_pb2_grpc import NodeServiceServicer

logger = logging.getLogger(__name__)


class ClientNodeServicer(NodeServiceServicer):
    """gRPC servicer for a client node to receive training requests from server or peers."""

    def __init__(self, node: Any) -> None:
        self.node = node

    def TrainRound(
        self, request: TrainRequest, context: grpc.ServicerContext
    ) -> TrainResponse:
        logger.info(
            "Node %s received TrainRound request (round %d, %d epochs)",
            self.node.node_id,
            request.round_id,
            request.epochs,
        )
        if hasattr(self.node, "on_train_round") and callable(self.node.on_train_round):
            return self.node.on_train_round(request)

        return TrainResponse(
            node_id=str(self.node.node_id),
            num_samples=0,
            weights=request.weights,
            success=True,
            message=f"Node {self.node.node_id} completed round {request.round_id}.",
        )
