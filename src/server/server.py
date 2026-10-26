import logging
import os
import sys
from concurrent import futures

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import grpc
from torch import nn

from src.baselines.base import ServerStrategy
from src.server.service import FederatedServerServicer
from src.transport.federated_pb2 import TrainRequest, TrainResponse
from src.transport.federated_pb2_grpc import (
    NodeServiceStub,
    add_FederatedServiceServicer_to_server,
)
from src.utils.serialization import load_bytes_into_model, model_to_bytes

logger = logging.getLogger(__name__)


class FederatedServer:
    """Manages the gRPC server lifecycle for the central federated coordinator."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        model: nn.Module | None = None,
        seed: int | None = None,
        servicer: FederatedServerServicer | None = None,
        strategy: ServerStrategy | None = None,
        criterion: nn.Module | None = None,
        max_workers: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.seed = seed
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        if model is not None:
            self.global_model = model
        elif seed is not None:
            from src.models.cnn import SimpleCNN

            self.global_model = SimpleCNN(seed=seed)
        else:
            self.global_model = None

        if strategy is not None:
            self.strategy = strategy
        else:
            from src.baselines.fedavg.strategy import FedAvgServerStrategy

            self.strategy = FedAvgServerStrategy()
        self.client_weights: dict[str, bytes] = {}
        self.servicer = servicer if servicer is not None else FederatedServerServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        add_FederatedServiceServicer_to_server(self.servicer, self.server)
        self.bound_port = self.server.add_insecure_port(f"{self.host}:{self.port}")

    def get_global_weights(self) -> bytes:
        """Returns the current global model parameters serialized as raw bytes."""
        if self.global_model is None:
            return b""
        return model_to_bytes(self.global_model)

    def set_global_weights(self, weights: bytes) -> None:
        """Loads serialized weights into the server's global model."""
        if self.global_model is not None and weights:
            load_bytes_into_model(weights, self.global_model)

    def start(self) -> None:
        """Starts the gRPC server to accept incoming connections."""
        self.server.start()
        logger.info(
            "FederatedServer started and listening on %s:%d",
            self.host,
            self.bound_port,
        )

    def stop(self, grace: int = 0) -> None:
        """Stops the gRPC server."""
        self.server.stop(grace)
        logger.info("FederatedServer stopped.")

    def wait_for_termination(self) -> None:
        """Blocks until the server terminates."""
        self.server.wait_for_termination()

    def trigger_round(
        self,
        round_id: int,
        weights: bytes | dict[str, bytes] | None = None,
        epochs: int = 1,
        timeout: float | None = None,
        aggregate: bool = False,
    ) -> list[TrainResponse]:
        """Commands all registered nodes to execute a training round in parallel

        and waits until all responses are collected (synchronous round barrier).
        If weights are not explicitly provided, uses client-specific weights if available,
        or the server's initial/current global model.
        If aggregate is True, aggregates the updates via the configured strategy and updates the global model.
        """
        if weights is not None:
            payload = weights
        elif self.client_weights:
            payload = self.client_weights
        else:
            payload = self.get_global_weights()

        registered = dict(self.servicer.registered_nodes)
        valid_nodes = {nid: addr for nid, addr in registered.items() if addr}

        if not valid_nodes:
            logger.warning(
                "No registered nodes with reachable addresses to trigger round %d.",
                round_id,
            )
            return []

        payload_desc = (
            f"{len(payload)} client models"
            if isinstance(payload, dict)
            else f"{len(payload)} bytes"
        )
        logger.info(
            "Triggering round %d on %d nodes in parallel (epochs: %d, payload: %s)...",
            round_id,
            len(valid_nodes),
            epochs,
            payload_desc,
        )

        def _call_node(node_id: str, address: str) -> TrainResponse:
            if isinstance(payload, dict):
                node_payload = payload.get(node_id, self.get_global_weights())
            else:
                node_payload = payload
            with grpc.insecure_channel(address) as channel:
                stub = NodeServiceStub(channel)
                req = TrainRequest(
                    round_id=round_id, weights=node_payload, epochs=epochs
                )
                return stub.TrainRound(req, timeout=timeout)

        responses: list[TrainResponse] = []
        with futures.ThreadPoolExecutor(max_workers=len(valid_nodes)) as executor:
            future_to_node = {
                executor.submit(_call_node, nid, addr): nid
                for nid, addr in valid_nodes.items()
            }
            for fut in futures.as_completed(future_to_node):
                nid = future_to_node[fut]
                try:
                    res = fut.result()
                    responses.append(res)
                    logger.info(
                        "Received TrainRound response from %s (success: %s)",
                        nid,
                        res.success,
                    )
                except Exception as err:  # noqa: BLE001
                    logger.error(
                        "Failed to train round %d on node %s: %s",
                        round_id,
                        nid,
                        err,
                    )

        logger.info(
            "Round %d completed. Gathered %d/%d updates.",
            round_id,
            len(responses),
            len(valid_nodes),
        )

        if aggregate:
            self.aggregate_responses(responses, round_id=round_id)

        return responses

    def aggregate_responses(
        self, responses: list[TrainResponse], round_id: int = 1
    ) -> bytes | None:
        """Aggregates successful TrainRound responses using the configured strategy and updates the global model.

        Returns:
            Serialized aggregated parameter bytes, or None if no valid updates.
        """
        valid_updates = [
            (r.weights, r.num_samples)
            for r in responses
            if r.success and r.weights
        ]
        if not valid_updates:
            logger.warning(
                "No successful client updates with weights to aggregate."
            )
            return None

        client_ids = [r.node_id for r in responses if r.success and r.weights]
        aggregated_bytes = self.strategy.aggregate(
            global_model=self.global_model,
            client_updates=valid_updates,
            round_id=round_id,
            client_ids=client_ids,
            responses=responses,
        )

        # Retain client-specific models if strategy generates them
        if hasattr(self.strategy, "client_weights") and isinstance(self.strategy.client_weights, dict):
            self.client_weights = dict(self.strategy.client_weights)

        logger.info(
            "Server global model updated via %s aggregation (%d bytes).",
            self.strategy.__class__.__name__,
            len(aggregated_bytes),
        )
        return aggregated_bytes


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [SERVER] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    server = FederatedServer(host="0.0.0.0", port=port)
    server.start()
    print(
        f"\n>>> Federated Server is running on port {server.bound_port}. Press Ctrl+C to stop.\n"
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop(grace=1)
