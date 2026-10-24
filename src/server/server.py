from concurrent import futures
import logging
import os
import sys
from typing import List, Optional

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import grpc
from src.server.service import FederatedServerServicer
from src.transport.federated_pb2 import TrainRequest, TrainResponse
from src.transport.federated_pb2_grpc import (
    NodeServiceStub,
    add_FederatedServiceServicer_to_server,
)

logger = logging.getLogger(__name__)


class FederatedServer:
    """Manages the gRPC server lifecycle for the central federated coordinator."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        servicer: Optional[FederatedServerServicer] = None,
        max_workers: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.servicer = servicer if servicer is not None else FederatedServerServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        add_FederatedServiceServicer_to_server(self.servicer, self.server)
        self.bound_port = self.server.add_insecure_port(f"{self.host}:{self.port}")

    def start(self) -> None:
        """Starts the gRPC server to accept incoming connections."""
        self.server.start()
        logger.info("FederatedServer started and listening on %s:%d", self.host, self.bound_port)

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
        weights: bytes,
        epochs: int = 1,
        timeout: Optional[float] = None,
    ) -> List[TrainResponse]:
        """Commands all registered nodes to execute a training round in parallel

        and waits until all responses are collected (synchronous round barrier).
        """
        registered = dict(self.servicer.registered_nodes)
        valid_nodes = {nid: addr for nid, addr in registered.items() if addr}

        if not valid_nodes:
            logger.warning(
                "No registered nodes with reachable addresses to trigger round %d.",
                round_id,
            )
            return []

        logger.info(
            "Triggering round %d on %d nodes in parallel (epochs: %d)...",
            round_id,
            len(valid_nodes),
            epochs,
        )

        def _call_node(node_id: str, address: str) -> TrainResponse:
            with grpc.insecure_channel(address) as channel:
                stub = NodeServiceStub(channel)
                req = TrainRequest(round_id=round_id, weights=weights, epochs=epochs)
                return stub.TrainRound(req, timeout=timeout)

        responses: List[TrainResponse] = []
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
                    logger.info("Received TrainRound response from %s (success: %s)", nid, res.success)
                except Exception as err:
                    logger.error("Failed to train round %d on node %s: %s", round_id, nid, err)

        logger.info(
            "Round %d completed. Gathered %d/%d updates.",
            round_id,
            len(responses),
            len(valid_nodes),
        )
        return responses


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [SERVER] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    server = FederatedServer(host="0.0.0.0", port=port)
    server.start()
    print(f"\n>>> Federated Server is running on port {server.bound_port}. Press Ctrl+C to stop.\n")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop(grace=1)
