import logging
import os
import sys
from collections.abc import Callable, Iterable
from concurrent import futures
from typing import Any

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import grpc
import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
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

    Manages local model parameters, private dataset partitions,
    and handles incoming gRPC TrainRound RPCs from the server.
    """

    def __init__(
        self,
        node_id: str | int,
        data: Any | None = None,
        test_data: Any | None = None,
        model: nn.Module | None = None,
        transport: FederatedClient | None = None,
        criterion: nn.Module | None = None,
        optimizer_fn: Callable[[Iterable[nn.Parameter]], Optimizer] | None = None,
        lr: float = 0.01,
        device: str | torch.device = "cpu",
        host: str = "127.0.0.1",
        port: int | None = None,
        trainer: ClientTrainer | None = None,
        max_workers: int = 5,
    ) -> None:
        self.node_id = str(node_id)
        self.data = data
        self.test_data = test_data
        self.device = torch.device(device)
        self.model = model.to(self.device) if model is not None else None
        self.transport = transport
        self.lr = lr
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        if optimizer_fn is not None:
            self._optimizer_fn = optimizer_fn
        else:
            self._optimizer_fn = lambda params: torch.optim.SGD(
                params, lr=self.lr, momentum=0.9
            )

        if trainer is not None:
            self.trainer = trainer
        else:
            from src.baselines.fedavg.trainer import FedAvgClientTrainer
            self.trainer = FedAvgClientTrainer()
        self.host = host
        self.port = port

        self.server: grpc.Server | None = None
        self.listening_port: int | None = None
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

    def register(self) -> RegisterResponse | None:
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
    ) -> UpdateResponse | None:
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

    def train_local(
        self, epochs: int = 1, global_weights: bytes | None = None
    ) -> tuple[bytes, int]:
        """Executes local training on self.data using the configured ClientTrainer.

        Returns:
            Tuple of (serialized_updated_weights, num_samples_trained).
        """
        if self.model is None or self.data is None:
            weights = model_to_bytes(self.model) if self.model is not None else b""
            return weights, 0

        optimizer = self._optimizer_fn(self.model.parameters())
        updated_weights, total_samples, _metrics = self.trainer.train(
            model=self.model,
            dataloader=self.data,
            epochs=epochs,
            criterion=self.criterion,
            optimizer=optimizer,
            device=self.device,
            global_weights=global_weights,
        )
        return updated_weights, total_samples

    def evaluate_local(
        self,
        weights: bytes | None = None,
        dataloader: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluates the local model on private test partition data.

        Args:
            weights: Optional serialized global or round weights to load before evaluation.
            dataloader: Optional explicit DataLoader override (defaults to self.test_data or self.data).

        Returns:
            Dictionary with 'loss', 'accuracy', 'correct', and 'total'.
        """
        if self.model is not None and weights is not None:
            load_bytes_into_model(weights, self.model)

        eval_loader = (
            dataloader
            if dataloader is not None
            else (self.test_data if self.test_data is not None else self.data)
        )
        if self.model is None or eval_loader is None:
            return {"loss": 0.0, "accuracy": 0.0, "correct": 0, "total": 0}

        from src.utils.evaluation import evaluate_model

        return evaluate_model(
            model=self.model,
            dataloader=eval_loader,
            criterion=self.criterion,
            device=self.device,
        )

    def on_train_round(self, request: TrainRequest) -> TrainResponse:
        """Handles an incoming gRPC TrainRound request from the server.

        1. Unpacks received global weights into self.model at the start of the round.
        2. Executes local training on self.data for request.epochs.
        3. Returns the updated weights and sample count directly in TrainResponse.
        """
        logger.info(
            "Node %s starting round %d (epochs: %d, received weights: %d bytes)",
            self.node_id,
            request.round_id,
            request.epochs,
            len(request.weights),
        )

        # 1. Unpack received global weights into local model
        if self.model is not None and request.weights:
            load_bytes_into_model(request.weights, self.model)
            logger.info(
                "Node %s synchronized local model with round %d global parameters",
                self.node_id,
                request.round_id,
            )

        # 2. Run local gradient descent training if data and model are present
        epochs = max(1, request.epochs)
        metrics: dict[str, float] = {}
        if self.model is not None and self.data is not None:
            optimizer = self._optimizer_fn(self.model.parameters())
            updated_weights, num_samples, metrics = self.trainer.train(
                model=self.model,
                dataloader=self.data,
                epochs=epochs,
                criterion=self.criterion,
                optimizer=optimizer,
                device=self.device,
                global_weights=request.weights,
            )
        elif self.model is not None:
            updated_weights, num_samples = model_to_bytes(self.model), 0
        else:
            updated_weights, num_samples = request.weights, 0

        logger.info(
            "Node %s completed round %d handling (trained samples: %d)",
            self.node_id,
            request.round_id,
            num_samples,
        )

        loss_val = metrics.get("loss", 0.0)
        msg = f"loss={loss_val:.6f};epochs={epochs};samples={num_samples}"

        # 3. Return updated weights directly over gRPC
        return TrainResponse(
            node_id=str(self.node_id),
            num_samples=num_samples,
            weights=updated_weights,
            success=True,
            message=msg,
        )


if __name__ == "__main__":
    from src.client.config import create_client_node

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [CLIENT] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path = sys.argv[1]
        print(f"\n>>> Starting Node from config: '{config_path}'...\n")
        node = create_client_node(config_path, load_dataset=True)
    else:
        node_id = sys.argv[1] if len(sys.argv) > 1 else "client_1"
        server_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:50051"
        node_port = int(sys.argv[3]) if len(sys.argv) > 3 else 0

        print(f"\n>>> Starting Node '{node_id}' (port: {node_port})...\n")
        transport = FederatedClient(target=server_address)
        node = Node(node_id=node_id, transport=transport, port=node_port)

    # Register with server
    reg_res = node.register()
    print(f"Server registration response: {reg_res.message if reg_res else 'Failed'}")

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
