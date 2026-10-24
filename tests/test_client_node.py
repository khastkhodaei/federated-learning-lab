from unittest.mock import MagicMock
import grpc
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client.node import Node
from src.client.service import ClientNodeServicer
from src.models.cnn import SimpleCNN
from src.transport.federated_pb2 import (
    RegisterResponse,
    TrainRequest,
    UpdateResponse,
)
from src.transport.federated_pb2_grpc import NodeServiceStub
from src.utils.serialization import bytes_to_vector, model_to_bytes


def test_node_creation() -> None:
    node = Node(node_id="client_1", data=None, model=None)
    assert node.node_id == "client_1"
    assert node.data is None
    assert node.model is None
    assert node.transport is None
    assert node.register() is None
    assert node.send_update(weights=b"weights", num_samples=10) is None


def test_node_register_with_transport() -> None:
    mock_transport = MagicMock()
    mock_transport.register_node.return_value = RegisterResponse(
        success=True, message="registered"
    )

    node = Node(node_id="client_1", transport=mock_transport)
    response = node.register()

    assert response is not None
    assert response.success is True
    mock_transport.register_node.assert_called_once_with("client_1")


def test_node_send_update_with_transport() -> None:
    mock_transport = MagicMock()
    mock_transport.submit_update.return_value = UpdateResponse(
        success=True, message="received"
    )

    node = Node(node_id="client_1", transport=mock_transport)
    response = node.send_update(weights=b"serialized_params", num_samples=32)

    assert response is not None
    assert response.success is True
    mock_transport.submit_update.assert_called_once_with(
        "client_1", num_samples=32, weights=b"serialized_params"
    )


def test_client_node_servicer_train_round() -> None:
    node = Node(node_id="test_worker")
    servicer = ClientNodeServicer(node)
    context = MagicMock()
    request = TrainRequest(round_id=1, weights=b"initial_weights", epochs=3)

    response = servicer.TrainRound(request, context)

    assert response.success is True
    assert response.node_id == "test_worker"
    assert response.weights == b"initial_weights"


def test_node_listener_lifecycle_and_rpc() -> None:
    mock_transport = MagicMock()
    mock_transport.register_node.return_value = RegisterResponse(
        success=True, message="ok"
    )

    # Start node with port 0 (ephemeral OS port)
    node = Node(node_id="worker_live", transport=mock_transport, port=0)
    assert node.server is not None
    assert node.listening_port is not None and node.listening_port > 0
    assert node.address == f"127.0.0.1:{node.listening_port}"

    try:
        # Registration should now pass address
        reg_response = node.register()
        assert reg_response.success is True
        mock_transport.register_node.assert_called_once_with(
            "worker_live", address=node.address
        )

        # Call TrainRound on the node's listener via gRPC stub
        channel = grpc.insecure_channel(node.address)
        stub = NodeServiceStub(channel)
        train_res = stub.TrainRound(
            TrainRequest(round_id=2, weights=b"global_weights", epochs=1)
        )
        assert train_res.success is True
        assert train_res.node_id == "worker_live"
        channel.close()
    finally:
        node.stop(grace=0)


def test_node_train_local_with_data() -> None:
    # 8 dummy CIFAR-10 images (3x32x32) with 10 classes
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    model = SimpleCNN(seed=42)
    initial_weights = model_to_bytes(model)

    node = Node(node_id="client_train_test", model=model, data=loader, lr=0.05)
    updated_weights, num_samples = node.train_local(epochs=1)

    assert num_samples == 8
    assert len(updated_weights) == len(initial_weights)
    # Model parameters must have updated after training
    assert updated_weights != initial_weights


def test_node_on_train_round_execution() -> None:
    inputs = torch.randn(12, 3, 32, 32)
    targets = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=6)

    # Node has local model and data
    local_model = SimpleCNN(seed=1)
    node = Node(node_id="worker_live_train", model=local_model, data=loader, lr=0.01)

    # Server sends global weights from seed=42
    server_model = SimpleCNN(seed=42)
    server_weights = model_to_bytes(server_model)

    request = TrainRequest(round_id=1, weights=server_weights, epochs=1)
    response = node.on_train_round(request)

    assert response.success is True
    assert response.node_id == "worker_live_train"
    assert response.num_samples == 12
    # Returned weights should differ from server weights because local gradient step occurred
    assert response.weights != server_weights
    assert len(response.weights) == len(server_weights)
