from unittest.mock import MagicMock
from src.client.node import Node
from src.transport.federated_pb2 import RegisterResponse, UpdateResponse


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
