from unittest.mock import MagicMock
from src.transport.client import FederatedClient
from src.transport.federated_pb2 import RegisterResponse, UpdateResponse


def test_federated_client_register() -> None:
    client = FederatedClient(channel=MagicMock())
    client.stub.RegisterNode = MagicMock(
        return_value=RegisterResponse(success=True, message="ok")
    )

    response = client.register_node("node_1")

    assert response.success is True
    assert response.message == "ok"
    client.stub.RegisterNode.assert_called_once()
    called_request = client.stub.RegisterNode.call_args[0][0]
    assert called_request.node_id == "node_1"


def test_federated_client_submit_update() -> None:
    client = FederatedClient(channel=MagicMock())
    client.stub.SubmitUpdate = MagicMock(
        return_value=UpdateResponse(success=True, message="received")
    )

    response = client.submit_update("node_1", 100, b"serialized_weights")

    assert response.success is True
    assert response.message == "received"
    client.stub.SubmitUpdate.assert_called_once()
    called_request = client.stub.SubmitUpdate.call_args[0][0]
    assert called_request.node_id == "node_1"
    assert called_request.num_samples == 100
    assert called_request.weights == b"serialized_weights"
