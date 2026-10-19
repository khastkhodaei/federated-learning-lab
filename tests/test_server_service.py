from unittest.mock import MagicMock
from src.server.service import FederatedServerServicer
from src.transport.federated_pb2 import RegisterRequest, UpdateRequest


def test_register_node() -> None:
    servicer = FederatedServerServicer()
    context = MagicMock()
    request = RegisterRequest(node_id="client_1")

    response = servicer.RegisterNode(request, context)

    assert response.success is True
    assert "client_1" in servicer.registered_nodes


def test_submit_update() -> None:
    servicer = FederatedServerServicer()
    context = MagicMock()
    request = UpdateRequest(node_id="client_1", num_samples=50, weights=b"dummy_weights")

    response = servicer.SubmitUpdate(request, context)

    assert response.success is True
    assert "client_1" in servicer.updates
    assert servicer.updates["client_1"].num_samples == 50
    assert servicer.updates["client_1"].weights == b"dummy_weights"
