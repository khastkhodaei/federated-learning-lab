"""Tests for centralized round dispatch module."""

from unittest.mock import MagicMock

from src.server.dispatch import dispatch_round
from src.server.server import FederatedServer
from src.transport.federated_pb2 import TrainResponse


def test_dispatch_round_forwards_parameters_without_weights() -> None:
    """dispatch_round must invoke trigger_round with aggregate=True and omit weights."""
    mock_server = MagicMock(spec=FederatedServer)
    dummy_resp = [TrainResponse(success=True, node_id="client_0", num_samples=10)]
    mock_server.trigger_round.return_value = dummy_resp

    responses = dispatch_round(mock_server, round_id=3, epochs=2, timeout=12.5)
    assert responses == dummy_resp

    mock_server.trigger_round.assert_called_once_with(
        round_id=3,
        epochs=2,
        timeout=12.5,
        aggregate=True,
    )


def test_dispatch_round_preserves_client_weights_path() -> None:
    """When server.client_weights is present, dispatch_round must allow trigger_round

    to deliver client_weights by not passing a global weights override.
    """
    server = FederatedServer.__new__(FederatedServer)
    server.client_weights = {
        "client_0": b"CLIENT_0_CUSTOM_WEIGHTS",
    }
    server.servicer = MagicMock()
    server.servicer.registered_nodes = {
        "client_0": "localhost:50052",
    }
    server.get_global_weights = MagicMock(return_value=b"GLOBAL_WEIGHTS")
    server.aggregate_responses = MagicMock(return_value=b"AGGREGATED_WEIGHTS")

    delivered_weights = {}

    def fake_call_node(nid, addr):
        # Emulate _call_node payload selection from trigger_round
        payload = server.client_weights.get(nid, server.get_global_weights())
        delivered_weights[nid] = payload
        return TrainResponse(
            success=True,
            node_id=nid,
            num_samples=10,
            weights=payload,
            message="ok",
        )

    # Calling dispatch_round on server
    def wrapped_trigger(round_id, epochs=1, timeout=None, aggregate=True, **kwargs):
        assert "weights" not in kwargs or kwargs["weights"] is None
        return [fake_call_node("client_0", "localhost:50052")]

    server.trigger_round = wrapped_trigger

    res = dispatch_round(server, round_id=1, epochs=1)
    assert len(res) == 1
    assert delivered_weights["client_0"] == b"CLIENT_0_CUSTOM_WEIGHTS"
