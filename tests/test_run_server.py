"""Tests for standalone server runner and strategy factory."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from run_server import resolve_server_settings
from src.baselines import (
    DittoServerStrategy,
    FedAvgServerStrategy,
    FedProxServerStrategy,
    PerFedAvgServerStrategy,
    PFedGraphServerStrategy,
    create_server_strategy,
)
from src.server.server import FederatedServer
from src.transport.federated_pb2 import TrainResponse
from src.utils.config import (
    BaselinesConfig,
    PFedGraphConfig,
    UnifiedConfig,
    load_unified_config,
)


def test_resolve_server_settings_from_config() -> None:
    """Effective settings must come from UnifiedConfig when no CLI overrides are provided."""
    args = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
    )
    u_cfg = UnifiedConfig(
        server_host="192.168.1.100",
        server_port=50099,
        num_clients=7,
        rounds=25,
        epochs=4,
        seed=1337,
        device="cpu",
        baseline="fedprox",
    )

    settings = resolve_server_settings(args, u_cfg)

    assert settings.host == "192.168.1.100"
    assert settings.port == 50099
    assert settings.num_clients == 7
    assert settings.clients == 7
    assert settings.rounds == 25
    assert settings.epochs == 4
    assert settings.seed == 1337
    assert settings.device == "cpu"
    assert settings.baseline == "fedprox"
    assert settings.report_dir == "reports"


def test_resolve_server_settings_cli_overrides_win() -> None:
    """CLI overrides must take precedence over UnifiedConfig values when explicitly provided."""
    args = argparse.Namespace(
        host="0.0.0.0",
        port=8080,
        clients=10,
        rounds=100,
        epochs=5,
        seed=999,
        device="cpu",
        baseline="ditto",
        report_dir="custom_reports",
    )
    u_cfg = UnifiedConfig(
        server_host="127.0.0.1",
        server_port=50051,
        num_clients=3,
        rounds=15,
        epochs=1,
        seed=42,
        device="cuda",
        baseline="fedavg",
    )

    settings = resolve_server_settings(args, u_cfg)

    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.num_clients == 10
    assert settings.rounds == 100
    assert settings.epochs == 5
    assert settings.seed == 999
    assert settings.device == "cpu"
    assert settings.baseline == "ditto"
    assert settings.report_dir == "custom_reports"


def test_resolve_server_settings_regression_real_unified_config() -> None:
    """Regression test exercising resolve_server_settings with the real configs/config.yaml.

    The legacy run_server.py accessed removed attributes like u_cfg.server.port, which
    raises AttributeError: 'UnifiedConfig' object has no attribute 'server'.
    """
    u_cfg = load_unified_config("configs/config.yaml")
    args = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
    )

    settings = resolve_server_settings(args, u_cfg)

    assert settings.port == 50051
    assert settings.num_clients == 3
    assert settings.rounds == 50
    assert settings.epochs == 1
    assert settings.seed == 42
    assert settings.baseline == "fedavg"


@pytest.mark.parametrize(
    ("baseline_name", "expected_cls"),
    [
        ("fedavg", FedAvgServerStrategy),
        ("fedprox", FedProxServerStrategy),
        ("ditto", DittoServerStrategy),
        ("perfedavg", PerFedAvgServerStrategy),
        ("pfedgraph", PFedGraphServerStrategy),
    ],
)
def test_strategy_factory_all_baselines(baseline_name: str, expected_cls: type) -> None:
    """Strategy factory must instantiate the correct class and pFedGraph alpha/lam from config."""
    u_cfg = UnifiedConfig(
        baseline=baseline_name,
        baselines=BaselinesConfig(
            pfedgraph=PFedGraphConfig(alpha=0.35, lam=0.07),
        ),
    )

    strategy = create_server_strategy(baseline_name, config=u_cfg)
    assert isinstance(strategy, expected_cls)

    if baseline_name == "pfedgraph":
        assert isinstance(strategy, PFedGraphServerStrategy)
        assert strategy.alpha == pytest.approx(0.35)
        assert strategy.lam == pytest.approx(0.07)

        # Explicit overrides must take precedence
        override_strategy = create_server_strategy(
            baseline_name,
            config=u_cfg,
            alpha=0.5,
            lam=0.02,
        )
        assert isinstance(override_strategy, PFedGraphServerStrategy)
        assert override_strategy.alpha == pytest.approx(0.5)
        assert override_strategy.lam == pytest.approx(0.02)


def test_server_trigger_round_dispatch_client_weights() -> None:
    """When server.client_weights is populated, trigger_round must send per-client payloads;

    when empty, it must fall back to the global model weights.
    """
    server = FederatedServer.__new__(FederatedServer)
    server.client_weights = {}
    server.servicer = MagicMock()
    server.servicer.registered_nodes = {
        "client_0": "localhost:50052",
        "client_1": "localhost:50053",
    }
    global_bytes = b"GLOBAL_MODEL_PAYLOAD"
    server.get_global_weights = MagicMock(return_value=global_bytes)

    with patch("grpc.insecure_channel"), patch(
        "src.server.server.NodeServiceStub"
    ) as mock_stub_cls:
        mock_stub = MagicMock()

        def fake_train_round(req, timeout=None):
            # Record payload delivered to each node
            return TrainResponse(
                success=True,
                node_id="client",
                num_samples=10,
                weights=req.weights,
                message="",
            )

        mock_stub.TrainRound.side_effect = fake_train_round
        mock_stub_cls.return_value = mock_stub

        # Case 1: Empty client_weights -> All clients receive global model weights
        server.client_weights = {}
        responses_global = server.trigger_round(round_id=1, weights=None, epochs=1)
        assert len(responses_global) == 2
        for resp in responses_global:
            assert resp.weights == global_bytes

        # Case 2: Populated client_weights -> Each client receives its personalized payload
        c0_bytes = b"CLIENT_0_PERSONALIZED_PAYLOAD"
        c1_bytes = b"CLIENT_1_PERSONALIZED_PAYLOAD"
        server.client_weights = {
            "client_0": c0_bytes,
            "client_1": c1_bytes,
        }

        responses_personalized = server.trigger_round(round_id=2, weights=None, epochs=1)
        assert len(responses_personalized) == 2
        delivered_payloads = {resp.weights for resp in responses_personalized}
        assert delivered_payloads == {c0_bytes, c1_bytes}
