"""Tests for standalone server runner and strategy factory."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from run_server import ServerSettings, dispatch_round, resolve_server_settings
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


def test_resolve_server_settings_round_timeout() -> None:
    """resolve_server_settings must default round_timeout to 60.0, honor overrides, and reject non-positive."""
    u_cfg = UnifiedConfig()

    # Default
    args_default = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        round_timeout=None,
    )
    s_default = resolve_server_settings(args_default, u_cfg)
    assert s_default.round_timeout == 60.0

    # Explicit override
    args_override = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        round_timeout=15.0,
    )
    s_override = resolve_server_settings(args_override, u_cfg)
    assert s_override.round_timeout == 15.0

    # Reject zero
    args_zero = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        round_timeout=0.0,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        resolve_server_settings(args_zero, u_cfg)

    # Reject negative
    args_neg = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        round_timeout=-5.0,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        resolve_server_settings(args_neg, u_cfg)


def test_resolve_server_settings_report_dir_precedence() -> None:
    """report_dir must default to 'reports' when args.report_dir is None and honor explicit overrides."""
    u_cfg = UnifiedConfig()

    args_none = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        report_dir=None,
    )
    s_none = resolve_server_settings(args_none, u_cfg)
    assert s_none.report_dir == "reports"

    args_custom = argparse.Namespace(
        host=None,
        port=None,
        clients=None,
        rounds=None,
        epochs=None,
        seed=None,
        device=None,
        baseline=None,
        report_dir="custom_dir",
    )
    s_custom = resolve_server_settings(args_custom, u_cfg)
    assert s_custom.report_dir == "custom_dir"


def test_dispatch_round() -> None:
    """dispatch_round must trigger round with timeout, epochs, aggregate=True, and no weights."""
    mock_server = MagicMock(spec=FederatedServer)
    mock_server.trigger_round.return_value = []

    settings = ServerSettings(
        host="127.0.0.1",
        port=50051,
        num_clients=3,
        rounds=10,
        epochs=3,
        device="cpu",
        seed=42,
        baseline="fedavg",
        report_dir="reports",
        round_timeout=45.0,
    )

    res = dispatch_round(mock_server, settings, round_id=5)
    assert res == []

    mock_server.trigger_round.assert_called_once()
    kwargs = mock_server.trigger_round.call_args.kwargs
    assert kwargs.get("round_id") == 5
    assert kwargs.get("epochs") == 3
    assert kwargs.get("timeout") == 45.0
    assert kwargs.get("aggregate") is True
    assert "weights" not in kwargs


def test_server_trigger_round_forwards_timeout() -> None:
    """trigger_round must forward the timeout argument to NodeServiceStub.TrainRound."""
    server = FederatedServer.__new__(FederatedServer)
    server.client_weights = {}
    server.servicer = MagicMock()
    server.servicer.registered_nodes = {
        "client_0": "localhost:50052",
    }
    server.get_global_weights = MagicMock(return_value=b"GLOBAL_BYTES")

    recorded_timeouts = []
    with patch("grpc.insecure_channel"), patch(
        "src.server.server.NodeServiceStub"
    ) as mock_stub_cls:
        mock_stub = MagicMock()

        def fake_train_round(req, timeout=None):
            recorded_timeouts.append(timeout)
            return TrainResponse(
                success=True,
                node_id="client_0",
                num_samples=10,
                weights=req.weights,
                message="",
            )

        mock_stub.TrainRound.side_effect = fake_train_round
        mock_stub_cls.return_value = mock_stub

        server.trigger_round(round_id=1, epochs=1, timeout=7.5)
        assert recorded_timeouts == [7.5]


def test_server_trigger_round_failing_node_does_not_block_sibling() -> None:
    """A failing or timed-out node must not prevent successful siblings from returning updates."""
    server = FederatedServer.__new__(FederatedServer)
    server.client_weights = {}
    server.servicer = MagicMock()
    server.servicer.registered_nodes = {
        "client_fail": "localhost:50052",
        "client_ok": "localhost:50053",
    }
    server.get_global_weights = MagicMock(return_value=b"GLOBAL_BYTES")

    with patch("grpc.insecure_channel"), patch(
        "src.server.server.NodeServiceStub"
    ) as mock_stub_cls:
        def fake_train_round(req, timeout=None):
            # One node throws an exception (simulating timeout or connection error)
            if "fail" in req.weights.decode("utf-8", errors="ignore"):
                raise RuntimeError("RPC deadline exceeded / connection failure")
            return TrainResponse(
                success=True,
                node_id="client_ok",
                num_samples=25,
                weights=b"OK_WEIGHTS",
                message="success",
            )

        server.client_weights = {
            "client_fail": b"payload_fail",
            "client_ok": b"payload_ok",
        }

        mock_stub = MagicMock()
        mock_stub.TrainRound.side_effect = fake_train_round
        mock_stub_cls.return_value = mock_stub

        responses = server.trigger_round(round_id=1, epochs=1, timeout=5.0)
        assert len(responses) == 1
        assert responses[0].node_id == "client_ok"
        assert responses[0].num_samples == 25


def test_load_unified_config_legacy_content_guard(tmp_path) -> None:
    """load_unified_config must reject legacy client configs containing server_address even if not named client.yaml."""
    legacy_file = tmp_path / "old_client.yaml"
    legacy_file.write_text("server_address: 127.0.0.1:50051\nnum_clients: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Legacy client configuration detected"):
        load_unified_config(legacy_file)

