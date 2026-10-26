"""Tests for standalone client and simulation configuration parity and device resolution."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from run_client import build_client_dataloaders
from run_server import resolve_server_settings
from run_simulation import run_federated_simulation
from src.baselines import (
    DittoClientTrainer,
    FedAvgClientTrainer,
    FedProxClientTrainer,
    PerFedAvgClientTrainer,
    PFedGraphClientTrainer,
)
from src.client.config import (
    ClientConfig,
    create_trainer_from_config,
    load_client_config,
)
from src.utils.config import load_unified_config
from src.utils.hardware import resolve_device


def test_load_client_config_fallbacks(tmp_path: Path) -> None:
    """load_client_config falls back to top-level fields when nested fields are absent."""
    # 1. Real configs/config.yaml has top-level num_clients=3, device="cuda", seed=42
    cfg = load_client_config("configs/config.yaml")
    assert cfg.dataset.num_clients == 3
    assert cfg.training.device == "cuda"
    assert cfg.dataset.seed == 42

    # 2. Nested values win over top-level defaults
    nested_yaml = tmp_path / "nested.yaml"
    nested_yaml.write_text(
        """
num_clients: 3
device: "cuda"
seed: 42
dataset:
  name: "cifar10"
  num_clients: 5
  seed: 77
training:
  device: "cuda:1"
""",
        encoding="utf-8",
    )
    nested_cfg = load_client_config(nested_yaml)
    assert nested_cfg.dataset.num_clients == 5
    assert nested_cfg.training.device == "cuda:1"
    assert nested_cfg.dataset.seed == 77

    # 3. Explicit argument wins over everything
    override_cfg = load_client_config("configs/config.yaml", num_clients=9)
    assert override_cfg.dataset.num_clients == 9


def test_algorithm_baseline_conflict_validation(tmp_path: Path) -> None:
    """Both loaders must raise ValueError if algorithm and baseline are both present and differ."""
    conflict_yaml = tmp_path / "conflict.yaml"
    conflict_yaml.write_text(
        """
algorithm: "fedavg"
baseline: "fedprox"
num_clients: 2
rounds: 1
epochs: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="[Cc]onflict"):
        load_unified_config(conflict_yaml)

    with pytest.raises(ValueError, match="[Cc]onflict"):
        load_client_config(conflict_yaml)


def test_create_trainer_from_config() -> None:
    """create_trainer_from_config instantiates correct class and passes baseline hyperparameters."""
    # FedProx
    cfg_prox = ClientConfig(algorithm="fedprox")
    cfg_prox.fedprox.mu = 0.05
    trainer_prox = create_trainer_from_config(cfg_prox)
    assert isinstance(trainer_prox, FedProxClientTrainer)
    assert trainer_prox.mu == pytest.approx(0.05)

    # Ditto
    cfg_ditto = ClientConfig(algorithm="ditto")
    cfg_ditto.ditto.lambda_penalty = 0.25
    cfg_ditto.ditto.personalized_epochs = 3
    trainer_ditto = create_trainer_from_config(cfg_ditto)
    assert isinstance(trainer_ditto, DittoClientTrainer)
    assert trainer_ditto.lambda_penalty == pytest.approx(0.25)
    assert trainer_ditto.personalized_epochs == 3

    # Per-FedAvg
    cfg_perfedavg = ClientConfig(algorithm="perfedavg")
    cfg_perfedavg.perfedavg.alpha_lr = 0.02
    cfg_perfedavg.perfedavg.beta_lr = 0.001
    trainer_perfedavg = create_trainer_from_config(cfg_perfedavg)
    assert isinstance(trainer_perfedavg, PerFedAvgClientTrainer)
    assert trainer_perfedavg.alpha_lr == pytest.approx(0.02)
    assert trainer_perfedavg.beta_lr == pytest.approx(0.001)

    # FedAvg
    cfg_fedavg = ClientConfig(algorithm="fedavg")
    trainer_fedavg = create_trainer_from_config(cfg_fedavg)
    assert isinstance(trainer_fedavg, FedAvgClientTrainer)

    # pFedGraph
    cfg_pfedgraph = ClientConfig(algorithm="pfedgraph")
    trainer_pfedgraph = create_trainer_from_config(cfg_pfedgraph)
    assert isinstance(trainer_pfedgraph, PFedGraphClientTrainer)


def test_standalone_config_algorithm_and_baseline_parity(tmp_path: Path) -> None:
    """Config with only algorithm or only baseline correctly configures trainer."""
    algo_yaml = tmp_path / "algo_only.yaml"
    algo_yaml.write_text(
        """
algorithm: "fedprox"
fedprox:
  mu: 0.08
""",
        encoding="utf-8",
    )
    cfg1 = load_client_config(algo_yaml)
    assert cfg1.algorithm == "fedprox"
    t1 = create_trainer_from_config(cfg1)
    assert isinstance(t1, FedProxClientTrainer)
    assert t1.mu == pytest.approx(0.08)

    base_yaml = tmp_path / "base_only.yaml"
    base_yaml.write_text(
        """
baseline: "fedprox"
fedprox:
  mu: 0.08
""",
        encoding="utf-8",
    )
    cfg2 = load_client_config(base_yaml)
    assert cfg2.algorithm == "fedprox"
    t2 = create_trainer_from_config(cfg2)
    assert isinstance(t2, FedProxClientTrainer)
    assert t2.mu == pytest.approx(0.08)


def test_build_client_dataloaders_passes_transform() -> None:
    """Dataset construction helper must pass non-None torchvision transform to FederatedDataset."""
    cfg = ClientConfig(node_id="client_0")
    cfg.dataset.name = "cifar10"
    cfg.dataset.val_ratio = 0.0

    mock_ds = MagicMock()
    mock_ds.get_client_dataloader.return_value = MagicMock()
    mock_ds.get_client_indices.return_value = [0, 1, 2]

    with patch("run_client.FederatedDataset", return_value=mock_ds) as mock_fed_cls:
        train_loader, val_loader = build_client_dataloaders(cfg, client_id=0)
        assert train_loader is not None
        assert val_loader is None
        mock_fed_cls.assert_called_once()
        _, kwargs = mock_fed_cls.call_args
        assert kwargs.get("transform") is not None


def test_run_federated_simulation_defers_to_yaml(tmp_path: Path) -> None:
    """When CLI parameters are omitted (None), run_federated_simulation reads YAML values."""
    sim_yaml = tmp_path / "sim.yaml"
    sim_yaml.write_text(
        """
experiment_name: "test_sim"
algorithm: "fedavg"
num_clients: 2
rounds: 1
epochs: 1
device: "cpu"
seed: 42
server_host: "127.0.0.1"
server_port: 0
dataset:
  name: "cifar10"
  batch_size: 16
""",
        encoding="utf-8",
    )

    results = run_federated_simulation(
        config_path=str(sim_yaml),
        synthetic=True,
        reports_dir=tmp_path / "reports",
    )

    assert results["num_clients"] == 2
    assert results["rounds_completed"] == 1
    assert len(results["round_history"]) == 1


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_device handles CUDA availability, explicit devices, and CPU fallback."""
    # 1. When CUDA is available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == "cuda"
    assert resolve_device("") == "cuda"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("cpu") == "cpu"

    # 2. When CUDA is NOT available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == "cpu"
    assert resolve_device("") == "cpu"
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cuda:0") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_server_device_resolution_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_server.resolve_server_settings resolves 'cuda' to 'cpu' when CUDA unavailable without altering config."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    u_cfg = load_unified_config("configs/config.yaml")
    assert u_cfg.device == "cuda"

    args = argparse.Namespace(
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
    settings = resolve_server_settings(args, u_cfg)
    assert settings.device == "cpu"
    assert u_cfg.device == "cuda"
