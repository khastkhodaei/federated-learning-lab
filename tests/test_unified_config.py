"""Unit tests for UnifiedConfig loader and schema."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.unified import (
    UnifiedConfig,
    load_unified_config,
)


def test_unified_config_defaults() -> None:
    cfg = UnifiedConfig()
    assert cfg.federated.num_clients == 3
    assert cfg.federated.rounds == 15
    assert cfg.federated.epochs_per_round == 2
    assert cfg.experiment.device == "cuda"
    assert cfg.baseline == "fedavg"
    assert cfg.server.port == 50051
    assert cfg.dataset.name == "cifar10"
    assert cfg.dataset.partition == "dirichlet"
    assert cfg.dataset.alpha == 0.1


def test_load_unified_config_from_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "config.yaml"
        yaml_content = {
            "experiment": {"name": "test_exp", "seed": 123, "device": "cpu"},
            "federated": {"num_clients": 5, "rounds": 10, "epochs_per_round": 3},
            "server": {"host": "0.0.0.0", "port": 50055},
            "dataset": {
                "name": "mnist",
                "partition": "iid",
                "alpha": 0.5,
                "batch_size": 64,
                "data_dir": "./custom_data",
                "seed": 123,
                "val_ratio": 0.1,
            },
            "model": {"name": "simple_cnn", "seed": 123},
            "training": {
                "learning_rate": 0.05,
                "loss": "cross_entropy",
                "optimizer": {"name": "adam", "momentum": 0.0, "weight_decay": 0.001},
            },
            "baseline": "fedprox",
            "baselines": {
                "fedavg": {},
                "fedprox": {"mu": 0.05},
                "scaffold": {},
                "pdfl": {"similarity_threshold": 0.7},
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        cfg = load_unified_config(yaml_path)
        assert cfg.experiment.name == "test_exp"
        assert cfg.experiment.device == "cpu"
        assert cfg.federated.num_clients == 5
        assert cfg.federated.rounds == 10
        assert cfg.federated.epochs_per_round == 3
        assert cfg.server.port == 50055
        assert cfg.dataset.name == "mnist"
        assert cfg.dataset.batch_size == 64
        assert cfg.dataset.val_ratio == 0.1
        assert cfg.training.learning_rate == 0.05
        assert cfg.training.optimizer.name == "adam"
        assert cfg.baseline == "fedprox"
        assert cfg.get_baseline_config() == {"mu": 0.05}
        assert cfg.get_baseline_config("pdfl") == {"similarity_threshold": 0.7}


def test_unified_config_to_client_config() -> None:
    cfg = UnifiedConfig()
    client_cfg = cfg.to_client_config(client_id=1)

    assert client_cfg.node_id == "client_1"
    assert client_cfg.server_address == "127.0.0.1:50051"
    assert client_cfg.dataset.client_id == 1
    assert client_cfg.dataset.num_clients == 3
    assert client_cfg.dataset.name == "cifar10"
    assert client_cfg.dataset.alpha == 0.1
    assert client_cfg.training.device == "cuda"
    assert client_cfg.training.learning_rate == 0.01


def test_unified_config_legacy_server_yaml_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        legacy_server = Path(tmp_dir) / "server.yaml"
        with open(legacy_server, "w", encoding="utf-8") as f:
            yaml.dump({
                "host": "127.0.0.1",
                "port": 50060,
                "num_clients": 4,
                "rounds": 8,
                "epochs_per_round": 2,
                "baseline": "fedavg",
                "device": "cpu",
            }, f)

        with pytest.warns(DeprecationWarning, match="legacy"):
            cfg = load_unified_config(legacy_server)

        assert cfg.server.port == 50060
        assert cfg.federated.num_clients == 4
        assert cfg.federated.rounds == 8
        assert cfg.experiment.device == "cpu"


def test_unified_config_legacy_client_yaml_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        legacy_client = Path(tmp_dir) / "client.yaml"
        with open(legacy_client, "w", encoding="utf-8") as f:
            yaml.dump({
                "server_address": "127.0.0.1:50070",
                "dataset": {
                    "name": "cifar10",
                    "num_clients": 6,
                    "alpha": 0.2,
                },
                "training": {
                    "learning_rate": 0.02,
                    "device": "cuda",
                },
                "baseline": "fedavg",
            }, f)

        with pytest.warns(DeprecationWarning, match="legacy"):
            cfg = load_unified_config(legacy_client)

        assert cfg.server.port == 50070
        assert cfg.dataset.num_clients == 6
        assert cfg.federated.num_clients == 6
        assert cfg.training.learning_rate == 0.02
        assert cfg.experiment.device == "cuda"
