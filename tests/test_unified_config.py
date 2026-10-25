"""Unit tests for the Single Source of Truth UnifiedConfig schema and loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    BaselinesConfig,
    DatasetConfig,
    FedProxConfig,
    GlobalConfig,
    ModelConfig,
    OptimizerConfig,
    PdflConfig,
    TrainingConfig,
    UnifiedConfig,
    load_unified_config,
)


def test_unified_config_defaults() -> None:
    cfg = UnifiedConfig()
    assert cfg.experiment_name == "cifar10_federated"
    assert cfg.baseline == "fedavg"
    assert cfg.num_clients == 3
    assert cfg.rounds == 15
    assert cfg.epochs == 2
    assert cfg.device == "cuda"
    assert cfg.seed == 42
    assert cfg.server_host == "127.0.0.1"
    assert cfg.server_port == 50051

    # Nested sub-configs
    assert isinstance(cfg.dataset, DatasetConfig)
    assert cfg.dataset.name == "cifar10"
    assert cfg.dataset.partition == "dirichlet"
    assert cfg.dataset.alpha == 0.1

    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.name == "simple_cnn"

    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.training.optimizer, OptimizerConfig)
    assert cfg.training.learning_rate == 0.01

    assert isinstance(cfg.baselines, BaselinesConfig)
    assert isinstance(cfg.baselines.fedprox, FedProxConfig)
    assert cfg.baselines.fedprox.mu == 0.01
    assert isinstance(cfg.baselines.pdfl, PdflConfig)
    assert cfg.baselines.pdfl.similarity_threshold == 0.5


def test_load_unified_config_from_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "config.yaml"
        raw_yaml = {
            "experiment_name": "custom_experiment",
            "baseline": "fedprox",
            "num_clients": 5,
            "rounds": 20,
            "epochs": 3,
            "device": "cpu",
            "seed": 999,
            "server_host": "0.0.0.0",
            "server_port": 50055,
            "dataset": {
                "name": "mnist",
                "partition": "iid",
                "alpha": 0.5,
                "batch_size": 64,
                "data_dir": "./custom_data",
                "val_ratio": 0.1,
            },
            "model": {"name": "simple_cnn", "seed": 999},
            "training": {
                "learning_rate": 0.05,
                "loss": "cross_entropy",
                "optimizer": {
                    "name": "adam",
                    "momentum": 0.0,
                    "weight_decay": 0.001,
                },
            },
            "baselines": {
                "fedavg": {},
                "fedprox": {"mu": 0.05},
                "scaffold": {},
                "pdfl": {"similarity_threshold": 0.8},
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(raw_yaml, f)

        cfg = load_unified_config(yaml_path)
        assert cfg.experiment_name == "custom_experiment"
        assert cfg.baseline == "fedprox"
        assert cfg.num_clients == 5
        assert cfg.rounds == 20
        assert cfg.epochs == 3
        assert cfg.device == "cpu"
        assert cfg.seed == 999
        assert cfg.server_port == 50055
        assert cfg.dataset.name == "mnist"
        assert cfg.dataset.batch_size == 64
        assert cfg.dataset.val_ratio == 0.1

        # Type-safe baseline accessors
        fedprox_cfg = cfg.get_baseline_config()
        assert isinstance(fedprox_cfg, FedProxConfig)
        assert fedprox_cfg.mu == 0.05

        pdfl_cfg = cfg.get_baseline_config("pdfl")
        assert isinstance(pdfl_cfg, PdflConfig)
        assert pdfl_cfg.similarity_threshold == 0.8


def test_legacy_config_hard_break() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        server_yaml = Path(tmp_dir) / "server.yaml"
        client_yaml = Path(tmp_dir) / "client.yaml"
        server_yaml.write_text("port: 50051", encoding="utf-8")
        client_yaml.write_text("server_address: 127.0.0.1:50051", encoding="utf-8")

        with pytest.raises(ValueError, match="Legacy split configuration"):
            load_unified_config(server_yaml)

        with pytest.raises(ValueError, match="Legacy split configuration"):
            load_unified_config(client_yaml)


def test_unified_config_to_client_config() -> None:
    cfg = UnifiedConfig()
    client_cfg = cfg.to_client_config(client_id=2)

    assert client_cfg.node_id == "client_2"
    assert client_cfg.server_address == "127.0.0.1:50051"
    assert client_cfg.dataset.client_id == 2
    assert client_cfg.dataset.num_clients == 3
    assert client_cfg.dataset.name == "cifar10"
    assert client_cfg.dataset.alpha == 0.1
    assert client_cfg.training.device == "cuda"
    assert client_cfg.training.learning_rate == 0.01


def test_global_config_alias() -> None:
    assert GlobalConfig is UnifiedConfig
