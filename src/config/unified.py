"""Unified experiment configuration schema and loader.

Single source of truth for federated coordinator server, client nodes,
dataset partitions, and baseline aggregation algorithms.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.client.config import (
    ClientConfig,
)
from src.client.config import (
    DatasetConfig as ClientDatasetConfig,
)
from src.client.config import (
    ModelConfig as ClientModelConfig,
)
from src.client.config import (
    OptimizerConfig as ClientOptimizerConfig,
)
from src.client.config import (
    TrainingConfig as ClientTrainingConfig,
)


@dataclass
class ExperimentConfig:
    """General experiment metadata and compute environment."""

    name: str = "cifar10_federated"
    seed: int = 42
    device: str = "cuda"


@dataclass
class FederatedConfig:
    """Federated coordination hyperparameters."""

    num_clients: int = 3
    rounds: int = 15
    epochs_per_round: int = 2


@dataclass
class ServerConfig:
    """gRPC networking configuration for coordinator."""

    host: str = "127.0.0.1"
    port: int = 50051


@dataclass
class DatasetConfig:
    """Benchmark dataset partitioning specification."""

    name: str = "cifar10"
    partition: str = "dirichlet"
    alpha: float = 0.1
    batch_size: int = 32
    data_dir: str = "./data"
    seed: int = 42
    val_ratio: float = 0.2
    num_clients: int = 3


@dataclass
class ModelConfig:
    """Neural network architecture specification."""

    name: str = "simple_cnn"
    seed: int = 42


@dataclass
class OptimizerConfig:
    """Local client optimizer hyperparameters."""

    name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0001


@dataclass
class TrainingConfig:
    """Client local training hyperparameters."""

    learning_rate: float = 0.01
    loss: str = "cross_entropy"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class UnifiedConfig:
    """Complete, single-source-of-truth configuration for a federated experiment."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    baseline: str = "fedavg"
    baselines: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "fedavg": {},
            "fedprox": {"mu": 0.01},
            "scaffold": {},
            "pdfl": {"similarity_threshold": 0.5},
        }
    )

    @property
    def num_clients(self) -> int:
        """Convenience property for total participating clients."""
        return self.federated.num_clients

    @property
    def rounds(self) -> int:
        """Convenience property for total federated rounds."""
        return self.federated.rounds

    @property
    def epochs(self) -> int:
        """Convenience property for local epochs per round."""
        return self.federated.epochs_per_round

    @property
    def device(self) -> str:
        """Convenience property for compute device."""
        return self.experiment.device

    def get_baseline_config(
        self, baseline_name: str | None = None
    ) -> dict[str, Any]:
        """Returns hyperparameters for the requested or currently active baseline."""
        key = (baseline_name or self.baseline).lower().strip()
        return dict(self.baselines.get(key, {}))

    def to_client_config(
        self, client_id: int, client_port: int = 0
    ) -> ClientConfig:
        """Converts to ClientConfig expected by Node and client runner."""
        return ClientConfig(
            node_id=f"client_{client_id}",
            server_address=f"{self.server.host}:{self.server.port}",
            client_host=self.server.host,
            client_port=client_port,
            model=ClientModelConfig(
                name=self.model.name,
                seed=self.model.seed,
            ),
            dataset=ClientDatasetConfig(
                name=self.dataset.name,
                batch_size=self.dataset.batch_size,
                client_id=client_id,
                num_clients=self.federated.num_clients,
                partition=self.dataset.partition,
                alpha=self.dataset.alpha,
                data_dir=self.dataset.data_dir,
                seed=self.dataset.seed,
                val_ratio=self.dataset.val_ratio,
            ),
            training=ClientTrainingConfig(
                learning_rate=self.training.learning_rate,
                loss=self.training.loss,
                optimizer=ClientOptimizerConfig(
                    name=self.training.optimizer.name,
                    momentum=self.training.optimizer.momentum,
                    weight_decay=self.training.optimizer.weight_decay,
                ),
                device=self.experiment.device,
            ),
        )


def load_unified_config(config_path: str | Path) -> UnifiedConfig:
    """Loads a UnifiedConfig from a unified YAML file or legacy split YAML file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Populated UnifiedConfig instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Check for legacy formats
    is_legacy_server = "epochs_per_round" in raw and "server" not in raw
    is_legacy_client = "server_address" in raw and "server" not in raw

    if is_legacy_server or is_legacy_client:
        warnings.warn(
            f"Loading legacy configuration from '{path.name}'. "
            "Please migrate to the unified 'configs/config.yaml'.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    # 1. Experiment
    exp_data = raw.get("experiment", {})
    exp_device = (
        exp_data.get("device")
        or raw.get("device")
        or raw.get("training", {}).get("device", "cuda")
    )
    exp_cfg = ExperimentConfig(
        name=exp_data.get("name", raw.get("name", "cifar10_federated")),
        seed=int(
            exp_data.get("seed", raw.get("model", {}).get("seed", 42))
        ),
        device=str(exp_device),
    )

    # 2. Federated
    fed_data = raw.get("federated", {})
    num_clients = (
        fed_data.get("num_clients")
        or raw.get("num_clients")
        or raw.get("dataset", {}).get("num_clients", 3)
    )
    rounds = fed_data.get("rounds") or raw.get("rounds", 15)
    epochs = (
        fed_data.get("epochs_per_round")
        or raw.get("epochs_per_round")
        or raw.get("epochs", 2)
    )
    fed_cfg = FederatedConfig(
        num_clients=int(num_clients),
        rounds=int(rounds),
        epochs_per_round=int(epochs),
    )

    # 3. Server
    srv_data = raw.get("server", {})
    host = srv_data.get("host") or raw.get("host", "127.0.0.1")
    port = srv_data.get("port") or raw.get("port")
    if port is None and "server_address" in raw:
        addr = str(raw["server_address"])
        if ":" in addr:
            host, port_str = addr.split(":", 1)
            port = int(port_str)
        else:
            port = 50051
    elif port is None:
        port = 50051

    srv_cfg = ServerConfig(
        host=str(host),
        port=int(port),
    )

    # 4. Dataset
    ds_data = raw.get("dataset", {})
    ds_cfg = DatasetConfig(
        name=str(ds_data.get("name", "cifar10")),
        partition=str(ds_data.get("partition", "dirichlet")),
        alpha=float(ds_data.get("alpha", 0.1)),
        batch_size=int(ds_data.get("batch_size", 32)),
        data_dir=str(ds_data.get("data_dir", "./data")),
        seed=int(ds_data.get("seed", 42)),
        val_ratio=float(ds_data.get("val_ratio", 0.2)),
        num_clients=int(num_clients),
    )

    # 5. Model
    model_data = raw.get("model", {})
    model_cfg = ModelConfig(
        name=str(model_data.get("name", "simple_cnn")),
        seed=int(model_data.get("seed", 42)),
    )

    # 6. Training & Optimizer
    tr_data = raw.get("training", {})
    opt_data = tr_data.get("optimizer", {})
    opt_cfg = OptimizerConfig(
        name=str(opt_data.get("name", "sgd")),
        momentum=float(opt_data.get("momentum", 0.9)),
        weight_decay=float(opt_data.get("weight_decay", 0.0001)),
    )
    tr_cfg = TrainingConfig(
        learning_rate=float(tr_data.get("learning_rate", 0.01)),
        loss=str(tr_data.get("loss", "cross_entropy")),
        optimizer=opt_cfg,
    )

    # 7. Baseline & Baselines
    baseline = str(raw.get("baseline", "fedavg")).lower().strip()
    baselines_dict = raw.get("baselines", {})
    if not baselines_dict:
        baselines_dict = {
            "fedavg": {},
            "fedprox": {"mu": 0.01},
            "scaffold": {},
            "pdfl": {"similarity_threshold": 0.5},
        }

    return UnifiedConfig(
        experiment=exp_cfg,
        federated=fed_cfg,
        server=srv_cfg,
        dataset=ds_cfg,
        model=model_cfg,
        training=tr_cfg,
        baseline=baseline,
        baselines=baselines_dict,
    )
