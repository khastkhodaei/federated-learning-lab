"""Single Source of Truth configuration management for Federated Learning Lab.

Implements strongly typed, validated dataclass schemas (Matt Pocock ergonomics)
for central coordinator server, client nodes, dataset partitioning, and baseline strategies.
"""

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
class DatasetConfig:
    """Benchmark dataset partitioning specification."""

    name: str = "cifar10"
    partition: str = "dirichlet"
    alpha: float = 0.1
    batch_size: int = 32
    data_dir: str = "./data"
    val_ratio: float = 0.2
    seed: int = 42


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


# ==============================================================================
# Baseline-Specific Hyperparameter Schemas
# ==============================================================================


@dataclass
class FedAvgConfig:
    """Hyperparameters for vanilla Federated Averaging (FedAvg)."""


@dataclass
class FedProxConfig:
    """Hyperparameters for FedProx with proximal regularization term mu."""

    mu: float = 0.01


@dataclass
class ScaffoldConfig:
    """Hyperparameters for SCAFFOLD with client and server control variates."""


@dataclass
class PdflConfig:
    """Hyperparameters for Personalized Diverse Federated Learning (PDFL)."""

    similarity_threshold: float = 0.5


@dataclass
class DittoConfig:
    """Hyperparameters for Ditto personalized federated learning."""

    lambda_penalty: float = 0.1
    personalized_epochs: int = 1


@dataclass
class PerFedAvgConfig:
    """Hyperparameters for Per-FedAvg meta-learning baseline (FO-MAML)."""

    alpha_lr: float = 0.01
    beta_lr: float = 0.005


@dataclass
class PFedGraphConfig:
    """Hyperparameters for pFedGraph personalized federated learning."""

    alpha: float = 0.1
    lam: float = 0.01


@dataclass
class BaselinesConfig:
    """Registry of typed baseline-specific hyperparameter configurations."""

    fedavg: FedAvgConfig = field(default_factory=FedAvgConfig)
    fedprox: FedProxConfig = field(default_factory=FedProxConfig)
    ditto: DittoConfig = field(default_factory=DittoConfig)
    perfedavg: PerFedAvgConfig = field(default_factory=PerFedAvgConfig)
    pfedgraph: PFedGraphConfig = field(default_factory=PFedGraphConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    pdfl: PdflConfig = field(default_factory=PdflConfig)

    def get(self, baseline_name: str) -> Any:
        """Returns the typed configuration for the requested baseline."""
        key = baseline_name.lower().strip()
        if hasattr(self, key):
            return getattr(self, key)
        return {}


# ==============================================================================
# Unified Configuration
# ==============================================================================


@dataclass
class UnifiedConfig:
    """Single source of truth configuration for the entire federated experiment."""

    experiment_name: str = "cifar10_federated"
    baseline: str = "fedavg"
    algorithm: str = "fedavg"
    num_clients: int = 3
    rounds: int = 15
    epochs: int = 2
    device: str = "cuda"
    seed: int = 42
    server_host: str = "127.0.0.1"
    server_port: int = 50051

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    def get_baseline_config(
        self, baseline_name: str | None = None
    ) -> Any:
        """Type-safe accessor for baseline-specific hyperparameter dataclasses."""
        name = baseline_name or self.baseline
        return self.baselines.get(name)

    def to_client_config(
        self, client_id: int, client_port: int = 0
    ) -> ClientConfig:
        """Translates unified parameters into individual client node specifications."""
        from src.client.config import (
            DittoClientConfig,
            FedProxClientConfig,
            PerFedAvgClientConfig,
            PFedGraphClientConfig,
        )

        algo = (self.algorithm or self.baseline).lower().strip()
        fedprox_mu = getattr(self.baselines.fedprox, "mu", 0.01)
        ditto_cfg = getattr(self.baselines, "ditto", None)
        ditto_lambda = getattr(ditto_cfg, "lambda_penalty", 0.1) if ditto_cfg else 0.1
        ditto_epochs = getattr(ditto_cfg, "personalized_epochs", 1) if ditto_cfg else 1

        perfedavg_cfg = getattr(self.baselines, "perfedavg", None)
        alpha_val = getattr(perfedavg_cfg, "alpha_lr", 0.01) if perfedavg_cfg else 0.01
        beta_val = getattr(perfedavg_cfg, "beta_lr", 0.005) if perfedavg_cfg else 0.005

        pfedgraph_cfg = getattr(self.baselines, "pfedgraph", None)
        graph_alpha = getattr(pfedgraph_cfg, "alpha", 0.1) if pfedgraph_cfg else 0.1
        graph_lam = getattr(pfedgraph_cfg, "lam", 0.01) if pfedgraph_cfg else 0.01

        return ClientConfig(
            node_id=f"client_{client_id}",
            server_address=f"{self.server_host}:{self.server_port}",
            client_host=self.server_host,
            client_port=client_port,
            algorithm=algo,
            model=ClientModelConfig(
                name=self.model.name,
                seed=self.model.seed,
            ),
            dataset=ClientDatasetConfig(
                name=self.dataset.name,
                batch_size=self.dataset.batch_size,
                client_id=client_id,
                num_clients=self.num_clients,
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
                device=self.device,
            ),
            fedprox=FedProxClientConfig(mu=fedprox_mu),
            ditto=DittoClientConfig(
                lambda_penalty=ditto_lambda, personalized_epochs=ditto_epochs
            ),
            perfedavg=PerFedAvgClientConfig(
                alpha_lr=alpha_val,
                beta_lr=beta_val,
            ),
            pfedgraph=PFedGraphClientConfig(
                alpha=graph_alpha,
                lam=graph_lam,
            ),
        )


GlobalConfig = UnifiedConfig


def load_unified_config(config_path: str | Path) -> UnifiedConfig:
    """Loads and validates UnifiedConfig from a single YAML configuration file.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If legacy split configuration files (server.yaml/client.yaml) are provided.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Enforce Hard Break on legacy files
    if path.name in ("server.yaml", "client.yaml"):
        raise ValueError(
            f"Legacy split configuration file '{path.name}' is no longer supported. "
            "Please use the unified 'configs/config.yaml'."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Validate algorithm vs baseline consistency if both are explicitly present
    raw_algo = raw.get("algorithm")
    raw_base = raw.get("baseline")
    if (
        raw_algo is not None
        and raw_base is not None
        and str(raw_algo).lower().strip() != str(raw_base).lower().strip()
    ):
        raise ValueError(
            f"Conflicting 'algorithm' ({raw_algo}) and 'baseline' ({raw_base}) in configuration."
        )

    # 1. Dataset Config
    ds_data = raw.get("dataset", {})
    ds_cfg = DatasetConfig(
        name=str(ds_data.get("name", "cifar10")),
        partition=str(ds_data.get("partition", "dirichlet")),
        alpha=float(ds_data.get("alpha", 0.1)),
        batch_size=int(ds_data.get("batch_size", 32)),
        data_dir=str(ds_data.get("data_dir", "./data")),
        val_ratio=float(ds_data.get("val_ratio", 0.2)),
        seed=int(ds_data.get("seed", raw.get("seed", 42))),
    )

    # 2. Model Config
    model_data = raw.get("model", {})
    model_cfg = ModelConfig(
        name=str(model_data.get("name", "simple_cnn")),
        seed=int(model_data.get("seed", raw.get("seed", 42))),
    )

    # 3. Training & Optimizer Config
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

    # 4. Baselines Config
    bl_data = raw.get("baselines", {})
    fedprox_data = bl_data.get("fedprox", {})
    ditto_data = bl_data.get("ditto", {})
    perfedavg_data = bl_data.get("perfedavg", {}) or bl_data.get("per_fedavg", {})
    pfedgraph_data = bl_data.get("pfedgraph", {})
    pdfl_data = bl_data.get("pdfl", {})

    bl_cfg = BaselinesConfig(
        fedavg=FedAvgConfig(),
        fedprox=FedProxConfig(mu=float(fedprox_data.get("mu", 0.01))),
        ditto=DittoConfig(
            lambda_penalty=float(ditto_data.get("lambda_penalty", ditto_data.get("lambda", 0.1))),
            personalized_epochs=int(ditto_data.get("personalized_epochs", 1)),
        ),
        perfedavg=PerFedAvgConfig(
            alpha_lr=float(perfedavg_data.get("alpha_lr", perfedavg_data.get("alpha", 0.01))),
            beta_lr=float(perfedavg_data.get("beta_lr", perfedavg_data.get("beta", 0.005))),
        ),
        pfedgraph=PFedGraphConfig(
            alpha=float(pfedgraph_data.get("alpha", 0.1)),
            lam=float(pfedgraph_data.get("lam", pfedgraph_data.get("lambda", 0.01))),
        ),
        scaffold=ScaffoldConfig(),
        pdfl=PdflConfig(
            similarity_threshold=float(
                pdfl_data.get("similarity_threshold", 0.5)
            )
        ),
    )

    algo_val = str(raw.get("algorithm") or raw.get("baseline", "fedavg")).lower().strip()

    # 5. Top-level Unified Config
    return UnifiedConfig(
        experiment_name=str(raw.get("experiment_name", "cifar10_federated")),
        baseline=algo_val,
        algorithm=algo_val,
        num_clients=int(raw.get("num_clients", 3)),
        rounds=int(raw.get("rounds", 15)),
        epochs=int(raw.get("epochs", raw.get("epochs_per_round", 2))),
        device=str(raw.get("device", "cuda")),
        seed=int(raw.get("seed", 42)),
        server_host=str(raw.get("server_host", "127.0.0.1")),
        server_port=int(raw.get("server_port", 50051)),
        dataset=ds_cfg,
        model=model_cfg,
        training=tr_cfg,
        baselines=bl_cfg,
    )
