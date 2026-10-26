import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.transport.client import FederatedClient

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Hyperparameters for local model optimizer."""

    name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """Hyperparameters for client local training."""

    learning_rate: float = 0.01
    loss: str = "cross_entropy"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    device: str = "cpu"


@dataclass
class ModelConfig:
    """Specification for client model architecture."""

    name: str = "simple_cnn"
    seed: int | None = None


@dataclass
class DatasetConfig:
    """Partitioning and dataset specification for client private data."""

    name: str = "cifar10"
    batch_size: int = 32
    client_id: int = 0
    num_clients: int = 2
    partition: str = "dirichlet"
    alpha: float = 0.5
    data_dir: str = "./data"
    seed: int | None = 42
    val_ratio: float = 0.2


@dataclass
class FedProxClientConfig:
    """Hyperparameters for FedProx on the client."""

    mu: float = 0.01


@dataclass
class ClientConfig:
    """Complete configuration specification for an individual client node."""

    node_id: str = "client_1"
    server_address: str = "127.0.0.1:50051"
    client_host: str = "127.0.0.1"
    client_port: int | None = 0
    algorithm: str = "fedavg"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fedprox: FedProxClientConfig = field(default_factory=FedProxClientConfig)


def load_client_config(
    config_path: str | Path,
    node_id: str | None = None,
    client_id: int | None = None,
    num_clients: int | None = None,
) -> ClientConfig:
    """Loads a ClientConfig from a YAML file with optional runtime overrides.

    Args:
        config_path: Path to the YAML configuration file.
        node_id: Optional node ID override. If None and not in YAML, defaults to f"client_{client_id}".
        client_id: Optional client partition index override.
        num_clients: Optional total participating clients override.

    Returns:
        Populated ClientConfig instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Client config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    opt_data = data.get("training", {}).get("optimizer", {})
    optimizer_cfg = OptimizerConfig(
        name=opt_data.get("name", "sgd"),
        momentum=float(opt_data.get("momentum", 0.9)),
        weight_decay=float(opt_data.get("weight_decay", 0.0)),
    )

    training_data = data.get("training", {})
    training_cfg = TrainingConfig(
        learning_rate=float(training_data.get("learning_rate", 0.01)),
        loss=training_data.get("loss", "cross_entropy"),
        optimizer=optimizer_cfg,
        device=training_data.get("device", "cpu"),
    )

    model_data = data.get("model", {})
    model_cfg = ModelConfig(
        name=model_data.get("name", "simple_cnn"),
        seed=model_data.get("seed"),
    )

    dataset_data = data.get("dataset", {})
    effective_client_id = (
        client_id if client_id is not None else int(dataset_data.get("client_id", 0))
    )
    effective_num_clients = (
        num_clients if num_clients is not None else int(dataset_data.get("num_clients", 2))
    )

    dataset_cfg = DatasetConfig(
        name=dataset_data.get("name", "cifar10"),
        batch_size=int(dataset_data.get("batch_size", 32)),
        client_id=effective_client_id,
        num_clients=effective_num_clients,
        partition=dataset_data.get("partition", "dirichlet"),
        alpha=float(dataset_data.get("alpha", 0.5)),
        data_dir=dataset_data.get("data_dir", "./data"),
        seed=dataset_data.get("seed", 42),
        val_ratio=float(dataset_data.get("val_ratio", 0.2)),
    )

    if node_id is not None:
        effective_node_id = str(node_id)
    elif data.get("node_id") is not None:
        effective_node_id = str(data["node_id"])
    else:
        effective_node_id = f"client_{effective_client_id}"

    algo = str(data.get("algorithm") or data.get("baseline", "fedavg")).lower()
    fedprox_data = data.get("fedprox", {}) or data.get("baselines", {}).get("fedprox", {})
    fedprox_cfg = FedProxClientConfig(mu=float(fedprox_data.get("mu", 0.01)))

    return ClientConfig(
        node_id=effective_node_id,
        server_address=str(data.get("server_address", "127.0.0.1:50051")),
        client_host=str(data.get("client_host", "127.0.0.1")),
        client_port=data.get("client_port", 0),
        algorithm=algo,
        model=model_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
        fedprox=fedprox_cfg,
    )


def create_criterion(loss_name: str) -> nn.Module:
    """Instantiates a PyTorch loss function based on configuration name."""
    name = loss_name.lower().replace("_", "").replace("-", "")
    if name in ("crossentropy", "crossentropyloss", "ce"):
        return nn.CrossEntropyLoss()
    elif name in ("mse", "mseloss"):
        return nn.MSELoss()
    elif name in ("l1", "l1loss"):
        return nn.L1Loss()
    elif name in ("nll", "nllloss"):
        return nn.NLLLoss()
    else:
        logger.warning("Unrecognized loss '%s'; defaulting to CrossEntropyLoss", loss_name)
        return nn.CrossEntropyLoss()


def create_optimizer_factory(
    opt_cfg: OptimizerConfig, lr: float
) -> Callable[[Iterable[nn.Parameter]], Optimizer]:
    """Returns a callable factory producing an optimizer instance for model parameters."""
    opt_name = opt_cfg.name.lower()
    if opt_name == "sgd":
        return lambda params: torch.optim.SGD(
            params, lr=lr, momentum=opt_cfg.momentum, weight_decay=opt_cfg.weight_decay
        )
    elif opt_name == "adam":
        return lambda params: torch.optim.Adam(
            params, lr=lr, weight_decay=opt_cfg.weight_decay
        )
    elif opt_name == "adamw":
        return lambda params: torch.optim.AdamW(
            params, lr=lr, weight_decay=opt_cfg.weight_decay
        )
    else:
        logger.warning("Unrecognized optimizer '%s'; defaulting to SGD", opt_cfg.name)
        return lambda params: torch.optim.SGD(
            params, lr=lr, momentum=opt_cfg.momentum, weight_decay=opt_cfg.weight_decay
        )


def create_model(model_cfg: ModelConfig) -> nn.Module:
    """Instantiates model architecture according to configuration."""
    name = model_cfg.name.lower().replace("_", "").replace("-", "")
    if name in ("simplecnn", "cnn"):
        return SimpleCNN(seed=model_cfg.seed)
    else:
        logger.warning("Unrecognized model '%s'; defaulting to SimpleCNN", model_cfg.name)
        return SimpleCNN(seed=model_cfg.seed)


def create_trainer(
    algorithm: str = "fedavg",
    mu: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Factory creating a ClientTrainer instance according to the selected algorithm."""
    algo = algorithm.lower().strip()
    if algo in ("fedprox", "prox"):
        from src.baselines.fedprox.trainer import FedProxClientTrainer

        return FedProxClientTrainer(mu=mu)
    elif algo in ("fedavg", "avg"):
        from src.baselines.fedavg.trainer import FedAvgClientTrainer

        return FedAvgClientTrainer()
    else:
        logger.warning(
            "Unrecognized algorithm '%s'; defaulting to FedAvgClientTrainer", algorithm
        )
        from src.baselines.fedavg.trainer import FedAvgClientTrainer

        return FedAvgClientTrainer()


def create_client_node(
    config: ClientConfig | str | Path,
    load_dataset: bool = True,
    client_id: int | None = None,
    num_clients: int | None = None,
    node_id: str | None = None,
    fed_dataset: Any | None = None,
    trainer: Any | None = None,
) -> Node:
    """Instantiates and configures a Node from a ClientConfig or YAML file path.

    Args:
        config: ClientConfig instance or path to a YAML configuration file.
        load_dataset: If True, uses fl_data_generator to download/load the client partition.
        client_id: Optional client partition index override.
        num_clients: Optional total participating clients override.
        node_id: Optional node identifier override.
        fed_dataset: Optional pre-initialized FederatedDataset to avoid re-partitioning.

    Returns:
        Ready-to-run Node instance with train and local validation DataLoaders.
    """
    if isinstance(config, (str, Path)):
        config = load_client_config(
            config, node_id=node_id, client_id=client_id, num_clients=num_clients
        )
    elif client_id is not None:
        config.dataset.client_id = client_id
        if num_clients is not None:
            config.dataset.num_clients = num_clients
        if node_id is not None:
            config.node_id = node_id

    # 1. Model
    model = create_model(config.model)

    # 2. Loss Criterion & Optimizer
    criterion = create_criterion(config.training.loss)
    optimizer_fn = create_optimizer_factory(
        config.training.optimizer, config.training.learning_rate
    )

    # 3. Dataset & Local Train/Validation Split
    data_loader = None
    test_loader = None
    if fed_dataset is not None:
        if config.dataset.val_ratio > 0.0 and hasattr(fed_dataset, "get_client_dataset"):
            client_ds = fed_dataset.get_client_dataset(config.dataset.client_id)
            if len(client_ds) > 1:
                val_len = max(1, int(len(client_ds) * config.dataset.val_ratio))
                train_len = len(client_ds) - val_len
                train_ds, val_ds = torch.utils.data.random_split(
                    client_ds,
                    [train_len, val_len],
                    generator=torch.Generator().manual_seed(config.dataset.seed or 42),
                )
                data_loader = DataLoader(
                    train_ds, batch_size=config.dataset.batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    val_ds, batch_size=config.dataset.batch_size, shuffle=False
                )
            else:
                data_loader = fed_dataset.get_client_dataloader(
                    client_id=config.dataset.client_id,
                    batch_size=config.dataset.batch_size,
                    shuffle=True,
                )
        else:
            data_loader = fed_dataset.get_client_dataloader(
                client_id=config.dataset.client_id,
                batch_size=config.dataset.batch_size,
                shuffle=True,
            )
    elif load_dataset:
        try:
            from fl_data_generator import FederatedDataset

            from src.datasets.transforms import get_default_transform

            transform = get_default_transform(config.dataset.name)
            fed_ds = FederatedDataset(
                dataset=config.dataset.name,
                data_dir=config.dataset.data_dir,
                num_clients=config.dataset.num_clients,
                partition=config.dataset.partition,
                alpha=config.dataset.alpha,
                seed=config.dataset.seed,
                transform=transform,
            )
            if config.dataset.val_ratio > 0.0 and hasattr(fed_ds, "get_client_dataset"):
                client_ds = fed_ds.get_client_dataset(config.dataset.client_id)
                if len(client_ds) > 1:
                    val_len = max(1, int(len(client_ds) * config.dataset.val_ratio))
                    train_len = len(client_ds) - val_len
                    train_ds, val_ds = torch.utils.data.random_split(
                        client_ds,
                        [train_len, val_len],
                        generator=torch.Generator().manual_seed(config.dataset.seed or 42),
                    )
                    data_loader = DataLoader(
                        train_ds, batch_size=config.dataset.batch_size, shuffle=True
                    )
                    test_loader = DataLoader(
                        val_ds, batch_size=config.dataset.batch_size, shuffle=False
                    )
                else:
                    data_loader = fed_ds.get_client_dataloader(
                        client_id=config.dataset.client_id,
                        batch_size=config.dataset.batch_size,
                        shuffle=True,
                    )
            else:
                data_loader = fed_ds.get_client_dataloader(
                    client_id=config.dataset.client_id,
                    batch_size=config.dataset.batch_size,
                    shuffle=True,
                )
            logger.info(
                "Loaded %s dataset partition for client %d (%d train batches, %s test batches)",
                config.dataset.name,
                config.dataset.client_id,
                len(data_loader),
                len(test_loader) if test_loader is not None else 0,
            )
        except Exception as e:
            logger.error("Failed to load dataset with fl_data_generator: %s", e)
            raise

    # 4. Trainer Factory
    trainer_instance = trainer
    if trainer_instance is None:
        algorithm_name = getattr(config, "algorithm", "fedavg")
        fedprox_cfg = getattr(config, "fedprox", None)
        mu_val = getattr(fedprox_cfg, "mu", 0.01) if fedprox_cfg is not None else 0.01
        trainer_instance = create_trainer(algorithm=algorithm_name, mu=mu_val)

    # 5. Transport
    transport = None
    if config.server_address:
        transport = FederatedClient(target=config.server_address)

    # 6. Build and return Node
    return Node(
        node_id=config.node_id,
        data=data_loader,
        test_data=test_loader,
        model=model,
        transport=transport,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        lr=config.training.learning_rate,
        device=config.training.device,
        host=config.client_host,
        port=config.client_port,
        trainer=trainer_instance,
    )
