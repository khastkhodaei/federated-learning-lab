from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union
import yaml

import torch
import torch.nn as nn
from torch.optim import Optimizer

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
    seed: Optional[int] = None


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
    seed: Optional[int] = 42


@dataclass
class ClientConfig:
    """Complete configuration specification for an individual client node."""
    node_id: str = "client_1"
    server_address: str = "127.0.0.1:50051"
    client_host: str = "127.0.0.1"
    client_port: Optional[int] = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_client_config(config_path: Union[str, Path]) -> ClientConfig:
    """Loads a ClientConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

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
    dataset_cfg = DatasetConfig(
        name=dataset_data.get("name", "cifar10"),
        batch_size=int(dataset_data.get("batch_size", 32)),
        client_id=int(dataset_data.get("client_id", 0)),
        num_clients=int(dataset_data.get("num_clients", 2)),
        partition=dataset_data.get("partition", "dirichlet"),
        alpha=float(dataset_data.get("alpha", 0.5)),
        data_dir=dataset_data.get("data_dir", "./data"),
        seed=dataset_data.get("seed", 42),
    )

    return ClientConfig(
        node_id=str(data.get("node_id", "client_1")),
        server_address=str(data.get("server_address", "127.0.0.1:50051")),
        client_host=str(data.get("client_host", "127.0.0.1")),
        client_port=data.get("client_port", 0),
        model=model_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
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


def create_client_node(
    config: Union[ClientConfig, str, Path],
    load_dataset: bool = True,
) -> Node:
    """Instantiates and configures a Node from a ClientConfig or YAML file path.

    Args:
        config: ClientConfig instance or path to a YAML configuration file.
        load_dataset: If True, uses fl_data_generator to download/load the client partition.

    Returns:
        Ready-to-run Node instance.
    """
    if isinstance(config, (str, Path)):
        config = load_client_config(config)

    # 1. Model
    model = create_model(config.model)

    # 2. Loss Criterion & Optimizer
    criterion = create_criterion(config.training.loss)
    optimizer_fn = create_optimizer_factory(
        config.training.optimizer, config.training.learning_rate
    )

    # 3. Dataset
    data_loader = None
    if load_dataset:
        try:
            from fl_data_generator import FederatedDataset

            fed_dataset = FederatedDataset(
                dataset=config.dataset.name,
                data_dir=config.dataset.data_dir,
                num_clients=config.dataset.num_clients,
                partition=config.dataset.partition,
                alpha=config.dataset.alpha,
                seed=config.dataset.seed,
            )
            data_loader = fed_dataset.get_client_dataloader(
                client_id=config.dataset.client_id,
                batch_size=config.dataset.batch_size,
                shuffle=True,
            )
            logger.info(
                "Loaded %s dataset partition for client %d (%d batches)",
                config.dataset.name,
                config.dataset.client_id,
                len(data_loader),
            )
        except Exception as e:
            logger.error("Failed to load dataset with fl_data_generator: %s", e)
            raise

    # 4. Transport
    transport = None
    if config.server_address:
        transport = FederatedClient(target=config.server_address)

    # 5. Build and return Node
    return Node(
        node_id=config.node_id,
        data=data_loader,
        model=model,
        transport=transport,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        lr=config.training.learning_rate,
        device=config.training.device,
        host=config.client_host,
        port=config.client_port,
    )
