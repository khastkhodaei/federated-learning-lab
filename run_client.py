"""Standalone federated learning client runner.

Connects to the central coordinator server, registers with its assigned client ID,
loads its local dataset partition via fl-data-generator, listens for training requests,
executes local epochs on its private data, and transmits updated weights back over gRPC.
"""

import argparse
import logging
import sys

import torch
from fl_data_generator import FederatedDataset
from torch.utils.data import DataLoader

from src.client.config import (
    ClientConfig,
    create_criterion,
    create_model,
    create_optimizer_factory,
    create_trainer_from_config,
    load_client_config,
)
from src.client.node import Node
from src.datasets.transforms import get_default_transform
from src.transport.client import FederatedClient
from src.utils.hardware import get_hardware_summary, resolve_device


def build_client_dataloaders(
    cfg: ClientConfig, client_id: int
) -> tuple[DataLoader, DataLoader | None]:
    """Builds training and optional validation DataLoader for a client partition."""
    transform = get_default_transform(cfg.dataset.name)
    fed_dataset = FederatedDataset(
        dataset=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        num_clients=cfg.dataset.num_clients,
        partition=cfg.dataset.partition,
        alpha=cfg.dataset.alpha,
        seed=cfg.dataset.seed,
        transform=transform,
    )
    dataloader = fed_dataset.get_client_dataloader(
        client_id=client_id, batch_size=cfg.dataset.batch_size, shuffle=True
    )
    test_loader = None
    if cfg.dataset.val_ratio > 0.0 and hasattr(
        fed_dataset, "get_client_dataset"
    ):
        client_ds = fed_dataset.get_client_dataset(client_id)
        if len(client_ds) > 1:
            val_len = max(1, int(len(client_ds) * cfg.dataset.val_ratio))
            train_len = len(client_ds) - val_len
            train_ds, val_ds = torch.utils.data.random_split(
                client_ds,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(
                    cfg.dataset.seed or 42
                ),
            )
            dataloader = DataLoader(
                train_ds, batch_size=cfg.dataset.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                val_ds, batch_size=cfg.dataset.batch_size, shuffle=False
            )
    return dataloader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Client Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        default="configs/config.yaml",
        help="Path to unified YAML configuration",
    )
    parser.add_argument(
        "--id",
        "-i",
        type=int,
        default=0,
        help="Client partition index (0, 1, 2, ...)",
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=None,
        help="Total participating clients (overrides config)",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        default=None,
        help="Central coordinator server address (overrides config)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Local listening port (0 for ephemeral)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset name: cifar10, mnist, fashion_mnist",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Partition type: dirichlet, iid, pathological, quantity",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet alpha parameter for label skew",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Local training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Local learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for local training (cuda, cpu)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline local training algorithm (default: fedavg)",
    )
    args = parser.parse_args()

    node_id = f"client_{args.id}"

    # Load configuration
    cfg = load_client_config(
        args.config,
        node_id=node_id,
        client_id=args.id,
        num_clients=args.clients,
    )

    # CLI Overrides
    if args.server:
        cfg.server_address = args.server
    if args.port is not None:
        cfg.client_port = args.port
    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.partition:
        cfg.dataset.partition = args.partition
    if args.alpha is not None:
        cfg.dataset.alpha = args.alpha
    if args.batch_size is not None:
        cfg.dataset.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.device is not None:
        cfg.training.device = args.device
    if args.baseline:
        cfg.algorithm = args.baseline.lower().strip()

    # Resolve device with shared helper before Node and get_hardware_summary
    cfg.training.device = resolve_device(cfg.training.device)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{node_id.upper()}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(node_id)

    hw_info = get_hardware_summary(cfg.training.device)

    print("\n" + "=" * 65)
    print(f"       FEDERATED LEARNING LAB - CLIENT NODE: {node_id}")
    print(f"       Baseline: {cfg.algorithm.upper()} | Hardware: {hw_info}")
    print(
        f"       Partition: {cfg.dataset.name} ({cfg.dataset.partition}, alpha={cfg.dataset.alpha}) | Client {args.id + 1}/{cfg.dataset.num_clients}"
    )
    print(f"       Server Address: {cfg.server_address}")
    print("=" * 65 + "\n")

    logger.info(
        "Partitioning %s across %d clients with %s (alpha=%.2f) via fl-data-generator...",
        cfg.dataset.name,
        cfg.dataset.num_clients,
        cfg.dataset.partition,
        cfg.dataset.alpha,
    )
    dataloader, test_loader = build_client_dataloaders(cfg, args.id)

    model = create_model(cfg.model)
    criterion = create_criterion(cfg.training.loss)
    optimizer_fn = create_optimizer_factory(
        cfg.training.optimizer, lr=cfg.training.learning_rate
    )

    logger.info(
        "Connecting to central coordinator at %s...", cfg.server_address
    )
    transport = FederatedClient(target=cfg.server_address)

    trainer = create_trainer_from_config(cfg)
    logger.info(
        "Configured baseline client trainer: %s (%s)",
        cfg.algorithm,
        type(trainer).__name__,
    )

    node = Node(
        node_id=node_id,
        model=model,
        data=dataloader,
        test_data=test_loader,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        transport=transport,
        port=cfg.client_port,
        trainer=trainer,
        device=cfg.training.device,
        lr=cfg.training.learning_rate,
    )
    logger.info("Listening for server RPCs on %s", node.address)

    reg_response = node.register()
    if not reg_response or not reg_response.success:
        logger.error("Failed to register with server: %s", reg_response)
        sys.exit(1)
    logger.info(
        "Registration confirmed by server: '%s'", reg_response.message
    )
    logger.info(
        "Ready and waiting for server to trigger federated training rounds..."
    )

    try:
        if node.server is not None:
            node.server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping client node...")
    finally:
        node.stop(grace=1)
        transport.close()
        logger.info("Client node shutdown complete.")


if __name__ == "__main__":
    main()
