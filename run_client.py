"""Standalone federated learning client runner.

Connects to the central coordinator server, registers with its assigned client ID,
loads its local dataset partition via fl-data-generator, listens for training requests,
executes local epochs on its private data, and transmits updated weights back over gRPC.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import yaml

from fl_data_generator import FederatedDataset
from src.client.config import (
    create_criterion,
    create_model,
    create_optimizer_factory,
    load_client_config,
)
from src.client.node import Node
from src.transport.client import FederatedClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Client Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-C", type=str, default="configs/client.yaml", help="Path to client YAML configuration")
    parser.add_argument("--id", "-i", type=int, default=0, help="Client partition index (0, 1, 2, ...)")
    parser.add_argument("--clients", "-c", type=int, default=None, help="Total participating clients (overrides config)")
    parser.add_argument("--server", "-s", type=str, default=None, help="Central coordinator server address (overrides config)")
    parser.add_argument("--port", "-p", type=int, default=None, help="Local listening port (0 for ephemeral)")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Dataset name: cifar10, mnist, fashion_mnist")
    parser.add_argument("--partition", type=str, default=None, help="Partition type: dirichlet, iid, pathological, quantity")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha parameter for label skew")
    parser.add_argument("--batch-size", "-b", type=int, default=None, help="Local training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Local learning rate")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline local training algorithm (default: fedavg)")
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

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{node_id.upper()}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(node_id)

    print("\n" + "=" * 65)
    print(f"       FEDERATED LEARNING LAB - CLIENT NODE: {node_id}")
    print(f"       Partition: {cfg.dataset.name} ({cfg.dataset.partition}, alpha={cfg.dataset.alpha}) | Client {args.id + 1}/{cfg.dataset.num_clients}")
    print(f"       Server Address: {cfg.server_address}")
    print("=" * 65 + "\n")

    logger.info(
        "Partitioning %s across %d clients with %s (alpha=%.2f) via fl-data-generator...",
        cfg.dataset.name,
        cfg.dataset.num_clients,
        cfg.dataset.partition,
        cfg.dataset.alpha,
    )
    fed_dataset = FederatedDataset(
        dataset=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        num_clients=cfg.dataset.num_clients,
        partition=cfg.dataset.partition,
        alpha=cfg.dataset.alpha,
        seed=cfg.dataset.seed,
    )
    dataloader = fed_dataset.get_client_dataloader(client_id=args.id, batch_size=cfg.dataset.batch_size, shuffle=True)
    num_samples = len(fed_dataset.get_client_indices(args.id))
    logger.info("Local partition ready: %d samples assigned to this client.", num_samples)

    model = create_model(cfg.model)
    criterion = create_criterion(cfg.training.loss)
    optimizer_fn = create_optimizer_factory(cfg.training.optimizer, lr=cfg.training.learning_rate)

    logger.info("Connecting to central coordinator at %s...", cfg.server_address)
    transport = FederatedClient(target=cfg.server_address)

    baseline_name = args.baseline
    if not baseline_name:
        cfg_p = Path(args.config)
        if cfg_p.is_file():
            with open(cfg_p, "r", encoding="utf-8") as f:
                raw_yaml = yaml.safe_load(f) or {}
                baseline_name = raw_yaml.get("baseline", "fedavg")
        else:
            baseline_name = "fedavg"
    baseline_name = str(baseline_name).lower()

    from src.baselines import get_baseline
    _, trainer_cls = get_baseline(baseline_name)
    trainer = trainer_cls()
    logger.info("Configured baseline client trainer: %s (%s)", baseline_name, trainer_cls.__name__)

    node = Node(
        node_id=node_id,
        model=model,
        data=dataloader,
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        transport=transport,
        port=cfg.client_port,
        trainer=trainer,
    )
    logger.info("Listening for server RPCs on %s", node.address)

    reg_response = node.register()
    if not reg_response or not reg_response.success:
        logger.error("Failed to register with server: %s", reg_response)
        sys.exit(1)
    logger.info("Registration confirmed by server: '%s'", reg_response.message)
    logger.info("Ready and waiting for server to trigger federated training rounds...")

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
