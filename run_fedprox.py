#!/usr/bin/env python3
"""FedProx Experiment Runner for Federated Learning Lab.

Executes federated learning training simulations using the FedProx algorithm
(Li et al., 2020) with proximal regularization parameter mu.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fl_data_generator import FederatedDataset
from src.baselines import get_baseline
from src.client import create_client_node
from src.datasets.transforms import get_default_transform
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.config import load_unified_config
from src.utils.evaluation import evaluate_model, get_global_test_loader, parse_client_metrics
from src.utils.serialization import model_to_bytes

logger = logging.getLogger("fedprox")


def run_fedprox_simulation(
    num_clients: int = 3,
    rounds: int = 2,
    epochs: int = 1,
    mu: float = 0.01,
    config_path: str = "configs/config.yaml",
    server_port: int = 0,
    synthetic: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Runs a complete federated training simulation using FedProx.

    Args:
        num_clients: Total number of participating clients.
        rounds: Number of federated training rounds.
        epochs: Local training epochs per client per round.
        mu: Proximal penalty parameter.
        config_path: Path to configuration YAML file.
        server_port: Port for coordinator gRPC server (0 for ephemeral OS port).
        synthetic: If True, uses synthetic tensors.
        device: Target compute device ('cuda' or 'cpu').

    Returns:
        Summary dict containing round history, metrics, and weights.
    """
    logger.info("=" * 65)
    logger.info("Starting FedProx Simulation (mu=%.4f, clients=%d, rounds=%d)", mu, num_clients, rounds)
    logger.info("=" * 65)

    u_cfg = load_unified_config(config_path)
    device_name = device or (u_cfg.device if torch.cuda.is_available() else "cpu")
    target_device = torch.device(device_name)
    seed = u_cfg.seed or 42

    # 1. Setup Server & FedProx Strategy
    strategy_cls, _ = get_baseline("fedprox")
    strategy = strategy_cls()
    global_model = SimpleCNN(seed=seed).to(target_device)
    initial_weights = model_to_bytes(global_model)

    server = FederatedServer(
        host="127.0.0.1", port=server_port, model=global_model, strategy=strategy
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info("FederatedServer started at %s with strategy %s", server_addr, strategy.__class__.__name__)

    # 2. Base Client Configuration
    base_cfg = u_cfg.to_client_config(client_id=0)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients
    base_cfg.algorithm = "fedprox"
    base_cfg.fedprox.mu = float(mu)
    base_cfg.training.device = device_name

    # 3. Data Preparation
    fed_dataset = None
    synthetic_loaders = []
    if synthetic:
        logger.info("Using synthetic data loaders...")
        for _ in range(num_clients):
            inputs = torch.randn(32, 3, 32, 32)
            targets = torch.randint(0, 10, (32,))
            synthetic_loaders.append(
                DataLoader(TensorDataset(inputs, targets), batch_size=base_cfg.dataset.batch_size)
            )
        global_test_loader = None
    else:
        logger.info(
            "Loading '%s' partition (%s, alpha=%.2f) via fl-data-generator...",
            base_cfg.dataset.name,
            base_cfg.dataset.partition,
            base_cfg.dataset.alpha,
        )
        transform = get_default_transform(base_cfg.dataset.name)
        fed_dataset = FederatedDataset(
            dataset=base_cfg.dataset.name,
            data_dir=base_cfg.dataset.data_dir,
            num_clients=num_clients,
            partition=base_cfg.dataset.partition,
            alpha=base_cfg.dataset.alpha,
            seed=base_cfg.dataset.seed,
            transform=transform,
        )
        global_test_loader = get_global_test_loader(
            dataset_name=base_cfg.dataset.name,
            data_dir=base_cfg.dataset.data_dir,
            batch_size=128,
            download=False,
        )

    # 4. Instantiate Clients
    nodes = []
    try:
        for i in range(num_clients):
            node_id = f"client_{i}"
            if synthetic:
                node = create_client_node(
                    base_cfg,
                    load_dataset=False,
                    client_id=i,
                    num_clients=num_clients,
                    node_id=node_id,
                )
                node.data = synthetic_loaders[i]
            else:
                node = create_client_node(
                    base_cfg,
                    load_dataset=True,
                    client_id=i,
                    num_clients=num_clients,
                    node_id=node_id,
                    fed_dataset=fed_dataset,
                )
            reg = node.register()
            if not reg or not reg.success:
                raise RuntimeError(f"Client {node_id} registration failed")
            nodes.append(node)

        logger.info("All %d client nodes initialized with FedProxTrainer (mu=%.4f).", len(nodes), mu)

        # 5. Execute Federated Rounds
        round_history = []
        for r in range(1, rounds + 1):
            t_start = time.time()
            responses = server.trigger_round(round_id=r, epochs=epochs, aggregate=True)
            elapsed = time.time() - t_start

            successful = [res for res in responses if res.success]
            client_losses = []
            client_accs = []
            for res in successful:
                parsed = parse_client_metrics(res.message)
                if "loss" in parsed:
                    client_losses.append(parsed["loss"])
                if "test_acc" in parsed:
                    client_accs.append(parsed["test_acc"])

            avg_client_loss = (sum(client_losses) / len(client_losses)) if client_losses else 0.0
            mean_local_acc = (sum(client_accs) / len(client_accs)) if client_accs else 0.0

            global_acc = 0.0
            if global_test_loader is not None and server.global_model is not None:
                eval_metrics = evaluate_model(
                    model=server.global_model,
                    dataloader=global_test_loader,
                    device=target_device,
                )
                global_acc = float(eval_metrics.get("accuracy", 0.0))

            logger.info(
                "Round %d/%d completed in %.2fs: Mean Loss: %.4f | Global Acc: %.2f%% | Mean Local Acc: %.2f%% | Clients: %s",
                r,
                rounds,
                elapsed,
                avg_client_loss,
                global_acc * 100,
                mean_local_acc * 100,
                [round(a * 100, 2) for a in client_accs],
            )

            round_history.append({
                "round_id": r,
                "elapsed_sec": round(elapsed, 2),
                "num_responses": len(successful),
                "avg_client_loss": round(avg_client_loss, 4),
                "global_accuracy": round(global_acc * 100, 2),
                "mean_local_accuracy": round(mean_local_acc * 100, 2),
                "client_accuracies": [round(a * 100, 2) for a in client_accs],
            })

        final_weights = server.get_global_weights()
        logger.info("FedProx simulation completed successfully.")

        return {
            "algorithm": "fedprox",
            "mu": mu,
            "num_clients": num_clients,
            "rounds_completed": rounds,
            "round_history": round_history,
            "initial_model_weights": initial_weights,
            "final_model_weights": final_weights,
        }

    finally:
        for node in nodes:
            try:
                node.stop(grace=0)
                if node.transport is not None:
                    node.transport.close()
            except Exception:
                pass
        server.stop(grace=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - FedProx Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clients", "-c", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", "-r", type=int, default=2, help="Number of federated rounds")
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--mu", "-m", type=float, default=0.01, help="Proximal parameter mu")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [FEDPROX] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_fedprox_simulation(
        num_clients=args.clients,
        rounds=args.rounds,
        epochs=args.epochs,
        mu=args.mu,
        config_path=args.config,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
