#!/usr/bin/env python3
"""pFedGraph Personalized Federated Learning Experiment Runner.

Executes federated learning simulations using the pFedGraph algorithm
(Chen et al., 'pFedGraph: Personalized Federated Learning via Graph').
The server learns a pairwise client collaboration graph (adjacency matrix W)
and aggregates personalized models for each client based on graph weights.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import numpy as np
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

logger = logging.getLogger("pfedgraph")


def run_pfedgraph_simulation(
    num_clients: int = 3,
    rounds: int = 2,
    epochs: int = 1,
    alpha: float = 0.1,
    lam: float = 0.01,
    config_path: str = "configs/config.yaml",
    server_port: int = 0,
    synthetic: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Runs a complete federated training simulation using pFedGraph.

    Args:
        num_clients: Total number of participating clients.
        rounds: Number of federated training rounds.
        epochs: Local training epochs per round.
        alpha: Graph learning rate / interpolation parameter.
        lam: Sparsity / L2 regularization penalty parameter lambda.
        config_path: Path to configuration YAML file.
        server_port: Port for coordinator gRPC server (0 for ephemeral OS port).
        synthetic: If True, uses synthetic tensors.
        device: Target compute device ('cuda' or 'cpu').

    Returns:
        Summary dict containing round history, learned adjacency matrices, and weights.
    """
    logger.info("=" * 70)
    logger.info(
        "Starting pFedGraph Simulation (alpha=%.4f, lam=%.4f, clients=%d, rounds=%d)",
        alpha,
        lam,
        num_clients,
        rounds,
    )
    logger.info("=" * 70)

    u_cfg = load_unified_config(config_path)
    device_name = device or (u_cfg.device if torch.cuda.is_available() else "cpu")
    target_device = torch.device(device_name)
    seed = u_cfg.seed or 42

    # 1. Setup Server & pFedGraph Strategy
    strategy_cls, _ = get_baseline("pfedgraph")
    strategy = strategy_cls(alpha=alpha, lam=lam)
    global_model = SimpleCNN(seed=seed).to(target_device)
    initial_weights = model_to_bytes(global_model)

    server = FederatedServer(
        host="127.0.0.1", port=server_port, model=global_model, strategy=strategy
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info(
        "FederatedServer started at %s with strategy %s",
        server_addr,
        strategy.__class__.__name__,
    )

    # 2. Base Client Configuration
    base_cfg = u_cfg.to_client_config(client_id=0)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients
    base_cfg.algorithm = "pfedgraph"
    if hasattr(base_cfg, "pfedgraph"):
        base_cfg.pfedgraph.alpha = float(alpha)
        base_cfg.pfedgraph.lam = float(lam)
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
                DataLoader(
                    TensorDataset(inputs, targets),
                    batch_size=base_cfg.dataset.batch_size,
                )
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

    # 4. Instantiate Clients with pFedGraph Trainers
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

        logger.info(
            "All %d client nodes initialized with PFedGraphClientTrainer (alpha=%.4f, lam=%.4f).",
            len(nodes),
            alpha,
            lam,
        )

        # 5. Execute Federated Rounds
        round_history = []
        for r in range(1, rounds + 1):
            t_start = time.time()
            responses = server.trigger_round(round_id=r, epochs=epochs, aggregate=True)
            elapsed = time.time() - t_start

            successful = [res for res in responses if res.success]
            client_losses = []
            client_pers_accs = []
            for res in successful:
                parsed = parse_client_metrics(res.message)
                if "loss" in parsed:
                    client_losses.append(parsed["loss"])
                if "test_acc" in parsed:
                    client_pers_accs.append(parsed["test_acc"])

            avg_client_loss = (
                (sum(client_losses) / len(client_losses)) if client_losses else 0.0
            )
            mean_pers_acc = (
                (sum(client_pers_accs) / len(client_pers_accs))
                if client_pers_accs
                else 0.0
            )

            global_acc = 0.0
            if global_test_loader is not None and server.global_model is not None:
                eval_metrics = evaluate_model(
                    model=server.global_model,
                    dataloader=global_test_loader,
                    device=target_device,
                )
                global_acc = float(eval_metrics.get("accuracy", 0.0))

            # Retrieve current learned adjacency matrix
            w_np = (
                strategy.adjacency_matrix.cpu().numpy()
                if strategy.adjacency_matrix is not None
                else np.zeros((num_clients, num_clients))
            )

            logger.info(
                "Round %d/%d completed in %.2fs: Global Acc: %.2f%% | Mean Personalized Local Acc: %.2f%% | Client Accs: %s",
                r,
                rounds,
                elapsed,
                global_acc * 100,
                mean_pers_acc * 100,
                [round(a * 100, 2) for a in client_pers_accs],
            )
            logger.info("Round %d Learned Graph Adjacency Matrix W:\n%s", r, np.round(w_np, 4))

            round_history.append(
                {
                    "round_id": r,
                    "elapsed_sec": round(elapsed, 2),
                    "num_responses": len(successful),
                    "avg_client_loss": round(avg_client_loss, 4),
                    "global_accuracy": round(global_acc * 100, 2),
                    "mean_personalized_accuracy": round(mean_pers_acc * 100, 2),
                    "client_personalized_accuracies": [
                        round(a * 100, 2) for a in client_pers_accs
                    ],
                    "adjacency_matrix": w_np.tolist(),
                }
            )

        final_weights = server.get_global_weights()
        final_adjacency = (
            strategy.adjacency_matrix.cpu().numpy()
            if strategy.adjacency_matrix is not None
            else np.zeros((num_clients, num_clients))
        )

        logger.info("pFedGraph simulation completed successfully.")
        print("\n" + "=" * 50)
        print("FINAL LEARNED GRAPH ADJACENCY MATRIX W (3x3):")
        print("=" * 50)
        print(np.round(final_adjacency, 4))
        print("=" * 50 + "\n")

        return {
            "algorithm": "pfedgraph",
            "alpha": alpha,
            "lam": lam,
            "num_clients": num_clients,
            "rounds_completed": rounds,
            "round_history": round_history,
            "final_adjacency_matrix": final_adjacency.tolist(),
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
        description="Federated Learning Lab - pFedGraph Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clients", "-c", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", "-r", type=int, default=2, help="Number of federated rounds")
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Local training epochs")
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.1,
        help="Graph learning rate / step size alpha",
    )
    parser.add_argument(
        "--lam",
        "-l",
        type=float,
        default=0.01,
        help="Graph sparsity / regularization parameter lambda",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [PFEDGRAPH] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pfedgraph_simulation(
        num_clients=args.clients,
        rounds=args.rounds,
        epochs=args.epochs,
        alpha=args.alpha,
        lam=args.lam,
        config_path=args.config,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
