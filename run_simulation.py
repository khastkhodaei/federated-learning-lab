#!/usr/bin/env python3
"""Federated Learning Lab: Unified Multi-Client Simulation Orchestrator.

Consolidates all baseline execution logic (FedAvg, FedProx, Ditto, Per-FedAvg, pFedGraph)
into a single modular orchestrator. Spawns the central coordinator server and N client nodes,
coordinates synchronous rounds via gRPC, evaluates global and personalized models, and logs telemetry.
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
from src.baselines import create_server_strategy
from src.client import create_client_node
from src.datasets.transforms import get_default_transform
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.config import load_unified_config
from src.utils.evaluation import evaluate_model, get_global_test_loader, parse_client_metrics
from src.utils.serialization import model_to_bytes

logger = logging.getLogger("simulation")


def run_federated_simulation(
    num_clients: int = 3,
    rounds: int = 2,
    epochs: int = 1,
    algorithm: str = "fedavg",
    config_path: str = "configs/config.yaml",
    server_port: int = 0,
    synthetic: bool = False,
    device: str | None = None,
    # FedProx
    mu: float | None = None,
    # Ditto
    lambda_penalty: float | None = None,
    personalized_epochs: int | None = None,
    # Per-FedAvg
    alpha_lr: float | None = None,
    beta_lr: float | None = None,
    # pFedGraph
    graph_alpha: float | None = None,
    graph_lam: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Runs a complete federated training simulation across any supported baseline.

    Args:
        num_clients: Total number of participating clients.
        rounds: Number of federated training rounds.
        epochs: Local training epochs per client per round.
        algorithm: Selected algorithm ('fedavg', 'fedprox', 'ditto', 'perfedavg', 'pfedgraph').
        config_path: Path to unified YAML configuration file.
        server_port: Port for coordinator gRPC server (0 for ephemeral OS port).
        synthetic: If True, uses fast synthetic tensors instead of real dataset.
        device: Target compute device ('cuda' or 'cpu').
        mu: FedProx proximal regularization parameter.
        lambda_penalty: Ditto personalized model proximal penalty.
        personalized_epochs: Ditto personalized model local epochs.
        alpha_lr: Per-FedAvg inner-loop adaptation learning rate.
        beta_lr: Per-FedAvg outer-loop meta-update learning rate.
        graph_alpha: pFedGraph graph learning rate / step size.
        graph_lam: pFedGraph graph sparsity / regularization parameter.

    Returns:
        Dictionary summarizing simulation results, metrics, and round history.
    """
    u_cfg = load_unified_config(config_path)
    effective_algo = (algorithm or u_cfg.algorithm or u_cfg.baseline or "fedavg").lower().strip()
    device_name = device or (u_cfg.device if torch.cuda.is_available() else "cpu")
    target_device = torch.device(device_name)
    seed = u_cfg.seed or 42

    logger.info("=" * 70)
    logger.info(
        "Starting %s Simulation (clients=%d, rounds=%d, epochs=%d, device=%s)",
        effective_algo.upper(),
        num_clients,
        rounds,
        epochs,
        device_name,
    )
    logger.info("=" * 70)

    # 1. Instantiate Strategy
    strategy = create_server_strategy(
        effective_algo,
        config=u_cfg,
        graph_alpha=graph_alpha,
        graph_lam=graph_lam,
    )

    # 2. Setup Central Coordinator Server
    global_model = SimpleCNN(seed=seed).to(target_device)
    initial_weights = model_to_bytes(global_model)

    server = FederatedServer(
        host="127.0.0.1", port=server_port, model=global_model, strategy=strategy
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info("FederatedServer started at %s with strategy %s", server_addr, strategy.__class__.__name__)

    # 3. Base Client Configuration with Overrides
    base_cfg = u_cfg.to_client_config(client_id=0)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients
    base_cfg.algorithm = effective_algo
    base_cfg.training.device = device_name

    # Apply algorithm-specific hyperparameter overrides
    if effective_algo in ("fedprox", "prox"):
        eff_mu = mu if mu is not None else getattr(u_cfg.baselines.fedprox, "mu", 0.01)
        base_cfg.fedprox.mu = float(eff_mu)
    elif effective_algo in ("ditto",):
        eff_lambda = lambda_penalty if lambda_penalty is not None else getattr(u_cfg.baselines.ditto, "lambda_penalty", 0.1)
        eff_pe = personalized_epochs if personalized_epochs is not None else getattr(u_cfg.baselines.ditto, "personalized_epochs", 1)
        base_cfg.ditto.lambda_penalty = float(eff_lambda)
        base_cfg.ditto.personalized_epochs = int(eff_pe)
    elif effective_algo in ("perfedavg", "per_fedavg", "fomaml"):
        eff_alpha_lr = alpha_lr if alpha_lr is not None else getattr(u_cfg.baselines.perfedavg, "alpha_lr", 0.01)
        eff_beta_lr = beta_lr if beta_lr is not None else getattr(u_cfg.baselines.perfedavg, "beta_lr", 0.005)
        base_cfg.perfedavg.alpha_lr = float(eff_alpha_lr)
        base_cfg.perfedavg.beta_lr = float(eff_beta_lr)
    elif effective_algo in ("pfedgraph", "graph"):
        eff_graph_alpha = graph_alpha if graph_alpha is not None else getattr(u_cfg.baselines.pfedgraph, "alpha", 0.1)
        eff_graph_lam = graph_lam if graph_lam is not None else getattr(u_cfg.baselines.pfedgraph, "lam", 0.01)
        base_cfg.pfedgraph.alpha = float(eff_graph_alpha)
        base_cfg.pfedgraph.lam = float(eff_graph_lam)

    # 4. Data Preparation
    fed_dataset = None
    synthetic_loaders = []
    if synthetic:
        logger.info("Using synthetic data partition for %d clients...", num_clients)
        for _ in range(num_clients):
            inputs = torch.randn(32, 3, 32, 32)
            targets = torch.randint(0, 10, (32,))
            synthetic_loaders.append(
                DataLoader(TensorDataset(inputs, targets), batch_size=base_cfg.dataset.batch_size)
            )
        global_test_loader = None
    else:
        logger.info(
            "Loading '%s' partition (%s, alpha=%.2f) across %d clients via fl-data-generator...",
            base_cfg.dataset.name,
            base_cfg.dataset.partition,
            base_cfg.dataset.alpha,
            num_clients,
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

    # 5. Initialize and Register Client Nodes
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
                node.test_data = synthetic_loaders[i]
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
                raise RuntimeError(f"Client {node_id} registration failed: {reg}")
            nodes.append(node)

        logger.info("All %d client nodes successfully registered.", len(nodes))

        # 6. Execute Federated Rounds
        round_history = []
        for r in range(1, rounds + 1):
            t_start = time.time()
            responses = server.trigger_round(round_id=r, epochs=epochs, aggregate=True)
            elapsed = time.time() - t_start

            successful = [res for res in responses if res.success]
            total_samples = sum(res.num_samples for res in successful)

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

            acc_label = "Personalized Local Acc" if effective_algo in ("ditto", "perfedavg", "pfedgraph") else "Local Acc"
            logger.info(
                "Round %d/%d completed in %.2fs: Global Acc: %.2f%% | Mean %s: %.2f%% | Client Accs: %s",
                r,
                rounds,
                elapsed,
                global_acc * 100,
                acc_label,
                mean_local_acc * 100,
                [round(a * 100, 2) for a in client_accs],
            )

            round_record: dict[str, Any] = {
                "round_id": r,
                "elapsed_sec": round(elapsed, 2),
                "num_responses": len(successful),
                "total_samples": total_samples,
                "avg_client_loss": round(avg_client_loss, 4),
                "global_accuracy": round(global_acc * 100, 2),
                "mean_local_accuracy": round(mean_local_acc * 100, 2),
                "client_accuracies": [round(a * 100, 2) for a in client_accs],
            }

            # If pFedGraph, record adjacency matrix
            if effective_algo == "pfedgraph" and hasattr(strategy, "adjacency_matrix") and strategy.adjacency_matrix is not None:
                round_record["adjacency_matrix"] = strategy.adjacency_matrix.cpu().numpy().tolist()

            round_history.append(round_record)

        final_weights = server.get_global_weights()
        logger.info("%s federated simulation completed successfully across %d rounds.", effective_algo.upper(), rounds)

        # Build standardized reporting history and save baseline artifacts
        reporting_history = []
        for r_rec in round_history:
            c_accs = [a / 100.0 for a in r_rec.get("client_accuracies", [])]
            rep_item = {
                "round": r_rec.get("round_id", 1),
                "duration_seconds": r_rec.get("elapsed_sec", 0.0),
                "total_samples": r_rec.get("total_samples", 0),
                "weighted_loss": r_rec.get("avg_client_loss", 0.0),
                "global_test_accuracy": r_rec.get("global_accuracy", 0.0) / 100.0,
                "mean_local_test_accuracy": r_rec.get("mean_local_accuracy", 0.0) / 100.0,
                "std_local_test_accuracy": float(np.std(c_accs)) if c_accs else 0.0,
                "worst_10_local_accuracy": float(np.percentile(c_accs, 10)) if c_accs else (r_rec.get("mean_local_accuracy", 0.0) / 100.0),
                "client_metrics": {
                    f"client_{idx}": {"test_accuracy": ca} for idx, ca in enumerate(c_accs)
                },
            }
            if "adjacency_matrix" in r_rec:
                rep_item["adjacency_matrix"] = r_rec["adjacency_matrix"]
            reporting_history.append(rep_item)

        try:
            from src.utils.reporting import save_baseline_report
            artifacts = save_baseline_report(
                baseline_name=effective_algo,
                round_history=reporting_history,
                output_dir="reports",
            )
        except Exception as rep_err:
            logger.warning("Could not save baseline report: %s", rep_err)
            artifacts = {}

        result: dict[str, Any] = {
            "algorithm": effective_algo,
            "num_clients": num_clients,
            "rounds_completed": rounds,
            "round_history": round_history,
            "initial_model_weights": initial_weights,
            "final_model_weights": final_weights,
            "artifacts": artifacts,
        }

        if effective_algo == "pfedgraph" and hasattr(strategy, "adjacency_matrix") and strategy.adjacency_matrix is not None:
            w_np = strategy.adjacency_matrix.cpu().numpy()
            result["final_adjacency_matrix"] = w_np.tolist()
            print("\n" + "=" * 50)
            print("FINAL LEARNED GRAPH ADJACENCY MATRIX W (pFedGraph):")
            print("=" * 50)
            print(np.round(w_np, 4))
            print("=" * 50 + "\n")

        return result

    finally:
        for node in nodes:
            try:
                node.stop(grace=0)
                if node.transport is not None:
                    node.transport.close()
            except Exception:
                pass
        server.stop(grace=0)


# Backward-compatible convenience aliases
run_simulation = run_federated_simulation
run_fedprox_simulation = lambda **kw: run_federated_simulation(algorithm="fedprox", **kw)
run_ditto_simulation = lambda **kw: run_federated_simulation(algorithm="ditto", **kw)
run_perfedavg_simulation = lambda **kw: run_federated_simulation(algorithm="perfedavg", **kw)
run_pfedgraph_simulation = lambda **kw: run_federated_simulation(algorithm="pfedgraph", **kw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Unified Simulation Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core experiment parameters
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph"],
        help="Federated optimization algorithm baseline",
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=3,
        help="Number of client nodes in the simulation",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=2,
        help="Number of federated aggregation rounds",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="Number of local client training epochs per round",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to unified YAML configuration file",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=0,
        help="Coordinator gRPC server port (0 for ephemeral OS-assigned port)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run using fast synthetic tensors (bypasses data downloading)",
    )

    # Algorithm-specific hyperparameters
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="[FedProx] Proximal regularization penalty parameter mu (default: 0.01)",
    )
    parser.add_argument(
        "--lambda-penalty",
        "-l",
        dest="lambda_penalty",
        type=float,
        default=None,
        help="[Ditto] Personalized model proximal penalty parameter lambda (default: 0.1)",
    )
    parser.add_argument(
        "--pers-epochs",
        dest="personalized_epochs",
        type=int,
        default=None,
        help="[Ditto] Personalized model local training epochs (default: 1)",
    )
    parser.add_argument(
        "--alpha-lr",
        dest="alpha_lr",
        type=float,
        default=None,
        help="[Per-FedAvg] Inner-loop adaptation learning rate alpha (default: 0.01)",
    )
    parser.add_argument(
        "--beta-lr",
        dest="beta_lr",
        type=float,
        default=None,
        help="[Per-FedAvg] Outer-loop meta-update learning rate beta (default: 0.005)",
    )
    parser.add_argument(
        "--graph-alpha",
        dest="graph_alpha",
        type=float,
        default=None,
        help="[pFedGraph] Graph learning rate / step size alpha (default: 0.1)",
    )
    parser.add_argument(
        "--graph-lam",
        dest="graph_lam",
        type=float,
        default=None,
        help="[pFedGraph] Graph sparsity / regularization parameter lambda (default: 0.01)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_federated_simulation(
        num_clients=args.clients,
        rounds=args.rounds,
        epochs=args.epochs,
        algorithm=args.algorithm,
        config_path=args.config,
        server_port=args.port,
        synthetic=args.synthetic,
        device=args.device,
        mu=args.mu,
        lambda_penalty=args.lambda_penalty,
        personalized_epochs=args.personalized_epochs,
        alpha_lr=args.alpha_lr,
        beta_lr=args.beta_lr,
        graph_alpha=args.graph_alpha,
        graph_lam=args.graph_lam,
    )


if __name__ == "__main__":
    main()
