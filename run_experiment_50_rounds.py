#!/usr/bin/env python3
"""50-Round Federated Learning Experiment Runner (FedAvg on CIFAR-10, Dirichlet alpha=0.1).

Runs an extended 50-round federated training benchmark, evaluates global and local
metrics at every round, saves checkpoints, and produces structured report artifacts.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fl_data_generator import FederatedDataset
from src.client import create_client_node
from src.datasets.transforms import get_default_transform
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.config import load_unified_config
from src.utils.evaluation import (
    compute_fairness_metrics,
    evaluate_model,
    get_global_test_loader,
    parse_client_metrics,
)
from src.utils.reporting import save_baseline_report

# Setup logging to both console and file
log_dir = Path("reports/fedavg")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "experiment_50_rounds.log"

logger = logging.getLogger("exp_50")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def run_experiment(config_path: str = "configs/config.yaml") -> dict:
    u_cfg = load_unified_config(config_path)
    device_name = u_cfg.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    seed = u_cfg.seed or 42
    num_clients = u_cfg.num_clients
    rounds = u_cfg.rounds  # 50
    epochs = u_cfg.epochs  # 1

    logger.info("=" * 65)
    logger.info("Starting 50-Round FedAvg Benchmark on CIFAR-10 (Dirichlet alpha=%.2f)", u_cfg.dataset.alpha)
    logger.info("Device: %s | Seed: %d | Clients: %d | Local Epochs: %d", device_name, seed, num_clients, epochs)
    logger.info("=" * 65)

    # 1. Initialize Server & Global Model
    global_model = SimpleCNN(seed=seed).to(device)
    server = FederatedServer(host="127.0.0.1", port=0, model=global_model)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info("FederatedServer listening at %s", server_addr)

    # 2. Checkpoints Directory
    checkpoint_dir = Path("checkpoints/fedavg")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Base Client Configuration
    base_cfg = u_cfg.to_client_config(client_id=0)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients
    base_cfg.training.device = device_name

    # 4. Partition Dataset with exact parameters
    logger.info(
        "Loading dataset '%s' (partition: %s, alpha=%.2f, seed=%s)...",
        base_cfg.dataset.name,
        base_cfg.dataset.partition,
        base_cfg.dataset.alpha,
        base_cfg.dataset.seed,
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

    # 5. Load Global Held-out Test Set
    global_test_loader = get_global_test_loader(
        dataset_name=base_cfg.dataset.name,
        data_dir=base_cfg.dataset.data_dir,
        batch_size=128,
        download=False,
    )
    logger.info("Global test loader loaded with %d batches (10,000 samples).", len(global_test_loader))

    # 6. Initialize & Register Clients
    nodes = []
    try:
        for i in range(num_clients):
            node = create_client_node(
                base_cfg,
                load_dataset=True,
                client_id=i,
                num_clients=num_clients,
                node_id=f"client_{i}",
                fed_dataset=fed_dataset,
            )
            reg = node.register()
            if not reg or not reg.success:
                raise RuntimeError(f"Failed to register client {i}: {reg}")
            nodes.append(node)
        logger.info("All %d client nodes successfully registered.", len(nodes))

        # 7. Coordinate 50 Federated Rounds
        round_history = []
        best_global_acc = 0.0
        best_round = 0

        for r in range(1, rounds + 1):
            t_start = time.time()
            weights = server.get_global_weights()
            responses = server.trigger_round(
                round_id=r,
                weights=weights,
                epochs=epochs,
                aggregate=True,
            )
            duration = time.time() - t_start

            successful = [res for res in responses if res.success]
            total_samples = sum(res.num_samples for res in successful)

            client_metrics = {}
            client_test_accs = []
            weighted_loss_sum = 0.0
            loss_samples = 0

            for res in successful:
                parsed = parse_client_metrics(res.message)
                c_loss = parsed.get("loss", 0.0)
                c_acc = parsed.get("test_acc", 0.0)
                c_test_loss = parsed.get("test_loss", 0.0)

                client_metrics[res.node_id] = {
                    "samples": res.num_samples,
                    "loss": round(c_loss, 4),
                    "test_accuracy": round(c_acc, 4),
                    "test_loss": round(c_test_loss, 4),
                }
                if "test_acc" in parsed:
                    client_test_accs.append(c_acc)

                if c_loss > 0:
                    weighted_loss_sum += c_loss * res.num_samples
                    loss_samples += res.num_samples

            round_loss = (weighted_loss_sum / loss_samples) if loss_samples > 0 else 0.0

            # Evaluate global model on held-out test set
            global_metrics = evaluate_model(
                model=server.global_model,
                dataloader=global_test_loader,
                device=device,
            )
            global_acc = round(global_metrics.get("accuracy", 0.0), 4)
            global_loss = round(global_metrics.get("loss", 0.0), 4)

            # Fairness & dispersion
            fairness = compute_fairness_metrics(client_test_accs)
            c_acc_list = [round(c_acc * 100, 2) for c_acc in client_test_accs]

            # Checkpoint tracking
            if global_acc > best_global_acc:
                best_global_acc = global_acc
                best_round = r
                torch.save(server.global_model.state_dict(), checkpoint_dir / "best_model.pt")

            if r % 10 == 0 or r == rounds:
                torch.save(server.global_model.state_dict(), checkpoint_dir / f"checkpoint_round_{r}.pt")

            round_record = {
                "round": r,
                "duration_seconds": round(duration, 2),
                "total_samples": total_samples,
                "weighted_loss": round(round_loss, 4),
                "global_test_accuracy": global_acc,
                "global_test_loss": global_loss,
                "mean_local_test_accuracy": round(fairness["mean"], 4),
                "std_local_test_accuracy": round(fairness["std"], 4),
                "min_local_test_accuracy": round(min(client_test_accs), 4) if client_test_accs else 0.0,
                "max_local_test_accuracy": round(max(client_test_accs), 4) if client_test_accs else 0.0,
                "worst_10_local_accuracy": round(fairness["worst_10"], 4),
                "client_metrics": client_metrics,
            }
            round_history.append(round_record)

            logger.info(
                "Round %2d/50 [%.1fs] | Loss: %.4f | Global Acc: %5.2f%% | Client Accs: %s | Mean Local: %5.2f%% (±%4.2f%%)",
                r,
                duration,
                round_loss,
                global_acc * 100,
                c_acc_list,
                fairness["mean"] * 100,
                fairness["std"] * 100,
            )

        # 8. Save structured report artifacts
        artifacts = save_baseline_report(
            baseline_name="fedavg",
            round_history=round_history,
            output_dir="reports",
        )
        logger.info("Baseline report artifacts successfully saved:")
        for k, v in artifacts.items():
            logger.info("  - %s: %s", k, v)

        # 9. Summary Statistics
        all_global_accs = [r["global_test_accuracy"] for r in round_history]
        max_acc = max(all_global_accs)
        max_round = all_global_accs.index(max_acc) + 1

        # Saturation point calculation: first round within 1.0% of max accuracy
        plateau_threshold = max_acc - 0.010
        plateau_round = next((i + 1 for i, a in enumerate(all_global_accs) if a >= plateau_threshold), max_round)

        final_r = round_history[-1]
        final_local_accs = [
            final_r["client_metrics"][k]["test_accuracy"] for k in final_r["client_metrics"]
        ]
        final_std = float(np.std(final_local_accs))
        final_min = min(final_local_accs)
        final_max = max(final_local_accs)
        final_spread = final_max - final_min

        logger.info("=" * 65)
        logger.info("50-ROUND EXPERIMENT COMPLETE")
        logger.info("Maximum Global Accuracy: %.2f%% (Achieved at Round %d)", max_acc * 100, max_round)
        logger.info("Accuracy Plateau Round (within 1%% of max): Round %d (%.2f%%)", plateau_round, all_global_accs[plateau_round - 1] * 100)
        logger.info("Final Round 50 Global Accuracy: %.2f%%", final_r["global_test_accuracy"] * 100)
        logger.info("Final Client Dispersion: Std = %.2f%%, Spread = [%.2f%%, %.2f%%] (diff: %.2f%%)", final_std * 100, final_min * 100, final_max * 100, final_spread * 100)
        logger.info("=" * 65)

        return {
            "round_history": round_history,
            "max_acc": max_acc,
            "max_round": max_round,
            "plateau_round": plateau_round,
            "final_std": final_std,
            "final_spread": final_spread,
            "final_min": final_min,
            "final_max": final_max,
            "artifacts": artifacts,
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


if __name__ == "__main__":
    run_experiment()
