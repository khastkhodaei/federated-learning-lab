#!/usr/bin/env python3
"""IID Calibration Benchmark Runner for Federated Averaging (FedAvg).

Executes a controlled sanity check by partitioning CIFAR-10 uniformly across
participating clients (IID), training for multiple rounds, and comparing
mean local test accuracy against central global test accuracy.
"""

import logging
import os
import sys
import time

import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fl_data_generator import FederatedDataset

from src.client import create_client_node
from src.datasets.transforms import get_default_transform
from src.models.cnn import SimpleCNN
from src.server.dispatch import dispatch_round
from src.server.server import FederatedServer
from src.utils.config import load_unified_config
from src.utils.evaluation import (
    evaluate_model,
    get_global_test_loader,
    parse_client_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibration")


def run_iid_calibration(
    num_clients: int = 3,
    rounds: int = 5,
    epochs: int = 1,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict[str, float]]:
    logger.info("Initializing IID Calibration on %s (Seed: %d)...", device, seed)

    # 1. Initialize Server
    global_model = SimpleCNN(seed=seed).to(device)
    server = FederatedServer(host="127.0.0.1", port=0, model=global_model)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info("FederatedServer started at %s", server_addr)

    # 2. Base Client Configuration
    u_cfg = load_unified_config("configs/config.yaml")
    base_cfg = u_cfg.to_client_config(client_id=0)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients
    base_cfg.dataset.partition = "iid"
    base_cfg.training.device = device

    # 3. Partition CIFAR-10 IID
    logger.info("Partitioning CIFAR-10 IID across %d clients...", num_clients)
    transform = get_default_transform(base_cfg.dataset.name)
    fed_dataset = FederatedDataset(
        dataset=base_cfg.dataset.name,
        data_dir=base_cfg.dataset.data_dir,
        num_clients=num_clients,
        partition="iid",
        seed=seed,
        transform=transform,
    )

    # 4. Load Central Global Test Loader
    global_test_loader = get_global_test_loader(
        dataset_name=base_cfg.dataset.name,
        data_dir=base_cfg.dataset.data_dir,
        batch_size=128,
        download=False,
    )

    # 5. Initialize Clients
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
                raise RuntimeError(f"Client {i} registration failed")
            nodes.append(node)

        logger.info("Registered %d client nodes successfully.", len(nodes))

        results: list[dict[str, float]] = []

        # 6. Coordinate Rounds
        for r in range(1, rounds + 1):
            t0 = time.time()
            responses = dispatch_round(server, round_id=r, epochs=epochs)
            elapsed = time.time() - t0

            # Collect local accuracies
            local_accs: list[float] = []
            for res in responses:
                if res.success:
                    parsed = parse_client_metrics(res.message)
                    if "test_acc" in parsed:
                        local_accs.append(parsed["test_acc"])

            mean_local_acc = (sum(local_accs) / len(local_accs)) if local_accs else 0.0

            # Evaluate global model on balanced 10,000 CIFAR-10 test set
            global_metrics = evaluate_model(
                model=server.global_model,
                dataloader=global_test_loader,
                device=device,
            )
            global_acc = float(global_metrics["accuracy"])
            gap = abs(mean_local_acc - global_acc)

            row = {
                "round": r,
                "global_acc": round(global_acc * 100, 2),
                "mean_local_acc": round(mean_local_acc * 100, 2),
                "gap": round(gap * 100, 2),
                "client_accs": [round(a * 100, 2) for a in local_accs],
                "elapsed": round(elapsed, 2),
            }
            results.append(row)

            logger.info(
                "Round %d: Global Acc = %.2f%% | Mean Local Acc = %.2f%% | Gap = %.2f%% (Client Accs: %s) [%.1fs]",
                r,
                row["global_acc"],
                row["mean_local_acc"],
                row["gap"],
                row["client_accs"],
                row["elapsed"],
            )

        return results

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
    rounds_count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_iid_calibration(rounds=rounds_count)
