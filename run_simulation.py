#!/usr/bin/env python3
"""Federated Learning Lab: Multi-Client Orchestrator.

Spawns a central server and N client nodes from a global client.yaml configuration,
coordinates synchronous federated rounds via gRPC, and aggregates updates with FedAvg.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.client import create_client_node
from src.client.config import load_client_config
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.serialization import model_to_bytes

logger = logging.getLogger("simulation")


def run_federated_simulation(
    num_clients: int = 10,
    rounds: int = 3,
    epochs: int = 1,
    config_path: str = "configs/client.yaml",
    server_port: int = 50051,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """Runs a complete federated learning training simulation.

    Args:
        num_clients: Total number of participating clients.
        rounds: Number of federated training rounds to coordinate.
        epochs: Local training epochs per client per round.
        config_path: Path to the global client YAML configuration template.
        server_port: Port for the central gRPC server (0 for ephemeral OS port).
        synthetic: If True, uses fast synthetic CIFAR-like tensors instead of downloading.

    Returns:
        Dictionary summarizing simulation results and round history.
    """
    logger.info("Initializing Central Federated Server...")
    global_model = SimpleCNN(seed=42)
    initial_weights = model_to_bytes(global_model)

    server = FederatedServer(host="127.0.0.1", port=server_port, model=global_model)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    logger.info("Server bound and listening at %s", server_addr)

    # Load base client configuration template
    base_cfg = load_client_config(config_path)
    base_cfg.server_address = server_addr
    base_cfg.dataset.num_clients = num_clients

    # Prepare dataset partitions
    fed_dataset = None
    synthetic_loaders = []
    if synthetic:
        logger.info("Using synthetic data partition for %d clients...", num_clients)
        for _ in range(num_clients):
            inputs = torch.randn(32, 3, 32, 32)
            targets = torch.randint(0, 10, (32,))
            loader = DataLoader(TensorDataset(inputs, targets), batch_size=base_cfg.dataset.batch_size)
            synthetic_loaders.append(loader)
    else:
        logger.info(
            "Loading '%s' partition (%s, alpha=%.2f) across %d clients via fl-data-generator...",
            base_cfg.dataset.name,
            base_cfg.dataset.partition,
            base_cfg.dataset.alpha,
            num_clients,
        )
        from fl_data_generator import FederatedDataset

        fed_dataset = FederatedDataset(
            dataset=base_cfg.dataset.name,
            data_dir=base_cfg.dataset.data_dir,
            num_clients=num_clients,
            partition=base_cfg.dataset.partition,
            alpha=base_cfg.dataset.alpha,
            seed=base_cfg.dataset.seed,
        )

    # Initialize and register all client nodes
    nodes: List[Any] = []
    logger.info("Launching and registering %d client nodes...", num_clients)
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

            # Register with central coordinator
            reg_res = node.register()
            if not reg_res or not reg_res.success:
                raise RuntimeError(f"Failed to register client {node_id}: {reg_res}")
            nodes.append(node)

        logger.info("All %d clients successfully registered with the server.", len(nodes))

        # Coordinate Federated Rounds
        round_history: List[Dict[str, Any]] = []
        for r in range(1, rounds + 1):
            logger.info("========================================")
            logger.info(">>> Starting Federated Round %d / %d", r, rounds)
            logger.info("========================================")

            start_time = time.time()
            responses = server.trigger_round(round_id=r, epochs=epochs, aggregate=True)
            elapsed = time.time() - start_time

            successful_responses = [res for res in responses if res.success]
            total_samples = sum(res.num_samples for res in successful_responses)

            logger.info(
                "Round %d completed in %.2fs: %d/%d nodes reported (total samples: %d).",
                r,
                elapsed,
                len(successful_responses),
                num_clients,
                total_samples,
            )

            round_history.append(
                {
                    "round_id": r,
                    "elapsed_sec": elapsed,
                    "num_responses": len(successful_responses),
                    "total_samples": total_samples,
                }
            )

        final_weights = server.get_global_weights()
        logger.info("Federated training completed successfully across %d rounds.", rounds)

        return {
            "num_clients": num_clients,
            "rounds_completed": rounds,
            "round_history": round_history,
            "initial_model_weights": initial_weights,
            "final_model_weights": final_weights,
        }

    finally:
        logger.info("Shutting down %d client nodes and central server...", len(nodes))
        for node in nodes:
            try:
                node.stop(grace=0)
                if node.transport is not None:
                    node.transport.close()
            except Exception:
                pass
        server.stop(grace=0)
        logger.info("Cleanup complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab Simulation Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=10,
        help="Number of client nodes to run in the federated network",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=3,
        help="Number of federated aggregation rounds",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="Number of local epochs each client executes per round",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/client.yaml",
        help="Path to global client YAML configuration template",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=50051,
        help="Port for the central coordinator server",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use fast synthetic tensors instead of downloading/partitioning CIFAR-10",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("      Federated Learning Lab - Simulation Runner")
    print(f"      Clients: {args.clients} | Rounds: {args.rounds} | Epochs: {args.epochs}")
    print(f"      Config:  {args.config}")
    print("=" * 60 + "\n")

    run_federated_simulation(
        num_clients=args.clients,
        rounds=args.rounds,
        epochs=args.epochs,
        config_path=args.config,
        server_port=args.port,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
