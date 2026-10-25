"""Standalone central federated server runner.

Starts the central coordinator, waits for the configured number of clients to
register over gRPC, broadcasts the global model, and orchestrates federated
training rounds with FedAvg aggregation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import yaml

from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.serialization import model_to_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Central Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-C", type=str, default="configs/server.yaml", help="Path to server YAML configuration")
    parser.add_argument("--port", "-p", type=int, default=None, help="Server port (overrides config)")
    parser.add_argument("--clients", "-c", type=int, default=None, help="Number of clients required before starting (overrides config)")
    parser.add_argument("--rounds", "-r", type=int, default=None, help="Number of federated training rounds (overrides config)")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Local epochs per round (overrides config)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Global model random seed (overrides config)")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline aggregation strategy (default: fedavg)")
    args = parser.parse_args()

    # Load defaults from config file if available
    cfg_data = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    port = args.port if args.port is not None else int(cfg_data.get("port", 50051))
    num_clients = args.clients if args.clients is not None else int(cfg_data.get("num_clients", 2))
    rounds = args.rounds if args.rounds is not None else int(cfg_data.get("rounds", 3))
    epochs = args.epochs if args.epochs is not None else int(cfg_data.get("epochs_per_round", 1))
    seed = args.seed if args.seed is not None else int(cfg_data.get("model", {}).get("seed", 42))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [SERVER] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("server")

    print("\n" + "=" * 65)
    print("       FEDERATED LEARNING LAB - CENTRAL COORDINATOR SERVER")
    print(f"       Port: {port} | Required Clients: {num_clients} | Rounds: {rounds} | Epochs: {epochs}")
    print(f"       Config: {args.config}")
    print("=" * 65 + "\n")

    logger.info("Initializing global SimpleCNN model (seed=%d)...", seed)
    global_model = SimpleCNN(seed=seed)
    initial_weights = model_to_bytes(global_model)
    logger.info("Global model initialized (%d bytes).", len(initial_weights))

    from src.baselines import get_baseline
    baseline_name = (args.baseline or cfg_data.get("baseline", "fedavg")).lower()
    strategy_cls, _ = get_baseline(baseline_name)
    strategy = strategy_cls()
    logger.info("Configured baseline aggregation strategy: %s (%s)", baseline_name, strategy_cls.__name__)

    server = FederatedServer(host="0.0.0.0", port=port, model=global_model, strategy=strategy)
    server.start()
    logger.info("Central Server is listening on 0.0.0.0:%d", server.bound_port)

    logger.info("Waiting for %d client(s) to connect and register...", num_clients)
    last_count = 0
    while True:
        registered = len(server.servicer.registered_nodes)
        if registered != last_count:
            nodes_list = list(server.servicer.registered_nodes.keys())
            logger.info("Registered clients (%d/%d): %s", registered, num_clients, nodes_list)
            last_count = registered

        if registered >= num_clients:
            logger.info("All %d clients connected! Starting training rounds in 2 seconds...", num_clients)
            time.sleep(2)
            break
        time.sleep(1)

    try:
        for r in range(1, rounds + 1):
            print("\n" + "-" * 55)
            logger.info(">>> STARTING FEDERATED ROUND %d / %d", r, rounds)
            print("-" * 55)

            weights_to_send = server.get_global_weights()
            logger.info(
                "Broadcasting global model weights (%d bytes) to %d clients...",
                len(weights_to_send),
                len(server.servicer.registered_nodes),
            )

            start_time = time.time()
            responses = server.trigger_round(
                round_id=r,
                weights=weights_to_send,
                epochs=epochs,
                aggregate=True,
            )
            elapsed = time.time() - start_time

            successful = [res for res in responses if res.success]
            total_samples = sum(res.num_samples for res in successful)

            for res in successful:
                logger.info(
                    "  [Update Received] Client '%s' submitted %d samples (%d bytes)",
                    res.node_id,
                    res.num_samples,
                    len(res.weights),
                )

            logger.info(
                "Round %d completed in %.2fs: %d/%d clients reported (total samples: %d).",
                r,
                elapsed,
                len(successful),
                num_clients,
                total_samples,
            )
            logger.info("FedAvg aggregation complete. Global model updated for round %d.", r)

        print("\n" + "=" * 65)
        logger.info("All %d federated training rounds completed successfully!", rounds)
        print("=" * 65 + "\n")

    except KeyboardInterrupt:
        logger.info("Stopping server...")
    finally:
        server.stop(grace=1)
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    main()
