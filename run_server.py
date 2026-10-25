"""Standalone central federated server runner.

Starts the central coordinator, waits for the configured number of clients to
register over gRPC, broadcasts the global model, and orchestrates federated
training rounds with FedAvg aggregation and publication-grade evaluation.
"""

import argparse
import logging
import time
from pathlib import Path

import yaml

from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.evaluation import (
    compute_fairness_metrics,
    evaluate_model,
    get_global_test_loader,
    parse_client_metrics,
)
from src.utils.hardware import get_hardware_summary
from src.utils.reporting import save_baseline_report
from src.utils.serialization import model_to_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Central Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        default="configs/server.yaml",
        help="Path to server YAML configuration",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Server port (overrides config)",
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=None,
        help="Number of clients required before starting (overrides config)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=None,
        help="Number of federated training rounds (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Local epochs per round (overrides config)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Global model random seed (overrides config)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline aggregation strategy (default: fedavg)",
    )
    args = parser.parse_args()

    # Load defaults from config file if available
    cfg_data = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    port = args.port if args.port is not None else int(cfg_data.get("port", 50051))
    num_clients = (
        args.clients
        if args.clients is not None
        else int(cfg_data.get("num_clients", 2))
    )
    rounds = args.rounds if args.rounds is not None else int(cfg_data.get("rounds", 3))
    epochs = (
        args.epochs
        if args.epochs is not None
        else int(cfg_data.get("epochs_per_round", 1))
    )
    seed = (
        args.seed
        if args.seed is not None
        else int(cfg_data.get("model", {}).get("seed", 42))
    )
    baseline_name = (args.baseline or cfg_data.get("baseline", "fedavg")).lower()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [SERVER] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("server")

    hw_info = get_hardware_summary()

    print("\n" + "=" * 65)
    print("       FEDERATED LEARNING LAB - CENTRAL COORDINATOR SERVER")
    print(f"       Baseline: {baseline_name.upper()} | Hardware: {hw_info}")
    print(
        f"       Port: {port} | Required Clients: {num_clients} | Rounds: {rounds} | Epochs: {epochs}"
    )
    print(f"       Config: {args.config}")
    print("=" * 65 + "\n")

    logger.info("Initializing global SimpleCNN model (seed=%d)...", seed)
    global_model = SimpleCNN(seed=seed)
    initial_weights = model_to_bytes(global_model)
    logger.info("Global model initialized (%d bytes).", len(initial_weights))

    from src.baselines import get_baseline

    strategy_cls, _ = get_baseline(baseline_name)
    strategy = strategy_cls()
    logger.info(
        "Configured baseline aggregation strategy: %s (%s)",
        baseline_name,
        strategy_cls.__name__,
    )

    server = FederatedServer(
        host="0.0.0.0", port=port, model=global_model, strategy=strategy
    )
    server.start()
    logger.info("Central Server is listening on 0.0.0.0:%d", server.bound_port)

    # Pre-load global held-out evaluation test set
    try:
        dataset_name = (
            cfg_data.get("dataset", {}).get("name", "cifar10")
            if isinstance(cfg_data.get("dataset"), dict)
            else "cifar10"
        )
        global_test_loader = get_global_test_loader(
            dataset_name=dataset_name,
            data_dir="./data",
            batch_size=128,
            download=False,
        )
        logger.info(
            "Global benchmark evaluation set loaded (%s test batches).",
            len(global_test_loader),
        )
    except Exception as err:  # noqa: BLE001
        logger.warning(
            "Could not initialize global test dataset: %s. Global evaluation will be skipped.",
            err,
        )
        global_test_loader = None

    logger.info("Waiting for %d client(s) to connect and register...", num_clients)
    last_count = 0
    while True:
        registered = len(server.servicer.registered_nodes)
        if registered != last_count:
            nodes_list = list(server.servicer.registered_nodes.keys())
            logger.info(
                "Registered clients (%d/%d): %s", registered, num_clients, nodes_list
            )
            last_count = registered

        if registered >= num_clients:
            logger.info(
                "All %d clients connected! Starting training rounds in 2 seconds...",
                num_clients,
            )
            time.sleep(2)
            break
        time.sleep(1)

    try:
        round_history = []
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

                logger.info(
                    "  [Update Received] Client '%s' submitted %d samples (loss: %.4f, test acc: %.2f%%)",
                    res.node_id,
                    res.num_samples,
                    c_loss,
                    c_acc * 100,
                )

            round_loss = (
                (weighted_loss_sum / loss_samples) if loss_samples > 0 else 0.0
            )

            # Evaluate aggregated global model on held-out global test set
            global_acc = 0.0
            global_loss = 0.0
            if global_test_loader is not None and server.global_model is not None:
                global_metrics = evaluate_model(
                    model=server.global_model,
                    dataloader=global_test_loader,
                    criterion=getattr(server, "criterion", None),
                )
                global_acc = round(global_metrics.get("accuracy", 0.0), 4)
                global_loss = round(global_metrics.get("loss", 0.0), 4)

            # Compute client fairness dispersion statistics
            fairness = compute_fairness_metrics(client_test_accs)

            round_history.append({
                "round": r,
                "duration_seconds": round(elapsed, 2),
                "total_samples": total_samples,
                "weighted_loss": round(round_loss, 4),
                "global_test_accuracy": global_acc,
                "global_test_loss": global_loss,
                "mean_local_test_accuracy": round(fairness["mean"], 4),
                "std_local_test_accuracy": round(fairness["std"], 4),
                "worst_10_local_accuracy": round(fairness["worst_10"], 4),
                "client_metrics": client_metrics,
            })

            logger.info(
                "Round %d completed in %.2fs: %d/%d clients | Loss: %.4f | Global Test Acc: %.2f%% | Mean Local Acc: %.2f%% (±%.2f%%, Worst 10%%: %.2f%%).",
                r,
                elapsed,
                len(successful),
                num_clients,
                round_loss,
                global_acc * 100,
                fairness["mean"] * 100,
                fairness["std"] * 100,
                fairness["worst_10"] * 100,
            )
            logger.info(
                "%s aggregation complete. Global model updated for round %d.",
                strategy_cls.__name__,
                r,
            )

        print("\n" + "=" * 65)
        logger.info(
            "All %d federated training rounds completed successfully!", rounds
        )
        print("=" * 65 + "\n")

        # Automatically save structured baseline report and training progression charts
        artifacts = save_baseline_report(
            baseline_name=baseline_name, round_history=round_history
        )
        logger.info("Baseline report artifacts generated successfully:")
        logger.info("  [Metrics JSON]    %s", artifacts["metrics_json"])
        logger.info("  [Loss Curve]          %s", artifacts["loss_curve"])
        logger.info("  [Accuracy Curves]     %s", artifacts["accuracy_curves"])
        logger.info("  [Client Dispersion]   %s", artifacts["client_dispersion"])
        logger.info("  [Round Duration]      %s", artifacts["round_duration"])

    except KeyboardInterrupt:
        logger.info("Stopping server...")
    finally:
        server.stop(grace=1)
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    main()
