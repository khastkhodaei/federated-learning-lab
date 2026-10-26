"""Standalone central federated server runner.

Starts the central coordinator, waits for the configured number of clients to
register over gRPC, broadcasts models, and orchestrates federated
training rounds with aggregation and publication-grade evaluation.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.baselines import create_server_strategy
from src.config import UnifiedConfig, load_unified_config
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.utils.evaluation import (
    compute_fairness_metrics,
    evaluate_model,
    get_global_test_loader,
    parse_client_metrics,
)
from src.utils.hardware import get_hardware_summary, resolve_device
from src.utils.reporting import save_baseline_report
from src.utils.serialization import model_to_bytes


@dataclass
class ServerSettings:
    """Resolved settings for central federated server coordinator."""

    host: str
    port: int
    num_clients: int
    rounds: int
    epochs: int
    device: str
    seed: int
    baseline: str
    report_dir: str = "reports"

    @property
    def clients(self) -> int:
        """Alias for num_clients for backward compatibility."""
        return self.num_clients

    def __getitem__(self, key: str) -> Any:
        if key == "clients":
            return self.num_clients
        return getattr(self, key)


def _get_override(obj: Any, *keys: str) -> Any:
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] is not None:
            return obj[k]
        val = getattr(obj, k, None)
        if val is not None:
            return val
    return None


def resolve_server_settings(args: Any, u_cfg: UnifiedConfig) -> ServerSettings:
    """Resolves server configuration settings by combining CLI overrides with UnifiedConfig defaults.

    CLI arguments take precedence only when explicitly provided (non-None).
    """
    host_override = _get_override(args, "host")
    port_override = _get_override(args, "port")
    clients_override = _get_override(args, "clients", "num_clients")
    rounds_override = _get_override(args, "rounds")
    epochs_override = _get_override(args, "epochs")
    seed_override = _get_override(args, "seed")
    device_override = _get_override(args, "device")
    baseline_override = _get_override(args, "baseline")
    report_dir_override = _get_override(args, "report_dir", "reports_dir")

    host = host_override if host_override is not None else getattr(u_cfg, "server_host", "127.0.0.1")
    port = port_override if port_override is not None else getattr(u_cfg, "server_port", 50051)
    num_clients = clients_override if clients_override is not None else getattr(u_cfg, "num_clients", 3)
    rounds = rounds_override if rounds_override is not None else getattr(u_cfg, "rounds", 15)
    epochs = epochs_override if epochs_override is not None else getattr(u_cfg, "epochs", 1)
    report_dir = str(report_dir_override) if report_dir_override is not None else "reports"

    if seed_override is not None:
        seed = seed_override
    else:
        cfg_seed = getattr(u_cfg, "seed", None)
        if cfg_seed is not None:
            seed = cfg_seed
        elif hasattr(u_cfg, "model") and getattr(u_cfg.model, "seed", None) is not None:
            seed = u_cfg.model.seed
        else:
            seed = 42

    if device_override is not None:
        device_candidate = device_override
    else:
        cfg_dev = getattr(u_cfg, "device", None)
        if cfg_dev is not None:
            device_candidate = cfg_dev
        elif hasattr(u_cfg, "experiment") and getattr(u_cfg.experiment, "device", None) is not None:
            device_candidate = u_cfg.experiment.device
        else:
            device_candidate = None
    device = resolve_device(device_candidate)

    if baseline_override is not None:
        baseline = str(baseline_override).lower().strip()
    else:
        raw_b = getattr(u_cfg, "baseline", None) or getattr(u_cfg, "algorithm", "fedavg")
        baseline = str(raw_b).lower().strip()

    return ServerSettings(
        host=str(host),
        port=int(port),
        num_clients=int(num_clients),
        rounds=int(rounds),
        epochs=int(epochs),
        device=str(device),
        seed=int(seed),
        baseline=str(baseline),
        report_dir=str(report_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Central Server",
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
        "--host",
        "-H",
        type=str,
        default=None,
        help="Server host address (overrides config)",
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
        "--device",
        type=str,
        default=None,
        help="Device to run coordinator and evaluation on (e.g. cuda, cpu)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline aggregation strategy (default: fedavg)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save baseline report artifacts (default: reports)",
    )
    args = parser.parse_args()

    # Load defaults from unified config file if available
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        u_cfg = load_unified_config(cfg_path)
    else:
        u_cfg = UnifiedConfig()

    settings = resolve_server_settings(args, u_cfg)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [SERVER] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("server")

    server_device = torch.device(settings.device)
    hw_info = get_hardware_summary(settings.device)

    print("\n" + "=" * 65)
    print("       FEDERATED LEARNING LAB - CENTRAL COORDINATOR SERVER")
    print(f"       Baseline: {settings.baseline.upper()} | Hardware: {hw_info}")
    print(
        f"       Host: {settings.host} | Port: {settings.port} | Required Clients: {settings.num_clients} | Rounds: {settings.rounds} | Epochs: {settings.epochs}"
    )
    print(f"       Config: {args.config}")
    print("=" * 65 + "\n")

    logger.info("Initializing global SimpleCNN model (seed=%d) on %s...", settings.seed, settings.device)
    global_model = SimpleCNN(seed=settings.seed).to(server_device)
    initial_weights = model_to_bytes(global_model)
    logger.info("Global model initialized (%d bytes).", len(initial_weights))

    strategy = create_server_strategy(settings.baseline, config=u_cfg)
    logger.info(
        "Configured baseline aggregation strategy: %s (%s)",
        settings.baseline,
        strategy.__class__.__name__,
    )

    server = FederatedServer(
        host=settings.host, port=settings.port, model=global_model, strategy=strategy
    )
    server.start()
    logger.info("Central Server is listening on %s:%d", settings.host, server.bound_port)

    # Pre-load global held-out evaluation test set
    try:
        dataset_name = u_cfg.dataset.name if hasattr(u_cfg, "dataset") else "cifar10"
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

    logger.info("Waiting for %d client(s) to connect and register...", settings.num_clients)
    last_count = 0
    while True:
        registered = len(server.servicer.registered_nodes)
        if registered != last_count:
            nodes_list = list(server.servicer.registered_nodes.keys())
            logger.info(
                "Registered clients (%d/%d): %s", registered, settings.num_clients, nodes_list
            )
            last_count = registered

        if registered >= settings.num_clients:
            logger.info(
                "All %d clients connected! Starting training rounds in 2 seconds...",
                settings.num_clients,
            )
            time.sleep(2)
            break
        time.sleep(1)

    try:
        round_history = []
        for r in range(1, settings.rounds + 1):
            print("\n" + "-" * 55)
            logger.info(">>> STARTING FEDERATED ROUND %d / %d", r, settings.rounds)
            print("-" * 55)

            payload_desc = (
                f"{len(server.client_weights)} client models"
                if server.client_weights
                else f"global model ({len(server.get_global_weights())} bytes)"
            )
            logger.info(
                "Dispatching %s to %d clients...",
                payload_desc,
                len(server.servicer.registered_nodes),
            )

            start_time = time.time()
            responses = server.trigger_round(
                round_id=r,
                epochs=settings.epochs,
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
                settings.num_clients,
                round_loss,
                global_acc * 100,
                fairness["mean"] * 100,
                fairness["std"] * 100,
                fairness["worst_10"] * 100,
            )
            logger.info(
                "%s aggregation complete. Global model updated for round %d.",
                strategy.__class__.__name__,
                r,
            )

        print("\n" + "=" * 65)
        logger.info(
            "All %d federated training rounds completed successfully!", settings.rounds
        )
        print("=" * 65 + "\n")

        # Automatically save structured baseline report and training progression charts
        artifacts = save_baseline_report(
            baseline_name=settings.baseline,
            round_history=round_history,
            output_dir=settings.report_dir,
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
