from src.config import load_unified_config

"""Terminal orchestrator for Federated Learning Lab.

Reads server and client configurations from YAML, parses IP, port, and number of clients,
automatically generates the client data distribution chart, and spawns separate, live
terminal windows for the central server and each participating client node.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from fl_data_generator import FederatedDataset

from src.utils.plotting import (
    DATASET_CLASS_NAMES,
    compute_client_class_distribution,
    plot_client_distribution,
)


def load_launch_config(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Loads experiment parameters from the unified configuration file."""
    cfg_p = Path(config_path)
    u_cfg = load_unified_config(cfg_p)
    return {
        "host": u_cfg.server_host,
        "port": u_cfg.server_port,
        "num_clients": u_cfg.num_clients,
        "rounds": u_cfg.rounds,
        "epochs": u_cfg.epochs,
        "baseline": u_cfg.baseline,
        "device": u_cfg.device,
        "dataset": u_cfg.dataset.name,
        "partition": u_cfg.dataset.partition,
        "alpha": u_cfg.dataset.alpha,
        "seed": u_cfg.dataset.seed,
        "config_path": str(cfg_p),
    }


def build_launch_commands(cfg: dict[str, Any]) -> dict[str, Any]:
    """Constructs command lines for the central server and all client nodes."""
    python_bin = sys.executable
    baseline_flag = f" --baseline {cfg['baseline']}" if cfg.get("baseline") else ""
    device_flag = f" --device {cfg['device']}" if cfg.get("device") else ""
    cfg_arg = cfg.get("config_path") or cfg.get("server_config", "configs/config.yaml")
    server_cmd = (
        f"{python_bin} run_server.py "
        f"--config {cfg_arg} "
        f"--port {cfg['port']} "
        f"--clients {cfg['num_clients']} "
        f"--rounds {cfg['rounds']} "
        f"--epochs {cfg['epochs']}"
        f"{baseline_flag}"
        f"{device_flag}"
    )

    server_address = f"{cfg['host']}:{cfg['port']}"
    clients = []
    for i in range(cfg["num_clients"]):
        cfg_arg = cfg.get("config_path") or cfg.get("client_config", "configs/config.yaml")
        client_cmd = (
            f"{python_bin} run_client.py "
            f"--config {cfg_arg} "
            f"--id {i} "
            f"--clients {cfg['num_clients']} "
            f"--server {server_address}"
            f"{baseline_flag}"
            f"{device_flag}"
        )
        clients.append({
            "id": i,
            "name": f"client_{i}",
            "command": client_cmd,
        })

    return {
        "server": server_cmd,
        "clients": clients,
    }


def generate_distribution_chart(
    cfg: dict[str, Any],
    output_path: str | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """Generates and saves the client data distribution chart inside the baseline reports folder."""
    baseline = str(cfg.get("baseline", "fedavg")).lower().strip()
    dataset = str(cfg.get("dataset") or cfg.get("dataset_name") or "cifar10")
    partition = str(cfg.get("partition", "dirichlet"))
    alpha = float(cfg.get("alpha", 0.5))
    seed = int(cfg.get("seed", 42))
    num_clients = int(cfg.get("num_clients", 3))

    if output_path is None:
        out_path = Path(output_dir) / baseline / "data_distribution.png"
    else:
        out_path = Path(output_path)

    print(f"Generating data distribution chart for {num_clients} clients ({dataset}, {partition})...")
    fed_dataset = FederatedDataset(
        dataset=dataset,
        data_dir="./data",
        num_clients=num_clients,
        partition=partition,
        alpha=alpha,
        seed=seed,
    )

    counts = compute_client_class_distribution(fed_dataset=fed_dataset)
    num_classes = counts.shape[1]
    class_names = DATASET_CLASS_NAMES.get(dataset.lower().strip(), [f"Class {i}" for i in range(num_classes)])

    title = (
        f"{dataset.upper()} Data Distribution across {num_clients} Clients ({baseline.upper()})\n"
        f"Partition: {partition.capitalize()} (alpha={alpha}) | Total Samples: {len(fed_dataset):,}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_client_distribution(
        counts=counts,
        class_names=class_names,
        title=title,
        output_path=str(out_path),
        show=False,
    )
    plt.close(fig)

    print(f"Chart saved to: {out_path}")
    return out_path


def launch_windows(
    cfg: dict[str, Any],
    dry_run: bool = False,
    plot: bool = True,
    repo_root: Path | None = None,
) -> list[subprocess.Popen]:
    """Launches server and client processes in individual terminal windows."""
    commands = build_launch_commands(cfg)
    root = repo_root or Path.cwd()

    from src.utils.hardware import get_hardware_summary
    baseline_raw = str(cfg.get("baseline", "fedavg")).strip()
    baseline_name = "FedAvg" if baseline_raw.lower() == "fedavg" else baseline_raw.upper()
    hardware_summary = get_hardware_summary(str(cfg.get("device", "cpu")))

    print("\n" + "=" * 65)
    print("      FEDERATED LEARNING LAB - MULTI-TERMINAL LAUNCHER")
    print(f"      Baseline Method: {baseline_name}")
    print(f"      Hardware:        {hardware_summary}")
    print(f"      Server Address:  {cfg['host']}:{cfg['port']}")
    print(f"      Clients: {cfg['num_clients']} | Rounds: {cfg['rounds']} | Epochs: {cfg['epochs']}")
    print(f"      Working Dir:     {root}")
    print("=" * 65 + "\n")

    # Generate data distribution chart if requested
    if plot:
        try:
            generate_distribution_chart(cfg)
        except (OSError, RuntimeError) as exc:
            print(f"Warning: Could not generate distribution plot: {exc}")

    print("[1/2] Server Command:")
    print(f"  {commands['server']}\n")

    print(f"[2/2] Client Commands ({len(commands['clients'])} nodes):")
    for c in commands["clients"]:
        print(f"  [{c['name']}]: {c['command']}")
    print()

    if dry_run:
        print("Dry run completed. No terminal processes were spawned.")
        return []

    processes: list[subprocess.Popen] = []
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if sys.platform == "win32" else 0

    # 1. Spawn Server Terminal
    print(">>> Spawning Central Server terminal window...")
    server_ps_cmd = f"Set-Location -Path '{root}'; $host.UI.RawUI.WindowTitle = 'Federated Server'; {commands['server']}"
    p_server = subprocess.Popen(
        ["powershell.exe", "-NoExit", "-Command", server_ps_cmd],
        cwd=str(root),
        creationflags=creationflags,
    )
    processes.append(p_server)

    # Wait for server to bind port
    print("Waiting 2 seconds for server to start listening...")
    time.sleep(2)

    # 2. Spawn Each Client Terminal
    for c in commands["clients"]:
        print(f">>> Spawning Terminal for {c['name']}...")
        client_ps_cmd = (
            f"Set-Location -Path '{root}'; "
            f"$host.UI.RawUI.WindowTitle = 'Client Node: {c['name']}'; "
            f"{c['command']}"
        )
        p_client = subprocess.Popen(
            ["powershell.exe", "-NoExit", "-Command", client_ps_cmd],
            cwd=str(root),
            creationflags=creationflags,
        )
        processes.append(p_client)
        time.sleep(0.5)

    print("\n" + "=" * 65)
    print(f"Successfully launched 1 server + {len(commands['clients'])} clients in individual terminal windows!")
    print("All terminals are live. You can monitor the communication process across each window.")
    print("=" * 65 + "\n")

    return processes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Multi-Terminal Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-C", type=str, default="configs/config.yaml", help="Path to unified YAML configuration")
    parser.add_argument("--clients", "-c", type=int, default=None, help="Override number of clients")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override server port")
    parser.add_argument("--rounds", "-r", type=int, default=None, help="Override training rounds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without spawning windows")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating data distribution chart")
    args = parser.parse_args()

    cfg = load_launch_config(config_path=args.config)
    if args.clients is not None:
        cfg["num_clients"] = args.clients
    if args.port is not None:
        cfg["port"] = args.port
    if args.rounds is not None:
        cfg["rounds"] = args.rounds

    launch_windows(cfg, dry_run=args.dry_run, plot=not args.no_plot)


if __name__ == "__main__":
    main()
