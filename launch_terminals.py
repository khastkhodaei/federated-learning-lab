"""Terminal orchestrator for Federated Learning Lab.

Reads server and client configurations from YAML, parses IP, port, and number of clients,
and automatically spawns separate, live terminal windows for the central server
and each participating client node.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def load_launch_config(
    server_config_path: str = "configs/server.yaml",
    client_config_path: str = "configs/client.yaml",
) -> Dict[str, Any]:
    """Loads and merges launch parameters from YAML files.

    Args:
        server_config_path: Path to server.yaml.
        client_config_path: Path to client.yaml.

    Returns:
        Dictionary containing resolved launch parameters.
    """
    server_path = Path(server_config_path)
    client_path = Path(client_config_path)

    server_data = {}
    if server_path.is_file():
        with open(server_path, "r", encoding="utf-8") as f:
            server_data = yaml.safe_load(f) or {}

    client_data = {}
    if client_path.is_file():
        with open(client_path, "r", encoding="utf-8") as f:
            client_data = yaml.safe_load(f) or {}

    host = str(server_data.get("host", "127.0.0.1"))
    port = int(server_data.get("port", 50051))
    num_clients = int(server_data.get("num_clients", client_data.get("dataset", {}).get("num_clients", 2)))
    rounds = int(server_data.get("rounds", 3))
    epochs = int(server_data.get("epochs_per_round", 1))

    return {
        "host": host,
        "port": port,
        "num_clients": num_clients,
        "rounds": rounds,
        "epochs": epochs,
        "server_config": str(server_config_path),
        "client_config": str(client_config_path),
    }


def build_launch_commands(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs command lines for the central server and all client nodes.

    Args:
        cfg: Launch configuration dictionary.

    Returns:
        Dictionary with 'server' command and list of 'clients' commands.
    """
    python_bin = sys.executable
    server_cmd = (
        f"{python_bin} run_server.py "
        f"--config {cfg['server_config']} "
        f"--port {cfg['port']} "
        f"--clients {cfg['num_clients']} "
        f"--rounds {cfg['rounds']} "
        f"--epochs {cfg['epochs']}"
    )

    server_address = f"{cfg['host']}:{cfg['port']}"
    clients = []
    for i in range(cfg["num_clients"]):
        client_cmd = (
            f"{python_bin} run_client.py "
            f"--config {cfg['client_config']} "
            f"--id {i} "
            f"--clients {cfg['num_clients']} "
            f"--server {server_address}"
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


def launch_windows(
    cfg: Dict[str, Any],
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> List[subprocess.Popen]:
    """Launches server and client processes in individual terminal windows.

    Args:
        cfg: Launch configuration dictionary.
        dry_run: If True, only prints commands without executing.
        repo_root: Working directory root (defaults to current directory).

    Returns:
        List of created subprocesses (empty if dry_run).
    """
    commands = build_launch_commands(cfg)
    root = repo_root or Path.cwd()

    print("\n" + "=" * 65)
    print("      FEDERATED LEARNING LAB - MULTI-TERMINAL LAUNCHER")
    print(f"      Server Address: {cfg['host']}:{cfg['port']}")
    print(f"      Clients: {cfg['num_clients']} | Rounds: {cfg['rounds']} | Epochs: {cfg['epochs']}")
    print(f"      Working Dir: {root}")
    print("=" * 65 + "\n")

    print("[1/2] Server Command:")
    print(f"  {commands['server']}\n")

    print(f"[2/2] Client Commands ({len(commands['clients'])} nodes):")
    for c in commands["clients"]:
        print(f"  [{c['name']}]: {c['command']}")
    print()

    if dry_run:
        print("Dry run completed. No terminal processes were spawned.")
        return []

    processes: List[subprocess.Popen] = []

    # 1. Spawn Server Terminal
    print(">>> Spawning Central Server terminal window...")
    server_ps_cmd = f"Set-Location -Path '{root}'; $host.UI.RawUI.WindowTitle = 'Federated Server'; {commands['server']}"
    p_server = subprocess.Popen(
        ["powershell.exe", "-NoExit", "-Command", server_ps_cmd],
        cwd=str(root),
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
    parser.add_argument("--server-config", type=str, default="configs/server.yaml", help="Path to server YAML")
    parser.add_argument("--client-config", type=str, default="configs/client.yaml", help="Path to client YAML")
    parser.add_argument("--clients", "-c", type=int, default=None, help="Override number of clients")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override server port")
    parser.add_argument("--rounds", "-r", type=int, default=None, help="Override training rounds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without spawning windows")
    args = parser.parse_args()

    cfg = load_launch_config(args.server_config, args.client_config)
    if args.clients is not None:
        cfg["num_clients"] = args.clients
    if args.port is not None:
        cfg["port"] = args.port
    if args.rounds is not None:
        cfg["rounds"] = args.rounds

    launch_windows(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
