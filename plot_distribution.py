"""Standalone CLI for visualizing Federated Learning dataset partitions.

Loads partitioning parameters from configs/server.yaml and configs/client.yaml,
partitions the benchmark dataset using fl-data-generator, and generates a
high-resolution stacked bar chart visualizing client data distribution.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from fl_data_generator import FederatedDataset

from src.utils.plotting import (
    DATASET_CLASS_NAMES,
    compute_client_class_distribution,
    plot_client_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Data Distribution Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-config", type=str, default="configs/server.yaml", help="Path to server YAML")
    parser.add_argument("--client-config", type=str, default="configs/client.yaml", help="Path to client YAML")
    parser.add_argument("--clients", "-c", type=int, default=None, help="Override number of clients")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Dataset: cifar10, mnist, fashion_mnist")
    parser.add_argument("--partition", "-p", type=str, default=None, help="Partition: dirichlet, iid, pathological, quantity")
    parser.add_argument("--alpha", "-a", type=float, default=None, help="Dirichlet alpha parameter")
    parser.add_argument("--baseline", "-b", type=str, default=None, help="Baseline algorithm name (default: from config or fedavg)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path (defaults to reports/<baseline>/data_distribution.png)")
    parser.add_argument("--interactive", action="store_true", help="Display plot interactively in a window")
    args = parser.parse_args()

    # 1. Load settings from YAML
    server_data = {}
    if Path(args.server_config).is_file():
        with open(args.server_config, "r", encoding="utf-8") as f:
            server_data = yaml.safe_load(f) or {}

    client_data = {}
    if Path(args.client_config).is_file():
        with open(args.client_config, "r", encoding="utf-8") as f:
            client_data = yaml.safe_load(f) or {}

    ds_cfg = client_data.get("dataset", {})

    num_clients = args.clients if args.clients is not None else int(server_data.get("num_clients", ds_cfg.get("num_clients", 10)))
    dataset_name = args.dataset if args.dataset is not None else str(ds_cfg.get("name", "cifar10"))
    partition = args.partition if args.partition is not None else str(ds_cfg.get("partition", "dirichlet"))
    alpha = args.alpha if args.alpha is not None else float(ds_cfg.get("alpha", 0.5))
    seed = int(ds_cfg.get("seed", 42))
    baseline = (args.baseline or server_data.get("baseline", client_data.get("baseline", "fedavg"))).lower().strip()
    out_path = Path(args.output) if args.output else Path("reports") / baseline / "data_distribution.png"

    print("\n" + "=" * 65)
    print("      FEDERATED LEARNING LAB - DATA DISTRIBUTION VISUALIZER")
    print(f"      Dataset: {dataset_name} | Partition: {partition} (alpha={alpha})")
    print(f"      Clients: {num_clients} | Baseline: {baseline} | Random Seed: {seed}")
    print(f"      Output:  {out_path}")
    print("=" * 65 + "\n")

    # 2. Partition dataset using fl-data-generator (v0.1.4)
    print(f"Loading and partitioning '{dataset_name}' across {num_clients} clients...")
    fed_dataset = FederatedDataset(
        dataset=dataset_name,
        data_dir="./data",
        num_clients=num_clients,
        partition=partition,
        alpha=alpha,
        seed=seed,
    )

    # 3. Compute distribution matrix
    counts = compute_client_class_distribution(fed_dataset=fed_dataset)
    num_classes = counts.shape[1]
    class_names = DATASET_CLASS_NAMES.get(dataset_name.lower().strip(), [f"Class {i}" for i in range(num_classes)])

    # Print summary table
    print("\n--- Client Sample Counts ---")
    for i, c_counts in enumerate(counts):
        total = np.sum(c_counts)
        top_classes = np.argsort(c_counts)[::-1][:3]
        top_summary = ", ".join([f"{class_names[c]}: {c_counts[c]}" for c in top_classes if c_counts[c] > 0])
        print(f"  Client {i:2d}: {total:5d} samples (Top classes -> {top_summary})")

    # 4. Generate Plot
    title = (
        f"{dataset_name.upper()} Data Distribution across {num_clients} Clients\n"
        f"Partition: {partition.capitalize()} (alpha={alpha}) | Total Samples: {len(fed_dataset):,}"
    )

    print(f"\nSaving chart to '{args.output}'...")
    fig = plot_client_distribution(
        counts=counts,
        class_names=class_names,
        title=title,
        output_path=args.output,
        show=args.interactive,
    )
    plt.close(fig)

    print(f"Successfully generated data distribution chart: {args.output}\n")


if __name__ == "__main__":
    main()
