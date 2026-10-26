#!/usr/bin/env python3
"""Automated Multi-Baseline Benchmark Orchestrator for Federated Learning Lab.

Sequentially executes all 5 federated learning baselines under identical experimental conditions:
- FedAvg (McMahan et al., 2017)
- FedProx (Li et al., 2020) [mu=0.01]
- Ditto (Li et al., 2021) [lambda=0.1, pers_epochs=1]
- Per-FedAvg (Fallah et al., 2020) [alpha_lr=0.01, beta_lr=0.005]
- pFedGraph (Chen et al., 2022) [graph_alpha=0.1, graph_lam=0.01]

Saves per-baseline metrics to reports/<algorithm>/metrics.json and generates combined
cross-algorithm comparison plots in reports/benchmark_comparison/.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_simulation import run_federated_simulation
from src.utils.benchmark_plotting import plot_benchmark_summary

logger = logging.getLogger("benchmark")

# Benchmark configuration specifications for each baseline
BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "fedavg": {
        "description": "Federated Averaging (Canonical Baseline)",
        "params": {},
    },
    "fedprox": {
        "description": "FedProx (Proximal Regularization, mu=0.01)",
        "params": {"mu": 0.01},
    },
    "ditto": {
        "description": "Ditto (Personalized Dual-Model, lambda=0.1)",
        "params": {"lambda_penalty": 0.1, "personalized_epochs": 1},
    },
    "perfedavg": {
        "description": "Per-FedAvg (FO-MAML Meta-Learning, alpha=0.01, beta=0.005)",
        "params": {"alpha_lr": 0.01, "beta_lr": 0.005},
    },
    "pfedgraph": {
        "description": "pFedGraph (Graph Collaboration, alpha=0.1, lam=0.01)",
        "params": {"graph_alpha": 0.1, "graph_lam": 0.01},
    },
}


def run_full_benchmark(
    rounds: int = 50,
    clients: int = 3,
    epochs: int = 1,
    algorithms: list[str] | None = None,
    config_path: str = "configs/config.yaml",
    reports_dir: str = "reports",
    output_dir: str = "reports/benchmark_comparison",
    skip_existing: bool = False,
    synthetic: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Orchestrates sequential execution of all baseline simulations and plots comparisons.

    Args:
        rounds: Number of federated training rounds per baseline.
        clients: Number of participating clients.
        epochs: Local training epochs per client.
        algorithms: List of algorithm names to run (default: all 5).
        config_path: Unified YAML config path.
        reports_dir: Directory containing per-baseline reports.
        output_dir: Directory to save combined comparison plots.
        skip_existing: If True, skips runs where metrics.json already contains >= rounds.
        synthetic: If True, uses fast synthetic tensors.
        device: Target compute device ('cuda' or 'cpu').

    Returns:
        Dictionary summarizing execution status, per-algorithm results, and generated plots.
    """
    target_algos = algorithms or list(BENCHMARK_CONFIGS.keys())
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("FEDERATED LEARNING LAB - COMPLETE MULTI-BASELINE BENCHMARK SUITE")
    logger.info("Algorithms to evaluate: %s", ", ".join(target_algos).upper())
    logger.info("Conditions: Rounds=%d, Clients=%d, Local Epochs=%d, Dirichlet alpha=0.1", rounds, clients, epochs)
    logger.info("=" * 75)

    benchmark_results: dict[str, Any] = {}
    total_start = time.time()

    for idx, algo in enumerate(target_algos, start=1):
        clean_algo = algo.lower().strip()
        spec = BENCHMARK_CONFIGS.get(clean_algo, {"description": clean_algo, "params": {}})
        metrics_file = reports_path / clean_algo / "metrics.json"

        # Check if existing run has completed matching or greater rounds
        if skip_existing and metrics_file.is_file():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_rounds = len(cached_data.get("rounds", []))
                if cached_rounds >= rounds:
                    logger.info(
                        "[%d/%d] Skipping '%s': metrics.json already exists with %d rounds.",
                        idx,
                        len(target_algos),
                        clean_algo.upper(),
                        cached_rounds,
                    )
                    benchmark_results[clean_algo] = cached_data
                    continue
            except Exception:
                pass

        logger.info("")
        logger.info("*" * 65)
        logger.info(
            "[%d/%d] Executing Baseline: %s",
            idx,
            len(target_algos),
            spec["description"],
        )
        logger.info("*" * 65)

        run_params = dict(spec["params"])
        sim_start = time.time()

        result = run_federated_simulation(
            num_clients=clients,
            rounds=rounds,
            epochs=epochs,
            algorithm=clean_algo,
            config_path=config_path,
            server_port=0,
            synthetic=synthetic,
            device=device,
            **run_params,
        )

        sim_duration = time.time() - sim_start
        logger.info(
            "[%d/%d] Completed '%s' in %.2f minutes (%.1fs).",
            idx,
            len(target_algos),
            clean_algo.upper(),
            sim_duration / 60.0,
            sim_duration,
        )
        benchmark_results[clean_algo] = result

    total_duration = time.time() - total_start
    logger.info("")
    logger.info("=" * 75)
    logger.info("All %d baseline runs completed in %.2f minutes.", len(target_algos), total_duration / 60.0)
    logger.info("Generating combined comparison plots in '%s'...", output_dir)
    logger.info("=" * 75)

    # Generate combined comparison charts
    plot_artifacts = plot_benchmark_summary(
        reports_dir=reports_dir,
        output_dir=output_dir,
        algorithms=target_algos,
    )

    for k, p in plot_artifacts.items():
        logger.info("  - Generated %s: %s", k, p)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Print final summary table
    print("\n" + "=" * 80)
    print(f"           BENCHMARK COMPARISON SUMMARY ({rounds} ROUNDS, DIRICHLET alpha = 0.1)")
    print("=" * 80)
    print(f"{'Algorithm':<15} | {'Max Global Acc':<16} | {'Final Global Acc':<18} | {'Final Local Acc':<18}")
    print("-" * 80)

    for algo in target_algos:
        m_file = reports_path / algo / "metrics.json"
        if m_file.is_file():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                r_list = d.get("rounds", [])
                if r_list:
                    g_accs = [float(r.get("global_test_accuracy", 0.0)) * 100 for r in r_list]
                    l_accs = [float(r.get("mean_local_test_accuracy", 0.0)) * 100 for r in r_list]
                    max_g = max(g_accs)
                    final_g = g_accs[-1]
                    final_l = l_accs[-1]
                    print(f"{algo.upper():<15} | {max_g:6.2f}%           | {final_g:6.2f}%             | {final_l:6.2f}%")
                    continue
            except Exception:
                pass
        print(f"{algo.upper():<15} | {'N/A':<16} | {'N/A':<18} | {'N/A':<18}")

    print("=" * 80 + "\n")

    return {
        "algorithms": target_algos,
        "rounds": rounds,
        "total_duration_sec": total_duration,
        "results": benchmark_results,
        "plots": plot_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning Lab - Automated Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=50,
        help="Number of federated training rounds for each baseline",
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=3,
        help="Number of participating clients",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="Local training epochs per client per round",
    )
    parser.add_argument(
        "--algorithms",
        "-a",
        nargs="+",
        default=["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph"],
        help="List of algorithms to benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Unified configuration YAML file path",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Base directory containing per-baseline reports",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/benchmark_comparison",
        help="Output directory for generated comparison plots",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-running baselines whose metrics.json already contains >= rounds",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use fast synthetic tensors for dry-run verification",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [BENCHMARK] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_full_benchmark(
        rounds=args.rounds,
        clients=args.clients,
        epochs=args.epochs,
        algorithms=args.algorithms,
        config_path=args.config,
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        synthetic=args.synthetic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
