#!/usr/bin/env python3
"""Automated Multi-Baseline Benchmark Orchestrator for Federated Learning Lab.

Sequentially executes federated learning baselines under identical experimental conditions:
- FedAvg (McMahan et al., 2017)
- FedProx (Li et al., 2020) [mu=0.01]
- Ditto (Li et al., 2021) [lambda=0.1, pers_epochs=1]
- Per-FedAvg (Fallah et al., 2020) [alpha_lr=0.01, beta_lr=0.005]
- pFedGraph (Chen et al., 2022) [graph_alpha=0.1, graph_lam=0.01]
- DFedPGP (Directed Partial Gradient Push, out_degree=1, time_varying=True)
- DFPL (Prototype Alignment, lambda=0.1)
- Local-Only (Private Local SGD Reference)

Saves per-baseline metrics to reports/<algorithm>/metrics.json and generates combined
cross-algorithm comparison plots in reports/benchmark_comparison/.
"""

import argparse
from collections.abc import Sequence
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

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
    "dfedpgp": {
        "description": "DFedPGP (Directed Partial Gradient Push, out_degree=1)",
        "params": {"out_degree": 1, "time_varying": True},
    },
    "dfpl": {
        "description": "DFPL (Prototype Alignment, lambda=0.1)",
        "params": {"lambda_proto": 0.1},
    },
    "local_only": {
        "description": "Local-Only (Private Local SGD Reference, Zero Aggregation)",
        "params": {},
    },
}


def parse_seeds(raw_seeds: Sequence[str] | str | None) -> list[int] | None:
    """Parses --seeds argument supporting comma-separated strings or token lists."""
    if raw_seeds is None:
        return None
    tokens = [raw_seeds] if isinstance(raw_seeds, str) else list(raw_seeds)
    parsed: list[int] = []
    for token in tokens:
        for part in str(token).split(","):
            stripped = part.strip()
            if not stripped:
                continue
            parsed.append(int(stripped))
    return parsed or None


def summarize_multi_seed_runs(
    seed_round_histories: Sequence[Sequence[dict[str, Any]]],
    last_n: int = 10,
    scale: float = 100.0,
) -> dict[str, Any]:
    """Computes Mean and Std across multiple random seeds for key accuracy metrics.

    Expects per-seed round lists (in ``metrics.json`` format with accuracies in ``[0, 1]``
    when ``scale=100.0``) and summarizes Final Round as well as Last-N-Round Mean metrics.
    """
    valid_histories = [list(h) for h in seed_round_histories if h]
    if not valid_histories:
        return {}

    metrics_accum: dict[str, list[float]] = {
        "global_test_accuracy": [],
        "last_10_global_accuracy": [],
        "global_on_local_accuracy": [],
        "last_10_global_on_local_accuracy": [],
        "local_test_accuracy": [],
        "last_10_local_accuracy": [],
        "majority_class_accuracy": [],
    }

    for rounds_list in valid_histories:
        window = min(max(1, last_n), len(rounds_list))
        g_vals = [float(r.get("global_test_accuracy", 0.0)) * scale for r in rounds_list]
        gol_vals = [float(r.get("global_on_local_accuracy", 0.0)) * scale for r in rounds_list]
        l_vals = [
            float(r.get("local_test_accuracy", r.get("mean_local_test_accuracy", 0.0))) * scale
            for r in rounds_list
        ]
        maj_vals = [float(r.get("majority_class_accuracy", 0.0)) * scale for r in rounds_list]

        metrics_accum["global_test_accuracy"].append(g_vals[-1])
        metrics_accum["last_10_global_accuracy"].append(sum(g_vals[-window:]) / window)
        metrics_accum["global_on_local_accuracy"].append(gol_vals[-1])
        metrics_accum["last_10_global_on_local_accuracy"].append(sum(gol_vals[-window:]) / window)
        metrics_accum["local_test_accuracy"].append(l_vals[-1])
        metrics_accum["last_10_local_accuracy"].append(sum(l_vals[-window:]) / window)
        metrics_accum["majority_class_accuracy"].append(maj_vals[-1])

    summary: dict[str, Any] = {"num_seeds": len(valid_histories)}
    for key, vals in metrics_accum.items():
        arr = np.array(vals, dtype=float)
        summary[f"{key}_mean"] = round(float(np.mean(arr)), 4)
        summary[f"{key}_std"] = round(float(np.std(arr)), 4)
    return summary


def _average_seed_metrics_files(
    seed_files: Sequence[Path],
    dest_file: Path,
) -> dict[str, Any] | None:
    """Averages per-round metrics across multiple seed metrics.json files for plotting."""
    loaded: list[dict[str, Any]] = []
    for fpath in seed_files:
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8") as f:
                loaded.append(json.load(f))
    if not loaded:
        return None

    base = dict(loaded[0])
    num_rounds = min(len(d.get("rounds", [])) for d in loaded)
    avg_rounds: list[dict[str, Any]] = []
    scalar_keys = (
        "duration_seconds",
        "total_samples",
        "weighted_loss",
        "global_test_accuracy",
        "global_on_local_accuracy",
        "local_test_accuracy",
        "mean_local_test_accuracy",
        "majority_class_accuracy",
        "std_local_test_accuracy",
        "worst_10_local_accuracy",
    )
    for r_idx in range(num_rounds):
        template = dict(loaded[0]["rounds"][r_idx])
        for key in scalar_keys:
            if key in template:
                vals = [float(d["rounds"][r_idx].get(key, 0.0)) for d in loaded]
                template[key] = float(np.mean(vals))
        avg_rounds.append(template)

    base["rounds"] = avg_rounds
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    return base


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
    seeds: list[int] | None = None,
    partition_seed: int | None = None,
) -> dict[str, Any]:
    """Orchestrates sequential execution of all baseline simulations and plots comparisons.

    Args:
        rounds: Number of federated training rounds per baseline.
        clients: Number of participating clients.
        epochs: Local training epochs per client.
        algorithms: List of algorithm names to run (default: all federated baselines).
        config_path: Unified YAML config path.
        reports_dir: Directory containing per-baseline reports.
        output_dir: Directory to save combined comparison plots.
        skip_existing: If True, skips runs where metrics.json already contains >= rounds.
        synthetic: If True, uses fast synthetic tensors.
        device: Target compute device ('cuda' or 'cpu').
        seeds: Optional list of random seeds for multi-seed Mean +/- Std benchmarking.
        partition_seed: Fixed dataset partition seed held constant across training seeds.

    Returns:
        Dictionary summarizing execution status, per-algorithm results, and generated plots.
    """
    default_algos = [
        "fedavg",
        "fedprox",
        "ditto",
        "perfedavg",
        "pfedgraph",
        "dfedpgp",
        "dfpl",
    ]
    target_algos = algorithms or default_algos
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    eff_seeds: list[int | None] = list(seeds) if seeds else [None]
    multi_seed_mode = len(eff_seeds) > 1
    eff_partition_seed = partition_seed if partition_seed is not None else 42

    logger.info("=" * 75)
    logger.info("FEDERATED LEARNING LAB - COMPLETE MULTI-BASELINE BENCHMARK SUITE")
    logger.info("Algorithms to evaluate: %s", ", ".join(target_algos).upper())
    logger.info(
        "Conditions: Rounds=%d, Clients=%d, Local Epochs=%d, Seeds=%s, PartitionSeed=%d",
        rounds,
        clients,
        epochs,
        eff_seeds if multi_seed_mode else "config-default",
        eff_partition_seed,
    )
    logger.info("=" * 75)

    benchmark_results: dict[str, Any] = {}
    multi_seed_summary: dict[str, dict[str, Any]] = {}
    total_start = time.time()

    for idx, algo in enumerate(target_algos, start=1):
        clean_algo = algo.lower().strip()
        spec = BENCHMARK_CONFIGS.get(clean_algo, {"description": clean_algo, "params": {}})
        seed_metrics_files: list[Path] = []
        seed_round_histories: list[list[dict[str, Any]]] = []

        for s_idx, run_seed in enumerate(eff_seeds, start=1):
            seed_reports_dir = (
                reports_path / f"seed_{run_seed}" if multi_seed_mode else reports_path
            )
            metrics_file = seed_reports_dir / clean_algo / "metrics.json"

            if skip_existing and metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    cached_rounds = len(cached_data.get("rounds", []))
                    if cached_rounds >= rounds:
                        logger.info(
                            "[%d/%d] Skipping '%s' (seed=%s): metrics.json exists with %d rounds.",
                            idx,
                            len(target_algos),
                            clean_algo.upper(),
                            run_seed,
                            cached_rounds,
                        )
                        seed_metrics_files.append(metrics_file)
                        seed_round_histories.append(cached_data.get("rounds", []))
                        if not multi_seed_mode:
                            benchmark_results[clean_algo] = cached_data
                        continue
                except Exception:
                    pass

            logger.info("")
            logger.info("*" * 65)
            seed_suffix = f" [Seed {s_idx}/{len(eff_seeds)}: {run_seed}]" if multi_seed_mode else ""
            logger.info(
                "[%d/%d] Executing Baseline: %s%s",
                idx,
                len(target_algos),
                spec["description"],
                seed_suffix,
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
                reports_dir=seed_reports_dir,
                server_port=0,
                synthetic=synthetic,
                device=device,
                seed=run_seed,
                partition_seed=eff_partition_seed,
                **run_params,
            )

            sim_duration = time.time() - sim_start
            logger.info(
                "[%d/%d] Completed '%s'%s in %.2f minutes (%.1fs).",
                idx,
                len(target_algos),
                clean_algo.upper(),
                seed_suffix,
                sim_duration / 60.0,
                sim_duration,
            )
            if metrics_file.is_file():
                seed_metrics_files.append(metrics_file)
                with open(metrics_file, "r", encoding="utf-8") as f:
                    loaded_seed_data = json.load(f)
                seed_round_histories.append(loaded_seed_data.get("rounds", []))
            if not multi_seed_mode:
                benchmark_results[clean_algo] = result

        if multi_seed_mode and seed_metrics_files:
            avg_file = reports_path / clean_algo / "metrics.json"
            avg_data = _average_seed_metrics_files(seed_metrics_files, avg_file)
            if avg_data is not None:
                benchmark_results[clean_algo] = avg_data
        if seed_round_histories:
            multi_seed_summary[clean_algo] = summarize_multi_seed_runs(seed_round_histories)

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

    # Enrich benchmark_summary.json with multi-seed Mean/Std summary when available
    summary_json_path = plot_artifacts.get("benchmark_summary_json")
    if summary_json_path and Path(summary_json_path).is_file():
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
        for algo_key, stats in multi_seed_summary.items():
            if algo_key in summary_payload:
                summary_payload[algo_key].update(stats)
            else:
                summary_payload[algo_key] = stats
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    for k, p in plot_artifacts.items():
        logger.info("  - Generated %s: %s", k, p)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Print final summary table (Last-10-Round Mean + Final Round, Mean +/- Std across seeds)
    print("\n" + "=" * 105)
    seed_label = f"{len(eff_seeds)} SEEDS" if multi_seed_mode else "SINGLE SEED"
    print(f"       BENCHMARK COMPARISON SUMMARY ({rounds} ROUNDS, {seed_label}, DIRICHLET alpha = 0.1)")
    print("=" * 105)
    print(
        f"{'Algorithm':<13} | {'Last-10 Global':<17} | {'Final Global':<17} | "
        f"{'Last-10 G-on-L':<17} | {'Last-10 Local':<17} | {'Floor':<7}"
    )
    print("-" * 105)

    for algo in target_algos:
        stats = multi_seed_summary.get(algo.lower().strip())
        if stats:
            if multi_seed_mode:
                l10_g = f"{stats['last_10_global_accuracy_mean']:.2f}±{stats['last_10_global_accuracy_std']:.2f}%"
                fin_g = f"{stats['global_test_accuracy_mean']:.2f}±{stats['global_test_accuracy_std']:.2f}%"
                l10_gol = (
                    f"{stats['last_10_global_on_local_accuracy_mean']:.2f}±"
                    f"{stats['last_10_global_on_local_accuracy_std']:.2f}%"
                )
                l10_l = f"{stats['last_10_local_accuracy_mean']:.2f}±{stats['last_10_local_accuracy_std']:.2f}%"
            else:
                l10_g = f"{stats['last_10_global_accuracy_mean']:6.2f}%"
                fin_g = f"{stats['global_test_accuracy_mean']:6.2f}%"
                l10_gol = f"{stats['last_10_global_on_local_accuracy_mean']:6.2f}%"
                l10_l = f"{stats['last_10_local_accuracy_mean']:6.2f}%"
            floor_s = f"{stats['majority_class_accuracy_mean']:5.2f}%"
            print(
                f"{algo.upper():<13} | {l10_g:<17} | {fin_g:<17} | "
                f"{l10_gol:<17} | {l10_l:<17} | {floor_s:<7}"
            )
            continue
        print(f"{algo.upper():<13} | {'N/A':<17} | {'N/A':<17} | {'N/A':<17} | {'N/A':<17} | {'N/A':<7}")

    print("=" * 105 + "\n")

    return {
        "algorithms": target_algos,
        "rounds": rounds,
        "seeds": eff_seeds if multi_seed_mode else None,
        "total_duration_sec": total_duration,
        "results": benchmark_results,
        "multi_seed_summary": multi_seed_summary,
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
        default=["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph", "dfedpgp", "dfpl"],
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Optional random seeds for multi-seed benchmarking (e.g. --seeds 42,43,44)",
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=42,
        help="Fixed dataset partition seed held constant across training seeds",
    )

    args = parser.parse_args()
    parsed_seeds = parse_seeds(args.seeds)

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
        seeds=parsed_seeds,
        partition_seed=args.partition_seed,
    )


if __name__ == "__main__":
    main()
