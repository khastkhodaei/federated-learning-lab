"""Benchmark comparison plotting utilities for Federated Learning Lab.

Reads structured metrics.json across all evaluated baselines (FedAvg, FedProx, Ditto,
Per-FedAvg, pFedGraph) and produces publication-quality multi-algorithm comparison charts:
1. Global Accuracy Comparison (Global test accuracy vs. Round)
2. Personalized / Local Accuracy Comparison (Mean local accuracy vs. Round)
3. Combined Benchmark Overview Figure
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("benchmark.plotting")

# Palette and marker configurations for distinct publication charts
ALGORITHM_STYLES = {
    "fedavg": {
        "color": "#1f77b4",  # Muted Blue
        "marker": "o",
        "linestyle": "-",
        "label": "FedAvg",
    },
    "fedprox": {
        "color": "#ff7f0e",  # Safety Orange
        "marker": "s",
        "linestyle": "--",
        "label": "FedProx (μ=0.01)",
    },
    "ditto": {
        "color": "#2ca02c",  # Cooked Asparagus Green
        "marker": "^",
        "linestyle": "-.",
        "label": "Ditto (λ=0.1)",
    },
    "perfedavg": {
        "color": "#d62728",  # Brick Red
        "marker": "D",
        "linestyle": ":",
        "label": "Per-FedAvg (FO-MAML)",
    },
    "pfedgraph": {
        "color": "#9467bd",  # Muted Purple
        "marker": "v",
        "linestyle": "-",
        "label": "pFedGraph (Graph W)",
    },
}

DEFAULT_STYLE = {
    "color": "#7f7f7f",
    "marker": "x",
    "linestyle": "-",
    "label": "Baseline",
}


def load_benchmark_metrics(
    reports_dir: str | Path = "reports",
    algorithms: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Loads metrics.json records for specified baseline algorithms.

    Args:
        reports_dir: Root directory containing algorithm subdirectories.
        algorithms: List of algorithm names to load (default: all 5 baselines).

    Returns:
        Dict mapping algorithm name to parsed metrics dictionary.
    """
    target_algos = algorithms or ["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph"]
    base_dir = Path(reports_dir)
    results: dict[str, dict[str, Any]] = {}

    for algo in target_algos:
        json_path = base_dir / algo.lower().strip() / "metrics.json"
        if not json_path.is_file():
            logger.warning("metrics.json not found for baseline '%s' at: %s", algo, json_path)
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[algo.lower().strip()] = data
            logger.info(
                "Loaded metrics for '%s' (%d rounds recorded)",
                algo,
                len(data.get("rounds", [])),
            )
        except Exception as e:
            logger.error("Failed to read metrics from %s: %s", json_path, e)

    return results


def plot_global_accuracy_comparison(
    benchmark_data: dict[str, dict[str, Any]],
    output_path: str | Path = "reports/benchmark_comparison/global_accuracy_comparison.png",
) -> str:
    """Renders line chart comparing Global Test Accuracy vs. Round across all baselines.

    Args:
        benchmark_data: Output of load_benchmark_metrics().
        output_path: Destination path for the saved image.

    Returns:
        String path of the generated plot file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)

    for algo, data in benchmark_data.items():
        rounds_data = data.get("rounds", [])
        if not rounds_data:
            continue

        rounds = [int(r.get("round", i + 1)) for i, r in enumerate(rounds_data)]
        global_accs = [
            float(r.get("global_test_accuracy", 0.0)) * 100.0 for r in rounds_data
        ]

        style = ALGORITHM_STYLES.get(algo, DEFAULT_STYLE)
        max_acc = max(global_accs) if global_accs else 0.0
        label_text = f"{style['label']} (Max: {max_acc:.1f}%)"

        ax.plot(
            rounds,
            global_accs,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=5,
            label=label_text,
            alpha=0.9,
        )

    ax.set_title(
        "Global Balanced Test Accuracy vs. Federated Round\nCIFAR-10 (Dirichlet Non-IID, α = 0.1)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Federated Round", fontsize=10, fontweight="medium")
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=10, fontweight="medium")
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved global accuracy comparison plot to: %s", out_file)
    return str(out_file)


def plot_local_accuracy_comparison(
    benchmark_data: dict[str, dict[str, Any]],
    output_path: str | Path = "reports/benchmark_comparison/local_accuracy_comparison.png",
) -> str:
    """Renders line chart comparing Mean Local/Personalized Test Accuracy vs. Round across all baselines.

    Args:
        benchmark_data: Output of load_benchmark_metrics().
        output_path: Destination path for the saved image.

    Returns:
        String path of the generated plot file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)

    for algo, data in benchmark_data.items():
        rounds_data = data.get("rounds", [])
        if not rounds_data:
            continue

        rounds = [int(r.get("round", i + 1)) for i, r in enumerate(rounds_data)]
        local_accs = [
            float(r.get("mean_local_test_accuracy", 0.0)) * 100.0 for r in rounds_data
        ]

        style = ALGORITHM_STYLES.get(algo, DEFAULT_STYLE)
        max_local = max(local_accs) if local_accs else 0.0
        label_text = f"{style['label']} (Max: {max_local:.1f}%)"

        ax.plot(
            rounds,
            local_accs,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=5,
            label=label_text,
            alpha=0.9,
        )

    ax.set_title(
        "Mean Personalized / Local Test Accuracy vs. Federated Round\nCIFAR-10 (Dirichlet Non-IID, α = 0.1)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Federated Round", fontsize=10, fontweight="medium")
    ax.set_ylabel("Mean Local Test Accuracy (%)", fontsize=10, fontweight="medium")
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved local accuracy comparison plot to: %s", out_file)
    return str(out_file)


def plot_benchmark_summary(
    reports_dir: str | Path = "reports",
    output_dir: str | Path = "reports/benchmark_comparison",
    algorithms: list[str] | None = None,
) -> dict[str, str]:
    """Generates all comparison plots from available baseline metrics.

    Args:
        reports_dir: Directory containing per-algorithm metrics.
        output_dir: Output directory for comparison plots.
        algorithms: Optional algorithm whitelist.

    Returns:
        Dict mapping plot keys to generated file paths.
    """
    target_out = Path(output_dir)
    target_out.mkdir(parents=True, exist_ok=True)

    benchmark_data = load_benchmark_metrics(reports_dir=reports_dir, algorithms=algorithms)
    if not benchmark_data:
        logger.warning("No benchmark data found to plot in '%s'", reports_dir)
        return {}

    global_plot = plot_global_accuracy_comparison(
        benchmark_data, target_out / "global_accuracy_comparison.png"
    )
    local_plot = plot_local_accuracy_comparison(
        benchmark_data, target_out / "local_accuracy_comparison.png"
    )

    # 3. Two-panel combined comparison figure
    combined_plot = target_out / "benchmark_summary.png"
    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(15.0, 5.2), dpi=180)

    for algo, data in benchmark_data.items():
        rounds_data = data.get("rounds", [])
        if not rounds_data:
            continue

        rounds = [int(r.get("round", i + 1)) for i, r in enumerate(rounds_data)]
        g_accs = [float(r.get("global_test_accuracy", 0.0)) * 100.0 for r in rounds_data]
        l_accs = [float(r.get("mean_local_test_accuracy", 0.0)) * 100.0 for r in rounds_data]

        style = ALGORITHM_STYLES.get(algo, DEFAULT_STYLE)
        lbl = style["label"]

        ax_g.plot(
            rounds, g_accs, marker=style["marker"], color=style["color"],
            linestyle=style["linestyle"], linewidth=1.8, markersize=4,
            label=f"{lbl} ({max(g_accs):.1f}%)", alpha=0.9,
        )
        ax_l.plot(
            rounds, l_accs, marker=style["marker"], color=style["color"],
            linestyle=style["linestyle"], linewidth=1.8, markersize=4,
            label=f"{lbl} ({max(l_accs):.1f}%)", alpha=0.9,
        )

    ax_g.set_title("Global Test Accuracy (Balanced)", fontsize=11, fontweight="bold")
    ax_g.set_xlabel("Federated Round", fontsize=10)
    ax_g.set_ylabel("Global Accuracy (%)", fontsize=10)
    ax_g.set_ylim(-2, 102)
    ax_g.grid(True, linestyle="--", alpha=0.55)
    ax_g.legend(frameon=True, fontsize=8.5, loc="lower right")

    ax_l.set_title("Mean Personalized / Local Accuracy", fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Federated Round", fontsize=10)
    ax_l.set_ylabel("Local Accuracy (%)", fontsize=10)
    ax_l.set_ylim(-2, 102)
    ax_l.grid(True, linestyle="--", alpha=0.55)
    ax_l.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.suptitle("FL Benchmark Comparison across 5 Baselines on CIFAR-10 (α = 0.1)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(combined_plot, bbox_inches="tight")
    plt.close(fig)

    return {
        "global_accuracy_comparison": global_plot,
        "local_accuracy_comparison": local_plot,
        "benchmark_summary": str(combined_plot),
    }
