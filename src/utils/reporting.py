"""Reporting and metric visualization utilities for Federated Learning baselines.

Provides automated structured reporting (metrics.json) and publication-ready
visual progression charts (training_curves.png) organized per baseline algorithm.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_client_message_metrics(message: str) -> dict[str, float]:
    """Parses metric key-values (loss, accuracy, etc.) from a client response message string.

    Supports JSON-encoded strings or 'key=value' tokens.
    """
    if not message or not isinstance(message, str):
        return {}

    # 1. Attempt JSON parsing
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            return {
                k: float(v)
                for k, v in data.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Key-value regex parsing (e.g. "loss=1.234 accuracy=0.85")
    metrics: dict[str, float] = {}
    pattern = re.compile(r"([a-zA-Z_]+)=([0-9\.\-]+)")
    for match in pattern.finditer(message):
        k, v = match.group(1), match.group(2)
        try:
            metrics[k] = float(v)
        except ValueError:
            continue
    return metrics


def save_baseline_report(
    baseline_name: str,
    round_history: list[dict[str, Any]],
    output_dir: str = "reports",
) -> dict[str, str]:
    """Saves structured metrics.json and training_curves.png under reports/<baseline>/.

    Args:
        baseline_name: Identifier for the algorithm (e.g. 'fedavg', 'fedprox').
        round_history: List of per-round metric dicts containing 'round', 'duration_seconds',
                       'weighted_loss', 'total_samples', etc.
        output_dir: Base directory for reports (default: 'reports').

    Returns:
        Dict mapping artifact keys ('metrics_json', 'training_curves') to their absolute file paths.
    """
    clean_baseline = baseline_name.lower().strip()
    target_dir = Path(output_dir) / clean_baseline
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "metrics.json"
    plot_path = target_dir / "training_curves.png"

    # 1. Build and serialize structured JSON
    from src.utils.hardware import get_hardware_summary
    report_data = {
        "baseline": clean_baseline,
        "hardware": get_hardware_summary(),
        "timestamp": datetime.now().isoformat(),
        "total_rounds": len(round_history),
        "rounds": round_history,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 2. Render 2-panel visual progression plot
    rounds = [int(r.get("round", i + 1)) for i, r in enumerate(round_history)]
    losses = [float(r.get("weighted_loss", 0.0)) for r in round_history]
    durations = [float(r.get("duration_seconds", 0.0)) for r in round_history]

    fig, (ax_loss, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left Panel: Loss Progression
    ax_loss.plot(rounds, losses, marker="o", color="#1f77b4", linewidth=2.2, markersize=6, label=f"{clean_baseline.upper()} Loss")
    ax_loss.set_title(f"{clean_baseline.upper()} - Training Loss vs. Round", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Federated Round", fontsize=10)
    ax_loss.set_ylabel("Weighted Cross-Entropy Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    if rounds:
        ax_loss.set_xticks(rounds)
    ax_loss.legend(frameon=True)

    # Right Panel: Round Latency / Duration
    ax_time.bar(rounds, durations, color="#2ca02c", alpha=0.75, width=0.5, edgecolor="#1b611b", label="Round Duration (s)")
    ax_time.set_title(f"{clean_baseline.upper()} - Round Duration & Communication", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Federated Round", fontsize=10)
    ax_time.set_ylabel("Duration (seconds)", fontsize=10)
    ax_time.grid(True, linestyle="--", alpha=0.6, axis="y")
    if rounds:
        ax_time.set_xticks(rounds)
    ax_time.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "metrics_json": str(json_path),
        "training_curves": str(plot_path),
    }
