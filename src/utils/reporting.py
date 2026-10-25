"""Reporting and metric visualization utilities for Federated Learning baselines.

Provides automated structured reporting (metrics.json) and publication-ready
visual progression charts (loss_curve.png, round_duration.png, accuracy_curves.png,
client_dispersion.png) organized per baseline algorithm.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    """Saves structured metrics.json, loss_curve.png, round_duration.png,

    accuracy_curves.png, and client_dispersion.png under reports/<baseline>/.

    Args:
        baseline_name: Identifier for the algorithm (e.g. 'fedavg', 'fedprox').
        round_history: List of per-round metric dicts containing 'round', 'duration_seconds',
                       'weighted_loss', 'total_samples', etc.
        output_dir: Base directory for reports (default: 'reports').

    Returns:
        Dict mapping artifact keys ('metrics_json', 'loss_curve', 'round_duration',
        'accuracy_curves', 'client_dispersion') to their absolute file paths.
    """
    clean_baseline = baseline_name.lower().strip()
    target_dir = Path(output_dir) / clean_baseline
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "metrics.json"
    loss_path = target_dir / "loss_curve.png"
    duration_path = target_dir / "round_duration.png"
    acc_path = target_dir / "accuracy_curves.png"
    dispersion_path = target_dir / "client_dispersion.png"

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

    rounds = [int(r.get("round", i + 1)) for i, r in enumerate(round_history)]
    losses = [float(r.get("weighted_loss", 0.0)) for r in round_history]
    durations = [float(r.get("duration_seconds", 0.0)) for r in round_history]
    global_accs = [
        float(r.get("global_test_accuracy", 0.0)) * 100 for r in round_history
    ]
    local_mean_accs = [
        float(r.get("mean_local_test_accuracy", 0.0)) * 100 for r in round_history
    ]
    local_std_accs = [
        float(r.get("std_local_test_accuracy", 0.0)) * 100 for r in round_history
    ]

    # 2. Render Plot 1: Training Loss vs. Round
    fig_loss, ax_l = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax_l.plot(
        rounds,
        losses,
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=6,
        label=f"{clean_baseline.upper()} Loss",
    )
    ax_l.set_title(
        f"{clean_baseline.upper()} - Training Loss vs. Round",
        fontsize=12,
        fontweight="bold",
    )
    ax_l.set_xlabel("Federated Round", fontsize=10)
    ax_l.set_ylabel("Weighted Cross-Entropy Loss", fontsize=10)
    ax_l.grid(True, linestyle="--", alpha=0.6)
    if rounds:
        ax_l.set_xticks(rounds)
    ax_l.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close(fig_loss)

    # 3. Render Plot 2: Round Duration & Communication
    fig_dur, ax_d = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax_d.bar(
        rounds,
        durations,
        color="#2ca02c",
        alpha=0.75,
        width=0.5,
        edgecolor="#1b611b",
        label="Round Duration (s)",
    )
    ax_d.set_title(
        f"{clean_baseline.upper()} - Round Duration & Communication",
        fontsize=12,
        fontweight="bold",
    )
    ax_d.set_xlabel("Federated Round", fontsize=10)
    ax_d.set_ylabel("Duration (seconds)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.6, axis="y")
    if rounds:
        ax_d.set_xticks(rounds)
    ax_d.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(duration_path, bbox_inches="tight")
    plt.close(fig_dur)

    # 4. Render Plot 3: Accuracy Curves (Global vs. Local)
    fig_acc, ax_a = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax_a.plot(
        rounds,
        global_accs,
        marker="s",
        color="#1f77b4",
        linewidth=2.2,
        markersize=6,
        label="Global Test Acc (Balanced)",
    )
    ax_a.plot(
        rounds,
        local_mean_accs,
        marker="^",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="Mean Local Test Acc",
    )

    if any(s > 0 for s in local_std_accs) and rounds:
        lower = np.maximum(
            0.0, np.array(local_mean_accs) - np.array(local_std_accs)
        )
        upper = np.minimum(
            100.0, np.array(local_mean_accs) + np.array(local_std_accs)
        )
        ax_a.fill_between(
            rounds,
            lower,
            upper,
            color="#ff7f0e",
            alpha=0.18,
            label="Local Acc ±1 Std",
        )

    ax_a.set_title(
        f"{clean_baseline.upper()} - Test Accuracy Progression",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.set_xlabel("Federated Round", fontsize=10)
    ax_a.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
    ax_a.set_ylim(-2, 102)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    if rounds:
        ax_a.set_xticks(rounds)
    ax_a.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close(fig_acc)

    # 5. Render Plot 4: Client Accuracy Dispersion & Fairness
    fig_disp, ax_disp = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    client_box_data = []
    for r in round_history:
        c_metrics = r.get("client_metrics", {})
        accs = [
            float(v.get("test_accuracy", 0.0)) * 100
            for v in c_metrics.values()
            if isinstance(v, dict) and "test_accuracy" in v
        ]
        if not accs:
            mean_a = float(r.get("mean_local_test_accuracy", 0.0)) * 100
            w10 = float(r.get("worst_10_local_accuracy", mean_a / 100.0)) * 100
            accs = [w10, mean_a]
        client_box_data.append(accs)

    if client_box_data and any(len(b) > 0 for b in client_box_data):
        ax_disp.boxplot(
            client_box_data,
            positions=rounds if rounds else [1],
            widths=0.4,
            patch_artist=True,
            boxprops={"facecolor": "#9467bd", "color": "#5c357d", "alpha": 0.6},
            medianprops={"color": "#1f77b4", "linewidth": 2},
        )
    ax_disp.set_title(
        f"{clean_baseline.upper()} - Client Accuracy Dispersion & Fairness",
        fontsize=12,
        fontweight="bold",
    )
    ax_disp.set_xlabel("Federated Round", fontsize=10)
    ax_disp.set_ylabel("Client Local Accuracy (%)", fontsize=10)
    ax_disp.set_ylim(-2, 102)
    ax_disp.grid(True, linestyle="--", alpha=0.6)
    if rounds:
        ax_disp.set_xticks(rounds)
    plt.tight_layout()
    plt.savefig(dispersion_path, bbox_inches="tight")
    plt.close(fig_disp)


    return {
        "metrics_json": str(json_path),
        "loss_curve": str(loss_path),
        "round_duration": str(duration_path),
        "accuracy_curves": str(acc_path),
        "client_dispersion": str(dispersion_path),
    }
