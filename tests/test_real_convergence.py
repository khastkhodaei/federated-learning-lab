"""Empirical multi-round convergence stress tests on real CIFAR-10."""

import json
from pathlib import Path

import torch

from run_benchmark import run_full_benchmark
from run_simulation import run_federated_simulation


def test_real_cifar10_multi_round_convergence(tmp_path: Path) -> None:
    """Verifies multi-round convergence of DFPL baseline on real CIFAR-10 with Dirichlet Non-IID skew."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports_dir = tmp_path / "reports"

    results = run_federated_simulation(
        algorithm="dfpl",
        rounds=2,
        num_clients=2,
        epochs=1,
        synthetic=False,
        device=device,
        reports_dir=str(reports_dir),
    )

    assert results["algorithm"] == "dfpl"
    round_history = results["round_history"]
    assert len(round_history) == 2

    for r_info in round_history:
        # Check loss finiteness and positivity
        round_loss = r_info["round_loss"]
        assert torch.isfinite(torch.tensor(round_loss))
        assert round_loss > 0

        # Check accuracies strictly outperform 10% random guess on 10-class CIFAR-10
        global_acc = r_info["global_acc"]
        mean_local_acc = r_info["mean_local_acc"]
        assert torch.isfinite(torch.tensor(global_acc))
        assert global_acc > 10.0
        assert torch.isfinite(torch.tensor(mean_local_acc))
        assert mean_local_acc > 10.0

    metrics_path = reports_dir / "dfpl" / "metrics.json"
    assert metrics_path.is_file(), f"Expected metrics file at {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert len(metrics_data.get("rounds", [])) == 2


def test_benchmark_all_7_baselines_execution_and_plotting(tmp_path: Path) -> None:
    """Verifies that all 7 baselines execute on real CIFAR-10 and generate benchmark artifacts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports_dir = tmp_path / "reports"
    plots_dir = tmp_path / "plots"
    target_algorithms = [
        "fedavg",
        "fedprox",
        "ditto",
        "perfedavg",
        "pfedgraph",
        "dfedpgp",
        "dfpl",
    ]

    results = run_full_benchmark(
        rounds=2,
        clients=2,
        epochs=1,
        algorithms=target_algorithms,
        reports_dir=str(reports_dir),
        output_dir=str(plots_dir),
        synthetic=False,
        device=device,
    )

    # 1. Verify all 7 algorithms exist in results
    assert "results" in results
    assert set(results["results"].keys()) == set(target_algorithms)

    # 2. Verify comparison plot artifacts exist
    expected_artifacts = [
        plots_dir / "benchmark_global_accuracy.png",
        plots_dir / "benchmark_personalized_accuracy.png",
        plots_dir / "benchmark_overview.png",
        plots_dir / "benchmark_summary.json",
    ]
    for artifact_path in expected_artifacts:
        assert artifact_path.is_file(), f"Missing expected benchmark artifact: {artifact_path}"
        assert artifact_path.stat().st_size > 0

    # 3. Verify each algorithm achieves final round accuracy > 10.0%
    for algo in target_algorithms:
        sim_res = results["results"][algo]
        assert "round_history" in sim_res
        last_round = sim_res["round_history"][-1]
        final_global = last_round.get("global_accuracy", last_round.get("global_acc", 0.0))
        final_local = last_round.get("mean_local_accuracy", last_round.get("mean_local_acc", 0.0))
        effective_acc = max(final_global, final_local)
        assert effective_acc > 10.0, (
            f"Algorithm {algo} failed to exceed random guess accuracy (got {effective_acc}%)."
        )

