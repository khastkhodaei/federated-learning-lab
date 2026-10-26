"""Tests for multi-client federated simulation runner."""

from pathlib import Path

import pytest

from run_simulation import run_federated_simulation


def test_run_federated_simulation_synthetic(tmp_path: Path) -> None:
    # Run a 3-node, 2-round federated simulation with synthetic data on ephemeral port
    results = run_federated_simulation(
        num_clients=3,
        rounds=2,
        epochs=1,
        server_port=0,
        synthetic=True,
        reports_dir=tmp_path / "reports",
    )

    assert results["num_clients"] == 3
    assert results["rounds_completed"] == 2
    assert results["final_model_weights"] is not None
    assert len(results["final_model_weights"]) > 0
    assert len(results["round_history"]) == 2
    for round_info in results["round_history"]:
        assert round_info["num_responses"] == 3
        assert round_info["total_samples"] > 0


@pytest.mark.parametrize(
    "algo",
    ["fedprox", "ditto", "perfedavg", "pfedgraph"],
)
def test_run_simulation_baselines_synthetic(algo: str, tmp_path: Path) -> None:
    results = run_federated_simulation(
        num_clients=2,
        rounds=1,
        epochs=1,
        algorithm=algo,
        server_port=0,
        synthetic=True,
        reports_dir=tmp_path / "reports",
    )

    assert results["algorithm"] == algo
    assert results["num_clients"] == 2
    assert results["rounds_completed"] == 1
    assert len(results["round_history"]) == 1


def test_simulation_reports_isolation(tmp_path: Path) -> None:
    """Simulation with custom reports_dir must generate artifacts only under tmp_path and not touch repo reports/."""
    sim_reports_dir = tmp_path / "reports"
    results = run_federated_simulation(
        num_clients=2,
        rounds=1,
        epochs=1,
        algorithm="fedavg",
        server_port=0,
        synthetic=True,
        reports_dir=sim_reports_dir,
    )

    assert results["rounds_completed"] == 1
    fedavg_dir = sim_reports_dir / "fedavg"
    assert (fedavg_dir / "metrics.json").is_file()
    assert (fedavg_dir / "loss_curve.png").is_file()
    assert (fedavg_dir / "accuracy_curves.png").is_file()
    assert (fedavg_dir / "client_dispersion.png").is_file()
    assert (fedavg_dir / "round_duration.png").is_file()
