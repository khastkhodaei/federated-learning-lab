"""Tests for multi-client federated simulation runner."""

import json
from pathlib import Path

import pytest
import torch

from run_simulation import run_federated_simulation
from src.utils.serialization import bytes_to_vector


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
    ["fedprox", "ditto", "perfedavg", "pfedgraph", "dfedpgp", "dfpl"],
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


def test_ditto_matches_fedavg_round1_under_multi_client_simulation(tmp_path: Path) -> None:
    """Verifies Ditto Phase 2 RNG isolation keeps Round 1 global weights identical to FedAvg."""
    res_fedavg = run_federated_simulation(
        num_clients=2,
        rounds=1,
        epochs=1,
        algorithm="fedavg",
        server_port=0,
        synthetic=True,
        device="cpu",
        seed=42,
        partition_seed=42,
        reports_dir=tmp_path / "fedavg_reports",
    )
    res_ditto = run_federated_simulation(
        num_clients=2,
        rounds=1,
        epochs=1,
        algorithm="ditto",
        server_port=0,
        synthetic=True,
        device="cpu",
        seed=42,
        partition_seed=42,
        reports_dir=tmp_path / "ditto_reports",
    )

    v_fedavg = bytes_to_vector(res_fedavg["final_model_weights"])
    v_ditto = bytes_to_vector(res_ditto["final_model_weights"])
    assert torch.allclose(v_fedavg, v_ditto, atol=1e-5)


def test_local_only_preserves_independent_client_weights_and_logs_metrics(
    tmp_path: Path,
) -> None:
    """Verifies local_only performs zero global aggregation and logs standardized evaluation metrics."""
    sim_reports_dir = tmp_path / "local_only_reports"
    results = run_federated_simulation(
        num_clients=2,
        rounds=2,
        epochs=1,
        algorithm="local_only",
        server_port=0,
        synthetic=True,
        device="cpu",
        seed=42,
        reports_dir=sim_reports_dir,
    )

    # Zero global aggregation -> global model parameters remain untouched at initial weights
    init_vec = bytes_to_vector(results["initial_model_weights"])
    final_vec = bytes_to_vector(results["final_model_weights"])
    assert torch.equal(init_vec, final_vec)

    metrics_file = sim_reports_dir / "local_only" / "metrics.json"
    assert metrics_file.is_file()
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["rounds"]) == 2
    for r_entry in data["rounds"]:
        assert "global_test_accuracy" in r_entry
        assert "global_on_local_accuracy" in r_entry
        assert "local_test_accuracy" in r_entry
        assert "majority_class_accuracy" in r_entry
        assert r_entry["local_accuracy_type"] == "personalized_pre_agg"
        assert r_entry["majority_class_accuracy"] > 0.0
