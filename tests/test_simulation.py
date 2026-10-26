import pytest
import torch

from run_simulation import (
    run_ditto_simulation,
    run_federated_simulation,
    run_fedprox_simulation,
    run_perfedavg_simulation,
    run_pfedgraph_simulation,
)


def test_run_federated_simulation_synthetic() -> None:
    # Run a 3-node, 2-round federated simulation with synthetic data on ephemeral port
    results = run_federated_simulation(
        num_clients=3,
        rounds=2,
        epochs=1,
        server_port=0,
        synthetic=True,
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
def test_run_simulation_baselines_synthetic(algo: str) -> None:
    results = run_federated_simulation(
        num_clients=2,
        rounds=1,
        epochs=1,
        algorithm=algo,
        server_port=0,
        synthetic=True,
    )

    assert results["algorithm"] == algo
    assert results["num_clients"] == 2
    assert results["rounds_completed"] == 1
    assert len(results["round_history"]) == 1
