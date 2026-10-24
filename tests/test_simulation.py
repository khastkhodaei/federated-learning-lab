import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_simulation import run_federated_simulation


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
