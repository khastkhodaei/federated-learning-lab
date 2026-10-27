import pytest
import torch

from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.models.cnn import SimpleCNN
from src.server.aggregation import fedavg_bytes, fedavg_vectors
from src.utils.serialization import bytes_to_vector, model_to_bytes


def test_fedavg_vectors_weighted_average() -> None:
    # 2 clients with different weights and sample sizes
    v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    v2 = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)

    # Client 1 has 10 samples, Client 2 has 30 samples (total 40)
    # Expected: (10*v1 + 30*v2) / 40 = 0.25 * [1, 2, 3] + 0.75 * [5, 6, 7] = [4.0, 5.0, 6.0]
    result = fedavg_vectors([(v1, 10), (v2, 30)])

    expected = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_fedavg_vectors_empty_or_zero_samples() -> None:
    # Empty list
    with pytest.raises(ValueError, match="Cannot aggregate empty updates"):
        fedavg_vectors([])

    # Total samples is 0 -> fall back to uniform unweighted mean
    v1 = torch.tensor([2.0, 4.0], dtype=torch.float32)
    v2 = torch.tensor([4.0, 6.0], dtype=torch.float32)
    result = fedavg_vectors([(v1, 0), (v2, 0)])
    assert torch.allclose(result, torch.tensor([3.0, 5.0], dtype=torch.float32))


def test_fedavg_bytes_integration() -> None:
    m1 = SimpleCNN(seed=1)
    m2 = SimpleCNN(seed=2)

    b1 = model_to_bytes(m1)
    b2 = model_to_bytes(m2)

    agg_bytes = fedavg_bytes([(b1, 50), (b2, 50)])
    agg_vec = bytes_to_vector(agg_bytes)

    expected_vec = 0.5 * bytes_to_vector(b1) + 0.5 * bytes_to_vector(b2)
    assert torch.allclose(agg_vec, expected_vec, atol=1e-6)


def test_fedavg_hand_computed_weights_100_300() -> None:
    v1 = torch.tensor([2.0, -4.0, 8.0, 10.0], dtype=torch.float32)
    v2 = torch.tensor([6.0, 4.0, -12.0, 2.0], dtype=torch.float32)

    result = fedavg_vectors([(v1, 100), (v2, 300)])
    hand_computed = 0.25 * v1 + 0.75 * v2

    assert torch.allclose(result, hand_computed, atol=1e-6)
    assert torch.equal(result, torch.tensor([5.0, 2.0, -7.0, 4.0], dtype=torch.float32))


def test_fedavg_single_client_identity() -> None:
    v1 = torch.tensor([3.14159, -2.71828, 42.0, 0.0], dtype=torch.float32)

    result = fedavg_vectors([(v1, 250)])
    assert torch.equal(result, v1)


def test_fedavg_weights_sum_to_one() -> None:
    sample_counts = [120, 380, 500]
    total_samples = sum(sample_counts)
    proportions = [s / total_samples for s in sample_counts]
    assert sum(proportions) == pytest.approx(1.0)

    v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    v2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    v3 = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)

    result = fedavg_vectors([(v1, 120), (v2, 380), (v3, 500)])
    manual_weighted = 0.12 * v1 + 0.38 * v2 + 0.50 * v3

    assert torch.allclose(result, manual_weighted, atol=1e-6)


def test_fedavg_strategy_aggregate_sample_weighted() -> None:
    m1 = SimpleCNN(seed=10)
    m2 = SimpleCNN(seed=20)
    b1 = model_to_bytes(m1)
    b2 = model_to_bytes(m2)

    strategy = FedAvgServerStrategy()
    global_model = SimpleCNN(seed=999)

    agg_bytes = strategy.aggregate(
        global_model=global_model,
        client_updates=[(b1, 100), (b2, 300)],
        round_id=1,
    )

    expected_vec = 0.25 * bytes_to_vector(b1) + 0.75 * bytes_to_vector(b2)
    assert torch.allclose(bytes_to_vector(agg_bytes), expected_vec, atol=1e-6)
    assert torch.allclose(bytes_to_vector(model_to_bytes(global_model)), expected_vec, atol=1e-6)
