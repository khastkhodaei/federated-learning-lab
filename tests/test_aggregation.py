import torch
import pytest

from src.models.cnn import SimpleCNN
from src.server.aggregation import fedavg_vectors, fedavg_bytes
from src.utils.serialization import model_to_bytes, bytes_to_vector, vector_to_bytes


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
