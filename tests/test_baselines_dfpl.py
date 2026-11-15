import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.dfpl import (
    aggregate_prototypes,
    bytes_to_prototypes,
    compute_local_prototypes,
    prototypes_to_bytes,
    prototypes_to_tensor,
)
from src.models.cnn import SimpleCNN


def test_compute_local_prototypes_synthetic() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    model.train()

    # Synthetic dataset with only classes {0, 1}; class 2 is absent
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    prototypes, sample_counts = compute_local_prototypes(model, dataloader)

    assert set(prototypes.keys()) == {0, 1}
    assert sample_counts == {0: 4, 1: 4}
    for c in (0, 1):
        assert isinstance(prototypes[c], torch.Tensor)
        assert prototypes[c].shape == (128,)

    # Model training mode must be preserved
    assert model.training is True


def test_compute_local_prototypes_empty_dataloader() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    dataset = TensorDataset(
        torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)
    )
    dataloader = DataLoader(dataset, batch_size=4)

    prototypes, sample_counts = compute_local_prototypes(model, dataloader)
    assert prototypes == {}
    assert sample_counts == {}


def test_compute_local_prototypes_normalization() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=7)
    images = torch.randn(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=3)

    prototypes, _ = compute_local_prototypes(model, dataloader, normalize=True)
    for c, proto in prototypes.items():
        norm = torch.norm(proto, p=2)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


def test_aggregate_prototypes_weighted() -> None:
    p1_0 = torch.randn(128)
    p1_1 = torch.randn(128)
    p2_0 = torch.randn(128)
    p2_2 = torch.randn(128)

    p1 = {0: p1_0, 1: p1_1}
    c1 = {0: 10, 1: 5}

    p2 = {0: p2_0, 2: p2_2}
    c2 = {0: 30, 2: 20}

    aggregated = aggregate_prototypes([p1, p2], [c1, c2])

    assert set(aggregated.keys()) == {0, 1, 2}
    expected_0 = (10 * p1_0 + 30 * p2_0) / 40.0
    assert torch.allclose(aggregated[0], expected_0, atol=1e-6)
    assert torch.allclose(aggregated[1], p1_1, atol=1e-6)
    assert torch.allclose(aggregated[2], p2_2, atol=1e-6)


def test_aggregate_prototypes_unweighted() -> None:
    p1_0 = torch.randn(128)
    p1_1 = torch.randn(128)
    p2_0 = torch.randn(128)
    p2_2 = torch.randn(128)

    p1 = {0: p1_0, 1: p1_1}
    p2 = {0: p2_0, 2: p2_2}

    aggregated = aggregate_prototypes([p1, p2], client_sample_counts=None)

    assert set(aggregated.keys()) == {0, 1, 2}
    expected_0 = (p1_0 + p2_0) / 2.0
    assert torch.allclose(aggregated[0], expected_0, atol=1e-6)
    assert torch.allclose(aggregated[1], p1_1, atol=1e-6)
    assert torch.allclose(aggregated[2], p2_2, atol=1e-6)


def test_prototypes_to_tensor_dense_conversion() -> None:
    tensor_0 = torch.randn(128)
    tensor_2 = torch.randn(128)
    prototypes = {0: tensor_0, 2: tensor_2}

    proto_matrix, mask = prototypes_to_tensor(
        prototypes, num_classes=4, feature_dim=128
    )

    assert proto_matrix.shape == (4, 128)
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool

    assert mask[0].item() is True
    assert mask[1].item() is False
    assert mask[2].item() is True
    assert mask[3].item() is False

    assert torch.allclose(proto_matrix[0], tensor_0)
    assert torch.allclose(proto_matrix[1], torch.zeros(128))
    assert torch.allclose(proto_matrix[2], tensor_2)
    assert torch.allclose(proto_matrix[3], torch.zeros(128))


def test_prototypes_serialization_roundtrip() -> None:
    prototypes = {
        0: torch.randn(128),
        3: torch.randn(128),
        7: torch.randn(128),
    }

    data = prototypes_to_bytes(prototypes)
    assert isinstance(data, bytes)
    assert len(data) > 0

    restored = bytes_to_prototypes(data)
    assert set(restored.keys()) == set(prototypes.keys())
    for k in prototypes:
        assert torch.allclose(restored[k], prototypes[k], atol=1e-6)

    # Empty bytes returns empty dict
    assert bytes_to_prototypes(b"") == {}
    # Empty dict returns empty bytes
    assert prototypes_to_bytes({}) == b""
