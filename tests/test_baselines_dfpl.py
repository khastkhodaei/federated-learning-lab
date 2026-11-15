import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.dfpl import (
    DFPLClientTrainer,
    aggregate_prototypes,
    bytes_to_prototypes,
    compute_local_prototypes,
    prototypes_to_bytes,
    prototypes_to_tensor,
)
from src.models.cnn import SimpleCNN
from src.utils.serialization import model_to_bytes


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


def test_dfpl_client_trainer_initialization() -> None:
    trainer = DFPLClientTrainer()
    assert trainer.lambda_proto == 0.1
    assert trainer.normalize_prototypes is False
    assert trainer.loss_type == "mse"
    assert trainer.last_local_prototypes == {}
    assert trainer.last_sample_counts == {}
    assert trainer.last_prototypes_bytes == b""

    with pytest.raises(ValueError, match="lambda_proto"):
        DFPLClientTrainer(lambda_proto=-0.5)

    with pytest.raises(ValueError, match="loss_type"):
        DFPLClientTrainer(loss_type="invalid")


def test_dfpl_client_trainer_train_without_prototypes() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    trainer = DFPLClientTrainer(lambda_proto=0.1)
    weights, total_samples, metrics = trainer.train(
        model, dataloader, epochs=1, criterion=criterion, optimizer=optimizer, device=device, global_prototypes=None
    )

    assert isinstance(weights, bytes)
    assert len(weights) > 0
    assert total_samples == 8
    assert metrics["loss_proto"] == 0.0
    assert metrics["loss_ce"] > 0.0
    assert metrics["loss"] > 0.0

    assert set(trainer.last_local_prototypes.keys()) == {0, 1}
    for c in (0, 1):
        assert trainer.last_local_prototypes[c].shape == (128,)

    assert isinstance(trainer.last_prototypes_bytes, bytes)
    assert len(trainer.last_prototypes_bytes) > 0
    assert trainer.last_sample_counts == {0: 4, 1: 4}


def test_dfpl_client_trainer_train_with_prototypes() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    initial_weights = model_to_bytes(model)

    target_protos = {0: torch.randn(128), 1: torch.randn(128)}
    trainer = DFPLClientTrainer(lambda_proto=0.5)

    weights, total_samples, metrics = trainer.train(
        model,
        dataloader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_prototypes=target_protos,
    )

    assert metrics["loss_proto"] > 0.0
    assert weights != initial_weights

    # Passing serialized bytes
    proto_bytes = prototypes_to_bytes(target_protos)
    weights2, _, metrics2 = trainer.train(
        model,
        dataloader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_prototypes=proto_bytes,
    )
    assert metrics2["loss_proto"] > 0.0


def test_dfpl_client_trainer_prototype_alignment_gradient() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    target_protos = {0: torch.randn(128), 1: torch.randn(128)}

    trainer = DFPLClientTrainer(lambda_proto=1.0)
    trainer.train(
        model,
        dataloader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_prototypes=target_protos,
    )

    assert model.conv1.weight.grad is not None
    assert model.conv2.weight.grad is not None
    assert model.fc1.weight.grad is not None
    assert torch.count_nonzero(model.conv1.weight.grad) > 0
    assert torch.count_nonzero(model.conv2.weight.grad) > 0
    assert torch.count_nonzero(model.fc1.weight.grad) > 0


def test_dfpl_client_trainer_missing_class_graceful() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    # Global prototypes only has class 9, but local dataloader only has 0 and 1
    global_protos = {9: torch.randn(128)}

    trainer = DFPLClientTrainer(lambda_proto=0.5)
    _, total_samples, metrics = trainer.train(
        model,
        dataloader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_prototypes=global_protos,
    )

    assert total_samples == 8
    assert metrics["loss_proto"] == 0.0
    assert metrics["loss_ce"] > 0.0


def test_dfpl_client_trainer_cosine_loss() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    target_protos = {0: torch.randn(128), 1: torch.randn(128)}

    trainer = DFPLClientTrainer(loss_type="cosine", lambda_proto=0.5)
    _, _, metrics = trainer.train(
        model,
        dataloader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_prototypes=target_protos,
    )

    assert metrics["loss_proto"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["loss_proto"]))


def test_dfpl_client_trainer_zero_epochs() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10, seed=42)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    initial_weights = model_to_bytes(model)

    trainer = DFPLClientTrainer()
    weights, total_samples, metrics = trainer.train(
        model, dataloader, epochs=0, criterion=criterion, optimizer=optimizer, device=device
    )

    assert weights == initial_weights
    assert total_samples == 0
    assert metrics["loss"] == 0.0
    assert metrics["loss_ce"] == 0.0
    assert metrics["loss_proto"] == 0.0

