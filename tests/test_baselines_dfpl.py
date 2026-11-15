import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import BASELINES, create_server_strategy, get_baseline
from src.baselines.dfpl import (
    DFPLClientTrainer,
    DFPLServerStrategy,
    aggregate_prototypes,
    bytes_to_prototypes,
    compute_local_prototypes,
    prototypes_to_bytes,
    prototypes_to_tensor,
)
from src.client.config import create_trainer, create_trainer_from_config
from src.models.cnn import SimpleCNN
from src.server.aggregation import fedavg_bytes
from src.utils.config import load_unified_config
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


def test_dfpl_server_strategy_initialization() -> None:
    strategy = DFPLServerStrategy()
    assert strategy.normalize_prototypes is False
    assert strategy.aggregate_weights is True
    assert strategy.global_prototypes == {}
    assert strategy.global_prototypes_bytes == b""


def test_dfpl_server_strategy_aggregate_empty_raises() -> None:
    strategy = DFPLServerStrategy()
    global_model = SimpleCNN(in_channels=3, num_classes=10)
    with pytest.raises(ValueError, match="empty updates"):
        strategy.aggregate(global_model, [], round_id=0)


def test_dfpl_server_strategy_aggregate_weights_only() -> None:
    m1 = SimpleCNN(in_channels=3, num_classes=10, seed=1)
    m2 = SimpleCNN(in_channels=3, num_classes=10, seed=2)
    global_model = SimpleCNN(in_channels=3, num_classes=10, seed=99)

    u1 = (model_to_bytes(m1), 10)
    u2 = (model_to_bytes(m2), 30)

    strategy = DFPLServerStrategy()
    aggregated_bytes = strategy.aggregate(global_model, [u1, u2], round_id=1)

    expected_bytes = fedavg_bytes([u1, u2])
    assert aggregated_bytes == expected_bytes
    assert model_to_bytes(global_model) == expected_bytes


def test_dfpl_server_strategy_aggregate_with_prototypes_dict() -> None:
    m1 = SimpleCNN(in_channels=3, num_classes=10, seed=1)
    m2 = SimpleCNN(in_channels=3, num_classes=10, seed=2)
    global_model = SimpleCNN(in_channels=3, num_classes=10, seed=99)

    p1 = {0: torch.randn(128), 1: torch.randn(128)}
    c1 = {0: 10, 1: 5}
    p2 = {0: torch.randn(128), 2: torch.randn(128)}
    c2 = {0: 30, 2: 20}

    u1 = (model_to_bytes(m1), 15)
    u2 = (model_to_bytes(m2), 50)

    strategy = DFPLServerStrategy()
    strategy.aggregate(
        global_model,
        [u1, u2],
        round_id=1,
        client_prototypes=[p1, p2],
        client_sample_counts=[c1, c2],
    )

    assert set(strategy.global_prototypes.keys()) == {0, 1, 2}
    expected_0 = (10 * p1[0] + 30 * p2[0]) / 40.0
    assert torch.allclose(strategy.global_prototypes[0], expected_0, atol=1e-6)
    assert torch.allclose(strategy.global_prototypes[1], p1[1], atol=1e-6)
    assert torch.allclose(strategy.global_prototypes[2], p2[2], atol=1e-6)

    assert len(strategy.global_prototypes_bytes) > 0
    restored = bytes_to_prototypes(strategy.global_prototypes_bytes)
    assert set(restored.keys()) == {0, 1, 2}
    for k in strategy.global_prototypes:
        assert torch.allclose(restored[k], strategy.global_prototypes[k], atol=1e-6)


def test_dfpl_server_strategy_aggregate_with_prototypes_bytes() -> None:
    m1 = SimpleCNN(in_channels=3, num_classes=10, seed=1)
    m2 = SimpleCNN(in_channels=3, num_classes=10, seed=2)
    global_model = SimpleCNN(in_channels=3, num_classes=10, seed=99)

    p1 = {0: torch.randn(128), 1: torch.randn(128)}
    c1 = {0: 10, 1: 5}
    p2 = {0: torch.randn(128), 2: torch.randn(128)}
    c2 = {0: 30, 2: 20}

    u1 = (model_to_bytes(m1), 15)
    u2 = (model_to_bytes(m2), 50)

    strategy = DFPLServerStrategy()
    strategy.aggregate(
        global_model,
        [u1, u2],
        round_id=1,
        client_prototypes=[prototypes_to_bytes(p1), prototypes_to_bytes(p2)],
        client_sample_counts=[c1, c2],
    )

    expected_0 = (10 * p1[0] + 30 * p2[0]) / 40.0
    assert torch.allclose(strategy.global_prototypes[0], expected_0, atol=1e-6)
    assert torch.allclose(strategy.global_prototypes[1], p1[1], atol=1e-6)
    assert torch.allclose(strategy.global_prototypes[2], p2[2], atol=1e-6)


def test_dfpl_server_strategy_get_client_payload() -> None:
    strategy = DFPLServerStrategy()
    p1 = {0: torch.randn(128)}
    m1 = SimpleCNN(in_channels=3, num_classes=10, seed=1)
    strategy.aggregate(
        None,
        [(model_to_bytes(m1), 10)],
        round_id=1,
        client_prototypes=[p1],
    )

    payload = strategy.get_client_payload()
    assert payload == strategy.global_prototypes_bytes
    assert strategy.get_client_payload("client_1") == strategy.global_prototypes_bytes
    assert strategy.get_global_prototypes() == strategy.global_prototypes
    assert strategy.get_global_prototypes_bytes() == strategy.global_prototypes_bytes


def test_dfpl_server_strategy_normalization() -> None:
    strategy = DFPLServerStrategy(normalize_prototypes=True)
    p1 = {0: torch.randn(128) * 10, 1: torch.randn(128) * 5}
    m1 = SimpleCNN(in_channels=3, num_classes=10, seed=1)

    strategy.aggregate(
        None,
        [(model_to_bytes(m1), 10)],
        round_id=1,
        client_prototypes=[p1],
    )

    for proto in strategy.global_prototypes.values():
        norm = torch.norm(proto, p=2)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


def test_dfpl_baseline_registry() -> None:
    assert "dfpl" in BASELINES
    server_cls, client_cls = get_baseline("dfpl")
    assert server_cls is DFPLServerStrategy
    assert client_cls is DFPLClientTrainer
    strategy = create_server_strategy("dfpl")
    assert isinstance(strategy, DFPLServerStrategy)


def test_dfpl_create_trainer_from_factory() -> None:
    trainer = create_trainer("dfpl", lambda_proto=0.25)
    assert isinstance(trainer, DFPLClientTrainer)
    assert trainer.lambda_proto == 0.25


def test_dfpl_config_yaml_loading() -> None:
    cfg = load_unified_config("configs/config.yaml")
    assert hasattr(cfg.baselines, "dfpl")
    assert cfg.baselines.dfpl.lambda_proto == 0.1
    assert cfg.baselines.dfpl.normalize_prototypes is False
    assert cfg.baselines.dfpl.loss_type == "mse"
    assert cfg.baselines.dfpl.aggregate_weights is True


def test_dfpl_to_client_config_and_trainer() -> None:
    cfg = load_unified_config("configs/config.yaml")
    cfg.algorithm = "dfpl"
    cfg.baseline = "dfpl"
    client_cfg = cfg.to_client_config(client_id=1)
    assert hasattr(client_cfg, "dfpl")
    assert client_cfg.dfpl.lambda_proto == 0.1
    assert client_cfg.algorithm == "dfpl"

    trainer = create_trainer_from_config(client_cfg)
    assert isinstance(trainer, DFPLClientTrainer)
    assert trainer.lambda_proto == 0.1


def test_dfpl_end_to_end_simulation(tmp_path: Path) -> None:
    from run_simulation import run_federated_simulation

    reports_dir = tmp_path / "reports"
    results = run_federated_simulation(
        algorithm="dfpl",
        rounds=2,
        num_clients=2,
        epochs=1,
        synthetic=True,
        reports_dir=str(reports_dir),
    )

    assert results["algorithm"] == "dfpl"
    assert len(results["round_history"]) == 2
    metrics_file = reports_dir / "dfpl" / "metrics.json"
    assert metrics_file.is_file()
    with open(metrics_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["baseline"] == "dfpl"
    assert len(payload["rounds"]) == 2




