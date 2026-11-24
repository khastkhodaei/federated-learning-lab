"""Unit tests for model evaluation and test dataset loading."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.transport.federated_pb2 import TrainResponse
from src.utils.evaluation import (
    aggregate_client_round_metrics,
    evaluate_model,
    get_global_test_loader,
)


def test_evaluate_model_accuracy_and_loss() -> None:
    """Tests evaluate_model computes accurate loss, accuracy, and counts on synthetic data."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.zero_()

    x = torch.tensor([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    metrics = evaluate_model(model, loader)

    assert metrics["total"] == 4
    assert metrics["correct"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] >= 0.0
    assert isinstance(metrics["loss"], float)


def test_evaluate_model_empty_loader() -> None:
    """Tests evaluate_model handles an empty DataLoader gracefully."""
    model = nn.Linear(2, 2)
    x = torch.empty((0, 2))
    y = torch.empty((0,), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_model(model, loader)

    assert metrics["total"] == 0
    assert metrics["correct"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["loss"] == 0.0


def test_get_global_test_loader_structure() -> None:
    """Tests get_global_test_loader returns a valid DataLoader for cifar10."""
    loader = get_global_test_loader(
        dataset_name="cifar10",
        data_dir="./data",
        batch_size=64,
        download=False,
    )
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 64


def test_evaluate_model_restores_train_eval_mode() -> None:
    """Verifies evaluate_model restores model.training regardless of initial mode."""
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)

    model.train()
    assert model.training is True
    evaluate_model(model, loader)
    assert model.training is True

    model.eval()
    assert model.training is False
    evaluate_model(model, loader)
    assert model.training is False


def test_aggregate_client_round_metrics_excludes_missing_without_zero_imputation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies missing client metrics are excluded from mean denominators and trigger warnings."""
    responses = [
        TrainResponse(
            node_id="client_0",
            success=True,
            num_samples=20,
            message="loss=0.4;test_acc=0.8",
        ),
        TrainResponse(
            node_id="client_1",
            success=True,
            num_samples=30,
            message="loss=0.6",  # missing test_acc
        ),
        TrainResponse(
            node_id="client_2",
            success=True,
            num_samples=10,
            message="",  # missing both loss and test_acc
        ),
    ]

    with caplog.at_level("WARNING"):
        summary = aggregate_client_round_metrics(responses, round_id=3)

    # client_1 and client_2 excluded from test_acc mean -> mean is 0.8 (not (0.8+0+0)/3)
    assert summary["mean_local_acc"] == pytest.approx(0.8)
    assert summary["client_accs"] == [pytest.approx(0.8)]
    # client_2 excluded from loss mean -> mean is (0.4 + 0.6)/2 = 0.5
    assert summary["mean_loss"] == pytest.approx(0.5)
    assert summary["weighted_loss"] == pytest.approx((0.4 * 20 + 0.6 * 30) / 50)
    assert "test_accuracy" not in summary["client_metrics"]["client_1"]
    assert "loss" not in summary["client_metrics"]["client_2"]
    assert any("client_1" in rec.message and "test_acc" in rec.message for rec in caplog.records)
    assert any("client_2" in rec.message for rec in caplog.records)
