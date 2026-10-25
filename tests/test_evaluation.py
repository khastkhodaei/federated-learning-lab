"""Unit tests for model evaluation and test dataset loading."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.evaluation import evaluate_model, get_global_test_loader


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
