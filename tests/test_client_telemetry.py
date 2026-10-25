"""Unit tests for client evaluation telemetry and metric parsing."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.client.node import Node
from src.transport.federated_pb2 import TrainRequest
from src.utils.evaluation import parse_client_metrics
from src.utils.serialization import model_to_bytes


def test_parse_client_metrics_standard_format() -> None:
    msg = "loss=0.456000;epochs=2;samples=64;test_acc=0.8750;test_loss=0.380000"
    parsed = parse_client_metrics(msg)

    assert parsed["loss"] == 0.456
    assert parsed["epochs"] == 2
    assert parsed["samples"] == 64
    assert parsed["test_acc"] == 0.875
    assert parsed["test_loss"] == 0.38


def test_parse_client_metrics_malformed_string() -> None:
    msg = "invalid;;something=bad;not_a_pair"
    parsed = parse_client_metrics(msg)
    assert isinstance(parsed, dict)
    assert len(parsed) == 0


def test_node_on_train_round_includes_test_metrics() -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.zero_()

    x = torch.tensor([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=2)
    test_loader = DataLoader(dataset, batch_size=2)

    node = Node(
        node_id="telemetry_test_node",
        model=model,
        data=train_loader,
        test_data=test_loader,
    )

    req = TrainRequest(
        round_id=1,
        weights=model_to_bytes(model),
        epochs=1,
    )
    res = node.on_train_round(req)

    assert res.success is True
    assert "test_acc=" in res.message
    metrics = parse_client_metrics(res.message)
    assert metrics["test_acc"] == 1.0
    assert "test_loss" in metrics
