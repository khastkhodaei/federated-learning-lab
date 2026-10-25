"""Unit tests for client node local evaluation and test partitioning."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.client.node import Node
from src.utils.serialization import model_to_bytes


def test_node_evaluate_local_with_test_data() -> None:
    """Verifies that Node.evaluate_local() computes loss and accuracy on its local test set."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.zero_()

    x = torch.tensor([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    test_loader = DataLoader(TensorDataset(x, y), batch_size=2)

    node = Node(node_id="client_eval_test", model=model, test_data=test_loader)

    metrics = node.evaluate_local()

    assert metrics["total"] == 4
    assert metrics["correct"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] >= 0.0


def test_node_evaluate_local_with_injected_weights() -> None:
    """Verifies Node.evaluate_local(weights=...) loads weights before evaluating."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.zero_()

    # Create target weights that correctly classify
    target_model = nn.Linear(2, 2)
    with torch.no_grad():
        target_model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        target_model.bias.zero_()
    serialized = model_to_bytes(target_model)

    x = torch.tensor([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    test_loader = DataLoader(TensorDataset(x, y), batch_size=2)

    node = Node(node_id="client_injected_weights", model=model, test_data=test_loader)

    metrics = node.evaluate_local(weights=serialized)

    assert metrics["total"] == 4
    assert metrics["correct"] == 4
    assert metrics["accuracy"] == 1.0


def test_node_evaluate_local_fallback_when_no_data() -> None:
    """Verifies Node.evaluate_local returns empty metrics gracefully when no test data exists."""
    model = nn.Linear(2, 2)
    node = Node(node_id="client_empty", model=model, data=None, test_data=None)

    metrics = node.evaluate_local()

    assert metrics["total"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["loss"] == 0.0
