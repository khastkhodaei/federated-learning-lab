import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import get_baseline
from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.baselines.fedavg.trainer import FedAvgClientTrainer
from src.utils.serialization import (
    load_bytes_into_model,
    model_to_bytes,
)


class TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_fedavg_server_strategy_mathematical_average():
    # Model with 2 parameters: [w1, w2]
    model_client1 = TinyLinearModel()
    model_client1.fc.weight.data = torch.tensor([[2.0, 4.0]])

    model_client2 = TinyLinearModel()
    model_client2.fc.weight.data = torch.tensor([[6.0, 8.0]])

    # Client 1 has 100 samples, Client 2 has 300 samples (total 400)
    # Expected weighted average:
    # w1 = (100 * 2.0 + 300 * 6.0) / 400 = (200 + 1800) / 400 = 5.0
    # w2 = (100 * 4.0 + 300 * 8.0) / 400 = (400 + 2400) / 400 = 7.0
    expected_weights = torch.tensor([[5.0, 7.0]])

    updates = [
        (model_to_bytes(model_client1), 100),
        (model_to_bytes(model_client2), 300),
    ]

    strategy = FedAvgServerStrategy()
    server_model = TinyLinearModel()
    aggregated_bytes = strategy.aggregate(
        global_model=server_model,
        client_updates=updates,
        round_id=1,
    )

    assert aggregated_bytes is not None
    assert torch.allclose(server_model.fc.weight.data, expected_weights)


def test_fedavg_server_strategy_empty_updates_raises():
    strategy = FedAvgServerStrategy()
    with pytest.raises(ValueError, match="Cannot aggregate empty updates"):
        strategy.aggregate(global_model=None, client_updates=[], round_id=1)


def test_fedavg_client_trainer_local_training():
    torch.manual_seed(42)
    model = TinyLinearModel()
    # Initial weight [1.0, 1.0]
    model.fc.weight.data = torch.tensor([[1.0, 1.0]])

    # Data where target = 0 (gradient should push weights towards 0)
    X = torch.ones((16, 2))
    y = torch.zeros((16, 1))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = FedAvgClientTrainer()
    updated_bytes, num_samples, metrics = trainer.train(
        model=model,
        dataloader=dataloader,
        epochs=2,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert num_samples == 32  # 16 samples * 2 epochs
    assert len(updated_bytes) > 0
    assert "loss" in metrics

    # Weight should have decreased from [1.0, 1.0]
    updated_model = TinyLinearModel()
    load_bytes_into_model(updated_bytes, updated_model)
    assert updated_model.fc.weight.data[0, 0] < 1.0


def test_baseline_factory_resolution():
    strategy_cls, trainer_cls = get_baseline("fedavg")
    assert strategy_cls is FedAvgServerStrategy
    assert trainer_cls is FedAvgClientTrainer

    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline("non_existent_baseline")


def test_server_and_node_default_to_fedavg_strategy():
    from src.client.node import Node
    from src.server.server import FederatedServer

    server = FederatedServer(host="127.0.0.1", port=0)
    assert isinstance(server.strategy, FedAvgServerStrategy)
    server.stop(grace=0)

    node = Node(node_id="client_test", port=0)
    assert isinstance(node.trainer, FedAvgClientTrainer)
    node.stop(grace=0)
