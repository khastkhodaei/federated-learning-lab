import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import get_baseline
from src.baselines.fedprox import FedProxClientTrainer, FedProxServerStrategy, FedProxTrainer
from src.client.config import create_trainer
from src.models.cnn import SimpleCNN
from src.utils.serialization import model_to_bytes


def test_fedprox_registration() -> None:
    strat_cls, trainer_cls = get_baseline("fedprox")
    assert strat_cls is FedProxServerStrategy
    assert trainer_cls is FedProxClientTrainer


def test_fedprox_trainer_factory() -> None:
    trainer = create_trainer("fedprox", mu=0.05)
    assert isinstance(trainer, FedProxClientTrainer)
    assert trainer.mu == 0.05


def test_fedprox_loss_includes_proximal_term() -> None:
    # Setup model and synthetic data
    torch.manual_seed(42)
    model = SimpleCNN(seed=42)
    device = torch.device("cpu")

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    # Create reference initial weights
    initial_weights = model_to_bytes(model)

    # Train with mu=0.0
    model_zero = SimpleCNN(seed=42)
    opt_zero = torch.optim.SGD(model_zero.parameters(), lr=0.01)
    trainer_zero = FedProxClientTrainer(mu=0.0)
    _, _, metrics_zero = trainer_zero.train(
        model=model_zero,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=opt_zero,
        device=device,
        global_weights=initial_weights,
    )

    # Train with mu=1.0 (strong proximal penalty)
    model_prox = SimpleCNN(seed=42)
    opt_prox = torch.optim.SGD(model_prox.parameters(), lr=0.01)
    trainer_prox = FedProxClientTrainer(mu=1.0)
    _, _, metrics_prox = trainer_prox.train(
        model=model_prox,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=opt_prox,
        device=device,
        global_weights=initial_weights,
    )

    # Base loss should be positive, and proximal loss should be tracked
    assert metrics_prox["loss"] > 0.0
    assert metrics_prox["proximal_loss"] > 0.0
    assert metrics_prox["loss"] > metrics_prox["base_loss"]
    assert metrics_zero["proximal_loss"] == 0.0
