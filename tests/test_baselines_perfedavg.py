import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import get_baseline
from src.baselines.perfedavg import (
    PerFedAvgClientTrainer,
    PerFedAvgServerStrategy,
    PerFedAvgTrainer,
)
from src.client.config import ClientConfig, create_client_node, create_trainer
from src.models.cnn import SimpleCNN
from src.utils.serialization import model_to_bytes


def test_perfedavg_registration() -> None:
    strat_cls, trainer_cls = get_baseline("perfedavg")
    assert strat_cls is PerFedAvgServerStrategy
    assert trainer_cls is PerFedAvgClientTrainer


def test_perfedavg_trainer_factory() -> None:
    trainer = create_trainer("perfedavg", alpha_lr=0.02, beta_lr=0.003)
    assert isinstance(trainer, PerFedAvgClientTrainer)
    assert trainer.alpha_lr == 0.02
    assert trainer.beta_lr == 0.003


def test_perfedavg_fomaml_training_step() -> None:
    torch.manual_seed(42)
    model = SimpleCNN(seed=42)
    device = torch.device("cpu")

    inputs = torch.randn(32, 3, 32, 32)
    targets = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    trainer = PerFedAvgClientTrainer(alpha_lr=0.01, beta_lr=0.005)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    initial_weights = model_to_bytes(model)

    updated_weights, num_samples, metrics = trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_weights=initial_weights,
    )

    # Weights must be updated
    assert updated_weights != initial_weights
    assert num_samples == 32
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)


def test_perfedavg_adapt_and_evaluate() -> None:
    torch.manual_seed(42)
    model = SimpleCNN(seed=42)
    device = torch.device("cpu")

    train_inputs = torch.randn(16, 3, 32, 32)
    train_targets = torch.randint(0, 10, (16,))
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=8)

    test_inputs = torch.randn(16, 3, 32, 32)
    test_targets = torch.randint(0, 10, (16,))
    test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=8)

    criterion = nn.CrossEntropyLoss()
    trainer = PerFedAvgClientTrainer(alpha_lr=0.05, beta_lr=0.005)

    initial_weights = model_to_bytes(model)

    # adapt_and_evaluate should fine-tune a copy of model without changing model
    eval_res = trainer.adapt_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_adaptation_steps=1,
    )

    assert eval_res["total"] == 16
    assert "accuracy" in eval_res
    assert "loss" in eval_res

    # Original model must remain unmodified
    assert model_to_bytes(model) == initial_weights


def test_perfedavg_node_adaptation_evaluation() -> None:
    cfg = ClientConfig(algorithm="perfedavg")
    node = create_client_node(cfg, load_dataset=False)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    node.data = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    node.test_data = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    assert isinstance(node.trainer, PerFedAvgClientTrainer)

    # Node.evaluate_local() should trigger adapt_and_evaluate()
    eval_res = node.evaluate_local()
    assert eval_res["total"] == 16
    assert "accuracy" in eval_res
    assert "loss" in eval_res
