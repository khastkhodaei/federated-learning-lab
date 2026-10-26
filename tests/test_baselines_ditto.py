import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import get_baseline
from src.baselines.ditto import DittoClientTrainer, DittoServerStrategy, DittoTrainer
from src.client.config import create_client_node, create_trainer, ClientConfig
from src.models.cnn import SimpleCNN
from src.utils.serialization import bytes_to_vector, model_to_bytes


def test_ditto_registration() -> None:
    strat_cls, trainer_cls = get_baseline("ditto")
    assert strat_cls is DittoServerStrategy
    assert trainer_cls is DittoClientTrainer


def test_ditto_trainer_factory() -> None:
    trainer = create_trainer("ditto", lambda_penalty=0.25, personalized_epochs=2)
    assert isinstance(trainer, DittoClientTrainer)
    assert trainer.lambda_penalty == 0.25
    assert trainer.personalized_epochs == 2


def test_ditto_dual_model_optimization() -> None:
    torch.manual_seed(42)
    model = SimpleCNN(seed=42)
    device = torch.device("cpu")

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    trainer = DittoClientTrainer(lambda_penalty=0.1, personalized_epochs=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    initial_global_bytes = model_to_bytes(model)

    # First round of training
    updated_bytes, num_samples, metrics = trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        global_weights=initial_global_bytes,
    )

    # Verify returned weights are for the global model and have moved
    assert updated_bytes != initial_global_bytes
    assert num_samples == 16
    assert "personalized_loss" in metrics
    assert "loss" in metrics

    # Verify personalized model exists and is a distinct PyTorch module
    assert trainer.personalized_model is not None
    assert trainer.personalized_model is not model

    pers_bytes = model_to_bytes(trainer.personalized_model)
    # The personalized model took SGD steps with proximal penalty; its weights differ from initial
    assert pers_bytes != initial_global_bytes


def test_ditto_node_evaluates_personalized_model() -> None:
    cfg = ClientConfig(algorithm="ditto")
    node = create_client_node(cfg, load_dataset=False)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    node.data = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    node.test_data = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    assert isinstance(node.trainer, DittoClientTrainer)

    # Before train, personalized model is not yet initialized
    assert node.trainer.personalized_model is None

    # Run train round to trigger dual-model setup
    req_weights = model_to_bytes(node.model)
    from src.transport.federated_pb2 import TrainRequest
    req = TrainRequest(round_id=1, weights=req_weights, epochs=1)
    res = node.on_train_round(req)

    assert res.success
    assert node.trainer.personalized_model is not None

    # evaluate_local() evaluates personalized_model
    eval_res = node.evaluate_local()
    assert eval_res["total"] == 16
    assert "accuracy" in eval_res
