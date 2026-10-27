import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import get_baseline
from src.baselines.pfedgraph import (
    PFedGraphClientTrainer,
    PFedGraphServerStrategy,
)
from src.baselines.pfedgraph.strategy import project_to_simplex
from src.client.config import create_trainer
from src.models.cnn import SimpleCNN
from src.utils.serialization import model_to_bytes


def test_project_to_simplex() -> None:
    # Test random vector
    v = torch.tensor([1.5, -0.5, 2.0, 0.0])
    p = project_to_simplex(v)
    assert torch.all(p >= 0.0)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5)

    # Test already on simplex
    v_simp = torch.tensor([0.25, 0.25, 0.25, 0.25])
    p_simp = project_to_simplex(v_simp)
    assert torch.allclose(p_simp, v_simp, atol=1e-5)

    # Test single element
    v_one = torch.tensor([5.0])
    p_one = project_to_simplex(v_one)
    assert torch.isclose(p_one, torch.tensor([1.0]))


def test_pfedgraph_registration() -> None:
    strat_cls, trainer_cls = get_baseline("pfedgraph")
    assert strat_cls is PFedGraphServerStrategy
    assert trainer_cls is PFedGraphClientTrainer


def test_pfedgraph_trainer_factory() -> None:
    trainer = create_trainer("pfedgraph")
    assert isinstance(trainer, PFedGraphClientTrainer)


def test_pfedgraph_server_strategy_graph_and_personalized_aggregation() -> None:
    strategy = PFedGraphServerStrategy(alpha=0.5, lam=0.01)

    # Create 3 distinct client models
    m1 = SimpleCNN(seed=1)
    m2 = SimpleCNN(seed=2)
    m3 = SimpleCNN(seed=3)

    b1 = model_to_bytes(m1)
    b2 = model_to_bytes(m2)
    b3 = model_to_bytes(m3)

    client_updates = [(b1, 100), (b2, 200), (b3, 300)]
    client_ids = ["client_0", "client_1", "client_2"]

    global_model = SimpleCNN(seed=42)
    aggregated_bytes = strategy.aggregate(
        global_model=global_model,
        client_updates=client_updates,
        round_id=1,
        client_ids=client_ids,
    )

    # Verify global model was aggregated
    assert len(aggregated_bytes) > 0

    # Verify adjacency matrix W is valid
    w = strategy.adjacency_matrix
    assert w is not None
    assert w.shape == (3, 3)
    # Each row must sum to 1
    assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)
    # Elements must be non-negative
    assert torch.all(w >= 0.0)

    # Verify client-specific personalized models were generated
    assert len(strategy.client_weights) == 3
    for cid in client_ids:
        assert cid in strategy.client_weights
        assert len(strategy.client_weights[cid]) > 0

    # Verify that different clients receive distinct personalized models
    p0 = strategy.client_weights["client_0"]
    p1 = strategy.client_weights["client_1"]
    p2 = strategy.client_weights["client_2"]
    assert p0 != p1 or p1 != p2


def test_pfedgraph_client_trainer() -> None:
    torch.manual_seed(42)
    model = SimpleCNN(seed=42)
    device = torch.device("cpu")

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    trainer = PFedGraphClientTrainer()
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

    assert updated_weights != initial_weights
    assert num_samples == 16
    assert "loss" in metrics


def test_pfedgraph_deterministic_client_order_permutation() -> None:
    m1 = SimpleCNN(seed=1)
    m2 = SimpleCNN(seed=2)
    b1 = model_to_bytes(m1)
    b2 = model_to_bytes(m2)

    strat_a = PFedGraphServerStrategy(alpha=0.5, lam=0.01)
    strat_b = PFedGraphServerStrategy(alpha=0.5, lam=0.01)

    strat_a.aggregate(
        global_model=None,
        client_updates=[(b1, 100), (b2, 200)],
        round_id=1,
        client_ids=["client_0", "client_1"],
    )
    strat_b.aggregate(
        global_model=None,
        client_updates=[(b2, 200), (b1, 100)],
        round_id=1,
        client_ids=["client_1", "client_0"],
    )

    assert strat_a.client_ids == ["client_0", "client_1"]
    assert strat_b.client_ids == ["client_0", "client_1"]
    assert torch.allclose(strat_a.adjacency_matrix, strat_b.adjacency_matrix, atol=1e-5)
    assert strat_a.client_weights["client_0"] == strat_b.client_weights["client_0"]
    assert strat_a.client_weights["client_1"] == strat_b.client_weights["client_1"]

