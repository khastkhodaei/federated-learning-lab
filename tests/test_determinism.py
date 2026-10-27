import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.ditto.trainer import DittoClientTrainer
from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.baselines.fedavg.trainer import FedAvgClientTrainer
from src.baselines.fedprox.trainer import FedProxClientTrainer
from src.baselines.perfedavg.trainer import PerFedAvgClientTrainer
from src.baselines.pfedgraph.strategy import PFedGraphServerStrategy
from src.baselines.pfedgraph.trainer import PFedGraphClientTrainer
from src.models.cnn import SimpleCNN
from src.utils.serialization import bytes_to_vector, model_to_bytes


def test_model_seed_reproducibility() -> None:
    """Verifies SimpleCNN seed reproducibility and non-identical initialization across distinct seeds."""
    m1 = SimpleCNN(seed=42)
    m2 = SimpleCNN(seed=42)
    m3 = SimpleCNN(seed=43)

    # Identical seed -> bit-exact parameters across all layers
    for (name1, p1), (name2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"Parameter {name1} mismatch for identical seed=42"

    # Distinct seeds -> differing parameter weights
    any_different = False
    for p1, p3 in zip(m1.parameters(), m3.parameters()):
        if not torch.equal(p1, p3):
            any_different = True
            break
    assert any_different, "Models with seed=42 and seed=43 must have differing parameters"


def test_client_trainer_trajectory_determinism() -> None:
    """Verifies bit-exact trajectory determinism across all five baseline client trainers."""
    device = torch.device("cpu")
    torch.manual_seed(999)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    dataset = TensorDataset(inputs, targets)
    criterion = nn.CrossEntropyLoss()

    trainers: list[tuple[str, object]] = [
        ("fedavg", FedAvgClientTrainer()),
        ("fedprox", FedProxClientTrainer(mu=0.01)),
        ("ditto", DittoClientTrainer(lambda_penalty=0.1, personalized_epochs=1)),
        ("perfedavg", PerFedAvgClientTrainer(alpha_lr=0.01, beta_lr=0.005)),
        ("pfedgraph", PFedGraphClientTrainer()),
    ]

    for name, trainer in trainers:
        m1 = SimpleCNN(seed=100)
        m2 = SimpleCNN(seed=100)
        initial_bytes = model_to_bytes(m1)

        loader1 = DataLoader(dataset, batch_size=8, shuffle=False)
        loader2 = DataLoader(dataset, batch_size=8, shuffle=False)

        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)

        w1, n1, met1 = getattr(trainer, "train")(
            model=m1,
            dataloader=loader1,
            epochs=1,
            criterion=criterion,
            optimizer=opt1,
            device=device,
            global_weights=initial_bytes,
        )

        w2, n2, met2 = getattr(trainer, "train")(
            model=m2,
            dataloader=loader2,
            epochs=1,
            criterion=criterion,
            optimizer=opt2,
            device=device,
            global_weights=initial_bytes,
        )

        assert n1 == n2 == 16, f"{name}: Sample count mismatch"
        v1 = bytes_to_vector(w1)
        v2 = bytes_to_vector(w2)
        assert torch.equal(v1, v2), f"{name}: Updated weights are not bit-exact across identical runs"
        assert pytest.approx(met1["loss"], rel=1e-6) == met2["loss"], (
            f"{name}: Loss metric discrepancy ({met1['loss']} vs {met2['loss']})"
        )


def test_strategy_aggregation_determinism() -> None:
    """Verifies that server strategy aggregation produces identical aggregated bytes given identical inputs."""
    m_c1 = SimpleCNN(seed=1)
    m_c2 = SimpleCNN(seed=2)
    b1 = model_to_bytes(m_c1)
    b2 = model_to_bytes(m_c2)

    updates = [(b1, 50), (b2, 100)]
    client_ids = ["client_0", "client_1"]

    # 1. FedAvgServerStrategy determinism
    s_fedavg = FedAvgServerStrategy()
    g_fed1 = SimpleCNN(seed=42)
    g_fed2 = SimpleCNN(seed=42)

    res_fed1 = s_fedavg.aggregate(g_fed1, updates, round_id=1)
    res_fed2 = s_fedavg.aggregate(g_fed2, updates, round_id=1)

    assert res_fed1 == res_fed2, "FedAvg aggregation is non-deterministic"
    for p1, p2 in zip(g_fed1.parameters(), g_fed2.parameters()):
        assert torch.equal(p1, p2)

    # 2. PFedGraphServerStrategy determinism
    s_pfed1 = PFedGraphServerStrategy(alpha=0.5, lam=0.01)
    s_pfed2 = PFedGraphServerStrategy(alpha=0.5, lam=0.01)
    g_pfed1 = SimpleCNN(seed=42)
    g_pfed2 = SimpleCNN(seed=42)

    res_pfed1 = s_pfed1.aggregate(g_pfed1, updates, round_id=1, client_ids=client_ids)
    res_pfed2 = s_pfed2.aggregate(g_pfed2, updates, round_id=1, client_ids=client_ids)

    assert res_pfed1 == res_pfed2, "pFedGraph global model aggregation is non-deterministic"
    assert s_pfed1.client_weights == s_pfed2.client_weights, (
        "pFedGraph personalized client models are non-deterministic"
    )
    if s_pfed1.adjacency_matrix is not None and s_pfed2.adjacency_matrix is not None:
        assert torch.equal(s_pfed1.adjacency_matrix, s_pfed2.adjacency_matrix), (
            "pFedGraph adjacency matrix is non-deterministic"
        )
