"""Rigorous algorithmic and mathematical verification test suite for federated learning baselines.

Validates exact analytical limits, reduction equivalences to FedAvg, structural graph
properties, directed Push-Sum consensus dynamics, autograd mechanics, and hyperparameter
monotonicity across all implemented federated baselines.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.dfedpgp.strategy import _push_sum_step
from src.baselines.dfedpgp.topology import build_directed_mixing_matrix
from src.baselines.dfpl.trainer import DFPLClientTrainer
from src.baselines.ditto.trainer import DittoClientTrainer
from src.baselines.fedavg.trainer import FedAvgClientTrainer
from src.baselines.fedprox.trainer import FedProxClientTrainer
from src.baselines.perfedavg.trainer import PerFedAvgClientTrainer
from src.baselines.pfedgraph.strategy import PFedGraphServerStrategy
from src.utils.serialization import (
    bytes_to_vector,
    model_to_bytes,
    vector_to_bytes,
)


class LinearRegressionNoBias(nn.Module):
    """Single-layer linear regression model without bias for closed-form convex testing."""

    def __init__(self, in_features: int, initial_weights: torch.Tensor) -> None:
        super().__init__()
        assert initial_weights.shape[0] == in_features, (
            f"Shape mismatch: {initial_weights.shape[0]} != {in_features}"
        )
        self.w = nn.Parameter(initial_weights.clone().T)  # Shape (1, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T


class HalfMSELoss(nn.Module):
    """Quadratic loss defined as (1/2) * ||Xw - y||_2^2."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((pred - target) ** 2)


class FeatureEmbeddingModel(nn.Module):
    """Toy classification model exposing intermediate feature representations."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Linear(in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits


# ==============================================================================
# Group 1: Limit-Case Reduction Equivalence Tests (Exact Parity with FedAvg)
# ==============================================================================


def test_fedprox_reduces_to_fedavg_when_mu_zero() -> None:
    """Verifies that FedProxClientTrainer(mu=0.0) produces updates identical to FedAvg."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    base_model = nn.Linear(4, 2)

    # 1. FedAvg standard descent
    model_fedavg = copy.deepcopy(base_model)
    opt_fedavg = torch.optim.SGD(model_fedavg.parameters(), lr=0.01)
    trainer_fedavg = FedAvgClientTrainer()
    bytes_fedavg, samples_fedavg, _ = trainer_fedavg.train(
        model=model_fedavg,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_fedavg,
        device=torch.device("cpu"),
    )

    # 2. FedProx with mu=0.0
    model_fedprox = copy.deepcopy(base_model)
    opt_fedprox = torch.optim.SGD(model_fedprox.parameters(), lr=0.01)
    trainer_fedprox = FedProxClientTrainer(mu=0.0)
    bytes_fedprox, samples_fedprox, _ = trainer_fedprox.train(
        model=model_fedprox,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_fedprox,
        device=torch.device("cpu"),
    )

    vec_fedavg = bytes_to_vector(bytes_fedavg)
    vec_fedprox = bytes_to_vector(bytes_fedprox)

    assert samples_fedavg == samples_fedprox == 40
    assert torch.allclose(vec_fedavg, vec_fedprox, atol=1e-5)


def test_ditto_global_branch_reduces_to_fedavg() -> None:
    """Verifies that Ditto's global model branch w updates identically to FedAvg."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    base_model = nn.Linear(4, 2)

    # 1. FedAvg
    model_fedavg = copy.deepcopy(base_model)
    opt_fedavg = torch.optim.SGD(model_fedavg.parameters(), lr=0.01)
    trainer_fedavg = FedAvgClientTrainer()
    bytes_fedavg, _, _ = trainer_fedavg.train(
        model=model_fedavg,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_fedavg,
        device=torch.device("cpu"),
    )

    # 2. Ditto
    model_ditto = copy.deepcopy(base_model)
    opt_ditto = torch.optim.SGD(model_ditto.parameters(), lr=0.01)
    trainer_ditto = DittoClientTrainer(lambda_penalty=0.5, personalized_epochs=2)
    bytes_ditto, _, _ = trainer_ditto.train(
        model=model_ditto,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_ditto,
        device=torch.device("cpu"),
    )

    vec_fedavg = bytes_to_vector(bytes_fedavg)
    vec_ditto = bytes_to_vector(bytes_ditto)

    # Check global model output bytes match FedAvg
    assert torch.allclose(vec_fedavg, vec_ditto, atol=1e-5)

    # Verify that v_k optimization in Phase 2 did not mutate model_ditto parameters
    vec_ditto_model = bytes_to_vector(model_to_bytes(model_ditto))
    assert torch.allclose(vec_ditto, vec_ditto_model, atol=1e-6)


def test_dfpl_reduces_to_fedavg_when_lambda_zero() -> None:
    """Verifies that DFPL with lambda_proto=0.0 reduces exactly to standard FedAvg."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    base_model = FeatureEmbeddingModel(in_features=4, hidden_dim=8, num_classes=2)

    # 1. FedAvg
    model_fedavg = copy.deepcopy(base_model)
    opt_fedavg = torch.optim.SGD(model_fedavg.parameters(), lr=0.01)
    trainer_fedavg = FedAvgClientTrainer()
    bytes_fedavg, _, _ = trainer_fedavg.train(
        model=model_fedavg,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_fedavg,
        device=torch.device("cpu"),
    )

    # 2. DFPL with lambda_proto=0.0
    model_dfpl = copy.deepcopy(base_model)
    opt_dfpl = torch.optim.SGD(model_dfpl.parameters(), lr=0.01)
    trainer_dfpl = DFPLClientTrainer(lambda_proto=0.0)
    bytes_dfpl, _, _ = trainer_dfpl.train(
        model=model_dfpl,
        dataloader=loader,
        epochs=2,
        criterion=criterion,
        optimizer=opt_dfpl,
        device=torch.device("cpu"),
        global_prototypes={0: torch.randn(8), 1: torch.randn(8)},
    )

    vec_fedavg = bytes_to_vector(bytes_fedavg)
    vec_dfpl = bytes_to_vector(bytes_dfpl)

    assert torch.allclose(vec_fedavg, vec_dfpl, atol=1e-5)


def test_perfedavg_reduces_to_fedavg_when_alpha_zero() -> None:
    """Verifies that PerFedAvg with alpha_lr=0.0 on a single batch equals SGD(lr=beta_lr)."""
    torch.manual_seed(42)
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(x, y)
    single_batch_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    base_model = nn.Linear(4, 2)
    learning_rate = 0.015

    # 1. FedAvg with single batch SGD
    model_fedavg = copy.deepcopy(base_model)
    opt_fedavg = torch.optim.SGD(model_fedavg.parameters(), lr=learning_rate, momentum=0.0)
    trainer_fedavg = FedAvgClientTrainer()
    bytes_fedavg, _, _ = trainer_fedavg.train(
        model=model_fedavg,
        dataloader=single_batch_loader,
        epochs=1,
        criterion=criterion,
        optimizer=opt_fedavg,
        device=torch.device("cpu"),
    )

    # 2. Per-FedAvg with alpha_lr=0.0, beta_lr=learning_rate
    model_perfedavg = copy.deepcopy(base_model)
    opt_perfedavg = torch.optim.SGD(model_perfedavg.parameters(), lr=learning_rate, momentum=0.0)
    trainer_perfedavg = PerFedAvgClientTrainer(alpha_lr=0.0, beta_lr=learning_rate)
    bytes_perfedavg, _, _ = trainer_perfedavg.train(
        model=model_perfedavg,
        dataloader=single_batch_loader,
        epochs=1,
        criterion=criterion,
        optimizer=opt_perfedavg,
        device=torch.device("cpu"),
    )

    vec_fedavg = bytes_to_vector(bytes_fedavg)
    vec_perfedavg = bytes_to_vector(bytes_perfedavg)

    assert torch.allclose(vec_fedavg, vec_perfedavg, atol=1e-5)


def test_pfedgraph_reduces_to_uniform_average_at_large_lambda() -> None:
    """Verifies that pFedGraph with large lambda forces uniform adjacency W_ij = 1/K."""
    torch.manual_seed(42)
    num_clients = 4
    param_dim = 12

    client_vectors = [torch.randn(param_dim) for _ in range(num_clients)]
    client_bytes = [vector_to_bytes(v) for v in client_vectors]
    client_ids = [f"client_{i}" for i in range(num_clients)]
    # Provide asymmetric sample counts to ensure personalized aggregation is purely graph-based
    client_updates = [(b, (i + 1) * 100) for i, b in enumerate(client_bytes)]

    strategy = PFedGraphServerStrategy(alpha=1.0, lam=1e6, step_mode="target_interp")
    strategy.aggregate(
        global_model=None,
        client_updates=client_updates,
        round_id=1,
        client_ids=client_ids,
    )

    expected_uniform_average = torch.stack(client_vectors, dim=0).mean(dim=0)

    for cid in client_ids:
        pers_vec = bytes_to_vector(strategy.client_weights[cid])
        assert torch.allclose(pers_vec, expected_uniform_average, atol=1e-5)


# ==============================================================================
# Group 2: Analytical & Structural Oracles (Exact Math Verification)
# ==============================================================================


def test_fedprox_quadratic_oracle_closed_form() -> None:
    """Verifies that FedProxClientTrainer converges to the exact closed-form ridge solution."""
    torch.manual_seed(42)
    n_samples, n_features = 60, 4
    x = torch.randn(n_samples, n_features)
    w_true = torch.randn(n_features, 1)
    y = x @ w_true + 0.05 * torch.randn(n_samples, 1)

    w0 = torch.randn(n_features, 1)
    mu = 2.5

    # Analytical minimizer:
    # f(w) = (1/2)*||Xw - y||_2^2 + (mu/2)*||w - w0||_2^2
    # grad = X^T(Xw - y) + mu(w - w0) = 0 => (X^T X + mu I) w* = X^T y + mu w0
    h_matrix = x.T @ x + mu * torch.eye(n_features)
    b_vector = x.T @ y + mu * w0
    w_star = torch.linalg.solve(h_matrix, b_vector)

    model = LinearRegressionNoBias(in_features=n_features, initial_weights=w0)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    trainer = FedProxClientTrainer(mu=mu)
    global_weights = model_to_bytes(model)
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=500,
        criterion=HalfMSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        global_weights=global_weights,
    )

    w_trained = model.w.detach().T
    assert torch.allclose(w_trained, w_star, atol=1e-3)


def test_ditto_quadratic_oracle_closed_form() -> None:
    """Verifies that Ditto's personal model v_k converges to closed-form (X_k^T X_k + lambda I)^-1."""
    torch.manual_seed(42)
    n_samples, n_features = 60, 4
    x = torch.randn(n_samples, n_features)
    w_true = torch.randn(n_features, 1)
    y = x @ w_true + 0.05 * torch.randn(n_samples, 1)

    w0 = torch.randn(n_features, 1)
    lambda_penalty = 2.0

    # Closed-form personalized minimizer towards w0:
    # v* = (X^T X + lambda I)^(-1) (X^T y + lambda w0)
    h_matrix = x.T @ x + lambda_penalty * torch.eye(n_features)
    b_vector = x.T @ y + lambda_penalty * w0
    v_star = torch.linalg.solve(h_matrix, b_vector)

    model = LinearRegressionNoBias(in_features=n_features, initial_weights=w0)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)

    trainer = DittoClientTrainer(lambda_penalty=lambda_penalty, personalized_epochs=1)
    # Initialize trainer state with epochs=0 so global w remains at w0
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=0,
        criterion=HalfMSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert isinstance(trainer.personalized_model, LinearRegressionNoBias)
    trainer.personalized_optimizer = torch.optim.Adam(
        trainer.personalized_model.parameters(), lr=0.05
    )
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=0,
        criterion=HalfMSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        personalized_epochs=500,
    )

    v_trained = trainer.personalized_model.w.detach().T
    assert torch.allclose(v_trained, v_star, atol=1e-3)


def test_pfedgraph_two_cluster_block_diagonal_oracle() -> None:
    """Verifies that pFedGraph recovers a block-diagonal collaboration graph for clustered clients."""
    torch.manual_seed(42)
    v_base = torch.randn(80)
    # Cluster A: nodes 0 and 1; Cluster B: nodes 2 and 3
    vectors = torch.stack([v_base, v_base, -v_base, -v_base], dim=0)

    strategy = PFedGraphServerStrategy(alpha=0.5, lam=0.1, step_mode="target_interp")
    for _ in range(10):
        w_matrix = strategy.update_graph(vectors)

    # Intra-cluster weights
    intra_01 = w_matrix[0, 1].item()
    intra_23 = w_matrix[2, 3].item()

    # Inter-cluster weights
    inter_02 = w_matrix[0, 2].item()
    inter_03 = w_matrix[0, 3].item()
    inter_12 = w_matrix[1, 2].item()
    inter_13 = w_matrix[1, 3].item()

    assert intra_01 > 0.45
    assert intra_23 > 0.45
    assert all(w < 0.05 for w in (inter_02, inter_03, inter_12, inter_13))
    assert intra_01 > max(inter_02, inter_03)
    assert intra_23 > max(inter_12, inter_13)


def test_dfedpgp_push_sum_unbiased_consensus_oracle() -> None:
    """Verifies that directed Push-Sum achieves unbiased average consensus on an asymmetric graph."""
    num_clients = 5
    # Generate an asymmetric column-stochastic (and non-row-stochastic) mixing matrix
    mixing_matrix = build_directed_mixing_matrix(
        num_clients=num_clients,
        round_id=0,
        out_degree=2,
        seed=42,
        time_varying=False,
    )

    # Confirm matrix is column-stochastic but NOT row-stochastic or symmetric
    assert torch.allclose(mixing_matrix.sum(dim=0), torch.ones(num_clients), atol=1e-6)
    assert not torch.allclose(mixing_matrix, mixing_matrix.T)
    assert not torch.allclose(mixing_matrix.sum(dim=1), torch.ones(num_clients), atol=1e-2)

    # Arbitrary initial client scalar state vectors
    initial_states = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
    true_mean = initial_states.mean().item()

    states = initial_states.clone()
    mu_weights = torch.ones(num_clients, dtype=torch.float32)

    for _ in range(50):
        states, mu_weights = _push_sum_step(states, mixing_matrix, mu_weights)

    max_consensus_error = (states - true_mean).abs().max().item()
    assert max_consensus_error < 1e-4


def test_perfedavg_manual_autograd_oracle() -> None:
    """Verifies that PerFedAvgClientTrainer executes the exact First-Order MAML autograd update."""
    torch.manual_seed(42)
    x1, y1 = torch.randn(8, 4), torch.randint(0, 2, (8,))
    x2, y2 = torch.randn(8, 4), torch.randint(0, 2, (8,))

    model = nn.Linear(4, 2, bias=True)
    criterion = nn.CrossEntropyLoss()
    alpha, beta = 0.02, 0.01

    # 1. Manual first-order MAML derivation
    # Step A: Inner adaptation gradient on D1
    out1 = model(x1)
    loss1 = criterion(out1, y1)
    grads1 = torch.autograd.grad(loss1, model.parameters())

    adapted_params = {
        name: (param - alpha * g).detach().requires_grad_(True)
        for (name, param), g in zip(model.named_parameters(), grads1, strict=True)
    }

    # Step B: Outer gradient evaluated at w' on D2
    out2 = torch.func.functional_call(model, adapted_params, x2)
    loss2 = criterion(out2, y2)
    grads2 = torch.autograd.grad(loss2, list(adapted_params.values()))

    expected_updated_params = [
        param.detach().clone() - beta * g
        for param, g in zip(model.parameters(), grads2, strict=True)
    ]

    # 2. PerFedAvgClientTrainer execution
    dataset = TensorDataset(torch.cat([x1, x2], dim=0), torch.cat([y1, y2], dim=0))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    trainer = PerFedAvgClientTrainer(alpha_lr=alpha, beta_lr=beta)
    dummy_opt = torch.optim.SGD(model.parameters(), lr=beta)
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=dummy_opt,
        device=torch.device("cpu"),
    )

    trained_params = list(model.parameters())
    for p_exp, p_trained in zip(expected_updated_params, trained_params, strict=True):
        assert torch.allclose(p_exp, p_trained, atol=1e-6)


def test_dfpl_prototype_alignment_distance_oracle() -> None:
    """Verifies that lambda_proto > 0 strictly pulls intermediate features towards prototypes."""
    torch.manual_seed(42)
    x = torch.randn(30, 4)
    y = torch.randint(0, 2, (30,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=15, shuffle=False)

    initial_model = FeatureEmbeddingModel(in_features=4, hidden_dim=8, num_classes=2)
    prototypes = {0: torch.ones(8) * 4.0, 1: -torch.ones(8) * 4.0}

    # 1. Train with prototype alignment disabled (lambda_proto = 0.0)
    model_unaligned = copy.deepcopy(initial_model)
    opt_unaligned = torch.optim.SGD(model_unaligned.parameters(), lr=0.05)
    trainer_unaligned = DFPLClientTrainer(lambda_proto=0.0)
    trainer_unaligned.train(
        model=model_unaligned,
        dataloader=loader,
        epochs=5,
        criterion=nn.CrossEntropyLoss(),
        optimizer=opt_unaligned,
        device=torch.device("cpu"),
        global_prototypes=prototypes,
    )

    # 2. Train with prototype alignment enabled (lambda_proto = 2.0)
    model_aligned = copy.deepcopy(initial_model)
    opt_aligned = torch.optim.SGD(model_aligned.parameters(), lr=0.05)
    trainer_aligned = DFPLClientTrainer(lambda_proto=2.0)
    trainer_aligned.train(
        model=model_aligned,
        dataloader=loader,
        epochs=5,
        criterion=nn.CrossEntropyLoss(),
        optimizer=opt_aligned,
        device=torch.device("cpu"),
        global_prototypes=prototypes,
    )

    def compute_mean_prototype_distance(model: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            _, feats = model(x, return_features=True)
            target_proto = torch.stack([prototypes[int(label.item())] for label in y], dim=0)
            return float(((feats - target_proto) ** 2).sum(dim=-1).mean().item())

    distance_unaligned = compute_mean_prototype_distance(model_unaligned)
    distance_aligned = compute_mean_prototype_distance(model_aligned)

    assert distance_aligned < distance_unaligned


# ==============================================================================
# Group 3: Hyperparameter Monotonicity Tests
# ==============================================================================


def test_fedprox_drift_monotonicity() -> None:
    """Verifies that increasing mu in {0.0, 0.1, 1.0, 10.0} strictly decreases parameter drift."""
    torch.manual_seed(42)
    x = torch.randn(50, 4)
    y = torch.randint(0, 2, (50,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    initial_model = nn.Linear(4, 2)
    w_init = nn.utils.parameters_to_vector(initial_model.parameters()).detach()
    global_weights = model_to_bytes(initial_model)

    mu_values = [0.0, 0.1, 1.0, 10.0]
    drift_values: list[float] = []

    for mu in mu_values:
        model = copy.deepcopy(initial_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        trainer = FedProxClientTrainer(mu=mu)
        trainer.train(
            model=model,
            dataloader=loader,
            epochs=3,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            global_weights=global_weights,
        )
        w_updated = nn.utils.parameters_to_vector(model.parameters()).detach()
        drift = float(torch.norm(w_updated - w_init).item())
        drift_values.append(drift)

    for i in range(len(drift_values) - 1):
        assert drift_values[i] > drift_values[i + 1], (
            f"Drift not strictly decreasing: mu={mu_values[i]} ({drift_values[i]}) "
            f"<= mu={mu_values[i+1]} ({drift_values[i+1]})"
        )


def test_ditto_regularization_monotonicity() -> None:
    """Verifies that increasing lambda in {0.0, 0.1, 1.0, 10.0} strictly decreases ||v_k - w||_2."""
    torch.manual_seed(42)
    x = torch.randn(50, 4)
    y = torch.randint(0, 2, (50,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    initial_model = nn.Linear(4, 2)

    lambda_values = [0.0, 0.1, 1.0, 10.0]
    distance_values: list[float] = []

    for lam in lambda_values:
        model = copy.deepcopy(initial_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = DittoClientTrainer(lambda_penalty=lam, personalized_epochs=5)
        # With epochs=0, global model w remains fixed at reference initial weights
        trainer.train(
            model=model,
            dataloader=loader,
            epochs=0,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
        assert isinstance(trainer.personalized_model, nn.Module)
        w_ref = nn.utils.parameters_to_vector(model.parameters()).detach()
        v_pers = nn.utils.parameters_to_vector(trainer.personalized_model.parameters()).detach()
        distance = float(torch.norm(v_pers - w_ref).item())
        distance_values.append(distance)

    for i in range(len(distance_values) - 1):
        assert distance_values[i] > distance_values[i + 1], (
            f"Distance not strictly decreasing: lambda={lambda_values[i]} ({distance_values[i]}) "
            f"<= lambda={lambda_values[i+1]} ({distance_values[i+1]})"
        )


def test_dfpl_shuffled_label_alignment_control() -> None:
    """Verifies normalized MSE DFPL aligns features closer to true prototypes than shuffled labels."""
    torch.manual_seed(42)
    # Create separable 2-class data with feature norms much larger than 1.0
    x_pos = torch.randn(16, 4) + 3.0
    x_neg = torch.randn(16, 4) - 3.0
    x = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)], dim=0)
    y_shuffled = 1 - y  # Deterministic derangement swapping every label

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = FeatureEmbeddingModel(in_features=4, hidden_dim=8, num_classes=2)
    with torch.no_grad():
        model.backbone.weight.mul_(3.0)  # Ensure raw feature norm >> 1.0

    # Unit-norm target class prototypes
    p0 = F.normalize(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), p=2, dim=0)
    p1 = F.normalize(torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), p=2, dim=0)
    prototypes = {0: p0, 1: p1}

    trainer = DFPLClientTrainer(
        lambda_proto=2.0,
        normalize_prototypes=True,
        loss_type="mse",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=25,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        global_prototypes=prototypes,
    )

    model.eval()
    with torch.no_grad():
        _, feats = model(x, return_features=True)
        norm_feats = F.normalize(feats, p=2, dim=-1)
        raw_norms = torch.norm(feats, p=2, dim=-1).mean().item()
        true_targets = torch.stack([prototypes[int(lbl.item())] for lbl in y], dim=0)
        shuffled_targets = torch.stack([prototypes[int(lbl.item())] for lbl in y_shuffled], dim=0)

        dist_true = float(((norm_feats - true_targets) ** 2).sum(dim=-1).mean().item())
        dist_shuffled = float(((norm_feats - shuffled_targets) ** 2).sum(dim=-1).mean().item())

    # Normalized features must align tightly with true prototypes compared to shuffled-label control
    assert dist_true < dist_shuffled
    assert dist_true < 0.25
    # With L2 normalization inside MSE loss, raw feature norms do not collapse toward 1.0
    assert raw_norms > 2.0


def test_ditto_syncs_personalized_optimizer_lr_across_rounds() -> None:
    """Verifies DittoClientTrainer propagates updated round LR into personalized_optimizer."""
    torch.manual_seed(42)
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    model = nn.Linear(4, 2)
    trainer = DittoClientTrainer(lambda_penalty=0.1, personalized_epochs=1)

    opt_round1 = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=nn.CrossEntropyLoss(),
        optimizer=opt_round1,
        device=torch.device("cpu"),
    )
    assert trainer.personalized_optimizer is not None
    assert trainer.personalized_optimizer.param_groups[0]["lr"] == 0.1

    opt_round2 = torch.optim.SGD(model.parameters(), lr=0.005)
    trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=nn.CrossEntropyLoss(),
        optimizer=opt_round2,
        device=torch.device("cpu"),
    )
    assert trainer.personalized_optimizer.param_groups[0]["lr"] == 0.005
