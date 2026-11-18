"""Stress tests for numerical poisoning, non-finite payloads, and invalid sample counts."""

import pytest
import torch

from src.baselines import create_server_strategy
from src.baselines.dfpl.prototype import aggregate_prototypes
from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.models.cnn import SimpleCNN
from src.server.aggregation import fedavg_vectors
from src.server.server import FederatedServer
from src.transport.federated_pb2 import TrainResponse
from src.utils.serialization import (
    bytes_to_vector,
    model_to_bytes,
    model_to_vector,
    vector_to_bytes,
)


def _poison_payload(base_bytes: bytes, index: int, value: float) -> bytes:
    """Returns a valid-length float32 weight payload with a single poisoned entry."""
    vec = bytes_to_vector(base_bytes).clone()
    vec[index] = value
    return vector_to_bytes(vec)


def test_server_rejects_nan_and_inf_weight_payloads() -> None:
    """Verifies aggregate_responses discards NaN/Inf weight payloads and aggregates healthy updates."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    try:
        assert server.global_model is not None
        healthy_payload = model_to_bytes(SimpleCNN(seed=10))
        nan_payload = _poison_payload(healthy_payload, index=0, value=float("nan"))
        pos_inf_payload = _poison_payload(healthy_payload, index=7, value=float("inf"))
        neg_inf_payload = _poison_payload(healthy_payload, index=15, value=float("-inf"))

        responses = [
            TrainResponse(node_id="client_healthy", success=True, weights=healthy_payload, num_samples=16),
            TrainResponse(node_id="client_nan", success=True, weights=nan_payload, num_samples=16),
            TrainResponse(node_id="client_pos_inf", success=True, weights=pos_inf_payload, num_samples=16),
            TrainResponse(node_id="client_neg_inf", success=True, weights=neg_inf_payload, num_samples=16),
        ]

        result = server.aggregate_responses(responses, round_id=1)
        assert result is not None
        result_vec = bytes_to_vector(result)
        healthy_vec = bytes_to_vector(healthy_payload)
        assert bool(torch.isfinite(result_vec).all().item())
        assert torch.allclose(result_vec, healthy_vec, atol=1e-6)

        global_vec = model_to_vector(server.global_model)
        assert bool(torch.isfinite(global_vec).all().item())
        assert torch.allclose(global_vec, healthy_vec, atol=1e-6)
    finally:
        server.stop(grace=0)


def test_server_preserves_global_model_when_all_updates_non_finite() -> None:
    """Verifies aggregate_responses returns None and preserves global weights if all updates are non-finite."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    try:
        assert server.global_model is not None
        pre_round_weights = model_to_bytes(server.global_model)
        nan_payload = _poison_payload(pre_round_weights, index=3, value=float("nan"))
        inf_payload = _poison_payload(pre_round_weights, index=11, value=float("inf"))

        responses = [
            TrainResponse(node_id="client_nan", success=True, weights=nan_payload, num_samples=10),
            TrainResponse(node_id="client_inf", success=True, weights=inf_payload, num_samples=10),
        ]

        result = server.aggregate_responses(responses, round_id=1)
        assert result is None
        post_round_weights = model_to_bytes(server.global_model)
        assert bool(post_round_weights == pre_round_weights), (
            "Global model weights were mutated despite all updates being non-finite"
        )
    finally:
        server.stop(grace=0)


def test_server_rejects_negative_sample_counts() -> None:
    """Verifies negative sample counts are discarded by the server and rejected by fedavg_vectors."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    try:
        assert server.global_model is not None
        healthy_payload = model_to_bytes(SimpleCNN(seed=10))
        adversarial_payload = model_to_bytes(SimpleCNN(seed=99))

        responses = [
            TrainResponse(node_id="client_good", success=True, weights=healthy_payload, num_samples=10),
            TrainResponse(node_id="client_negative", success=True, weights=adversarial_payload, num_samples=-10),
        ]

        result = server.aggregate_responses(responses, round_id=1)
        assert result is not None
        assert torch.allclose(
            bytes_to_vector(result),
            bytes_to_vector(healthy_payload),
            atol=1e-6,
        )
    finally:
        server.stop(grace=0)

    v1 = torch.ones(8, dtype=torch.float32)
    v2 = torch.full((8,), 5.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-negative"):
        fedavg_vectors([(v1, 10), (v2, -5)])


@pytest.mark.parametrize("normalize", [False, True])
def test_dfpl_aggregate_prototypes_filters_non_finite_vectors(normalize: bool) -> None:
    """Verifies aggregate_prototypes ignores NaN/Inf class prototypes and excludes fully poisoned classes."""
    gen = torch.Generator().manual_seed(123)
    p0_clean = torch.randn(16, generator=gen, dtype=torch.float32)
    p1_clean = torch.randn(16, generator=gen, dtype=torch.float32)

    p0_nan = p0_clean.clone()
    p0_nan[2] = float("nan")
    p2_inf = torch.full((16,), float("inf"), dtype=torch.float32)
    p2_neg_inf = torch.full((16,), float("-inf"), dtype=torch.float32)

    client_prototypes = [
        {0: p0_clean, 1: p1_clean, 2: p2_inf},
        {0: p0_nan, 2: p2_neg_inf},
    ]
    client_counts = [
        {0: 10, 1: 8, 2: 5},
        {0: 50, 2: 20},
    ]

    aggregated = aggregate_prototypes(
        client_prototypes=client_prototypes,
        client_sample_counts=client_counts,
        normalize=normalize,
    )

    # Class 2 had only non-finite prototypes and must be omitted
    assert set(aggregated.keys()) == {0, 1}

    for cls_id, proto in aggregated.items():
        assert bool(torch.isfinite(proto).all().item()), f"Class {cls_id} prototype is non-finite"

    expected_p0 = p0_clean / torch.norm(p0_clean, p=2) if normalize else p0_clean
    expected_p1 = p1_clean / torch.norm(p1_clean, p=2) if normalize else p1_clean
    assert torch.allclose(aggregated[0], expected_p0, atol=1e-5)
    assert torch.allclose(aggregated[1], expected_p1, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["fedavg", "pfedgraph", "dfedpgp", "dfpl"])
def test_all_stateful_strategies_immune_to_nan_payload_via_server(algorithm: str) -> None:
    """Verifies poisoned NaN payloads are filtered before reaching any server aggregation strategy."""
    strategy = create_server_strategy(algorithm)
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=strategy,
    )
    try:
        assert server.global_model is not None
        initial_vec = model_to_vector(server.global_model).clone()

        w_good_1 = model_to_bytes(SimpleCNN(seed=11))
        w_good_2 = model_to_bytes(SimpleCNN(seed=22))
        w_poisoned = _poison_payload(w_good_1, index=5, value=float("nan"))

        responses = [
            TrainResponse(node_id="client_0", success=True, weights=w_good_1, num_samples=12),
            TrainResponse(node_id="client_1", success=True, weights=w_good_2, num_samples=18),
            TrainResponse(node_id="client_bad", success=True, weights=w_poisoned, num_samples=30),
        ]

        result = server.aggregate_responses(responses, round_id=1)
        assert result is not None

        result_vec = bytes_to_vector(result)
        assert bool(torch.isfinite(result_vec).all().item())

        updated_vec = model_to_vector(server.global_model)
        assert bool(torch.isfinite(updated_vec).all().item())
        assert not torch.allclose(updated_vec, initial_vec)
    finally:
        server.stop(grace=0)
