"""Property-based stress tests verifying mathematical invariants across federated components."""

from hypothesis import given, settings
import hypothesis.strategies as st
import torch

from src.baselines.dfedpgp.topology import build_directed_mixing_matrix
from src.baselines.dfpl.prototype import (
    aggregate_prototypes,
    bytes_to_prototypes,
    prototypes_to_bytes,
)
from src.server.aggregation import fedavg_vectors
from src.utils.serialization import bytes_to_vector, vector_to_bytes


@settings(max_examples=50, deadline=None)
@given(
    num_nodes=st.integers(min_value=2, max_value=20),
    out_degree=st.integers(min_value=0, max_value=25),
    seed=st.integers(min_value=0, max_value=100_000),
    round_id=st.integers(min_value=0, max_value=100),
    time_varying=st.booleans(),
)
def test_fuzz_mixing_matrix_column_stochasticity(
    num_nodes: int,
    out_degree: int,
    seed: int,
    round_id: int,
    time_varying: bool,
) -> None:
    """Verifies that generated mixing matrices satisfy column-stochasticity and non-negativity."""
    matrix = build_directed_mixing_matrix(
        num_clients=num_nodes,
        round_id=round_id,
        out_degree=out_degree,
        seed=seed,
        time_varying=time_varying,
    )

    # Dimension invariant
    assert matrix.shape == (num_nodes, num_nodes)

    # Column stochasticity invariant: sum_i W_{i, j} == 1.0 (within 1e-5 tolerance)
    col_sums = matrix.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones(num_nodes), atol=1e-5), (
        f"Column sum violation: max diff = {(col_sums - 1.0).abs().max()}"
    )

    # Element-wise non-negativity: W_{i, j} >= 0
    assert (matrix >= 0.0).all(), "Mixing matrix contains negative elements"

    # Strictly positive self-loops: W_{j, j} > 0
    diag = torch.diagonal(matrix)
    assert (diag > 0.0).all(), "Mixing matrix contains non-positive diagonal element"


@settings(max_examples=50, deadline=None)
@given(
    vector_dim=st.integers(min_value=1, max_value=64),
    num_clients=st.integers(min_value=1, max_value=10),
    client_weights=st.lists(
        st.integers(min_value=1, max_value=5000), min_size=1, max_size=10
    ),
    scale_factor=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_fuzz_fedavg_convex_combination_and_scaling(
    vector_dim: int,
    num_clients: int,
    client_weights: list[int],
    scale_factor: int,
    seed: int,
) -> None:
    """Verifies unanimity and sample-count scale invariance in FedAvg aggregation."""
    k = min(num_clients, len(client_weights))
    weights = client_weights[:k]

    # 1. Unanimity invariant: if all clients report identical vector v, fedavg_vectors returns v
    rng = torch.Generator().manual_seed(seed)
    consensus_vec = torch.randn(vector_dim, generator=rng, dtype=torch.float32)
    unanimous_updates = [(consensus_vec.clone(), w) for w in weights]
    agg_unanimous = fedavg_vectors(unanimous_updates)
    assert torch.allclose(agg_unanimous, consensus_vec, atol=1e-5), (
        "FedAvg unanimity invariant violated"
    )

    # 2. Scale invariance: multiplying all sample counts by scalar c >= 2 yields identical vector
    heterogeneous_vectors = [
        torch.randn(vector_dim, generator=rng, dtype=torch.float32) for _ in range(k)
    ]
    base_updates = list(zip(heterogeneous_vectors, weights))
    scaled_updates = [(vec, w * scale_factor) for vec, w in base_updates]

    base_agg = fedavg_vectors(base_updates)
    scaled_agg = fedavg_vectors(scaled_updates)
    assert torch.allclose(base_agg, scaled_agg, atol=1e-5), (
        "FedAvg scale invariance invariant violated"
    )


@settings(max_examples=50, deadline=None)
@given(
    num_classes=st.integers(min_value=1, max_value=8),
    feature_dim=st.integers(min_value=2, max_value=32),
    client_count=st.integers(min_value=1, max_value=6),
    normalize=st.booleans(),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_fuzz_prototype_aggregation_unanimity_and_skew(
    num_classes: int,
    feature_dim: int,
    client_count: int,
    normalize: bool,
    seed: int,
) -> None:
    """Verifies prototype aggregation unanimity, unit-norm invariance, and skew robustness."""
    rng = torch.Generator().manual_seed(seed)
    canonical_prototypes: dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        vec = torch.randn(feature_dim, generator=rng, dtype=torch.float32)
        if torch.norm(vec, p=2) < 1e-2:
            vec = vec + 1.0
        canonical_prototypes[c] = vec

    client_prototypes: list[dict[int, torch.Tensor]] = []
    client_counts: list[dict[int, int]] = []
    present_classes: set[int] = set()

    for i in range(client_count):
        c_protos: dict[int, torch.Tensor] = {}
        c_counts: dict[int, int] = {}
        for c in range(num_classes):
            include_class = (seed + i * 13 + c * 7) % 2 == 0
            if include_class:
                c_protos[c] = canonical_prototypes[c].clone()
                c_counts[c] = 1 + ((seed + i * 17 + c * 3) % 200)
                present_classes.add(c)
        client_prototypes.append(c_protos)
        client_counts.append(c_counts)

    if not present_classes:
        client_prototypes[0][0] = canonical_prototypes[0].clone()
        client_counts[0][0] = 50
        present_classes.add(0)

    # Must handle label skew and missing classes without KeyError or ZeroDivisionError
    aggregated = aggregate_prototypes(
        client_prototypes=client_prototypes,
        client_sample_counts=client_counts,
        normalize=normalize,
    )

    assert set(aggregated.keys()) == present_classes

    for c, agg_proto in aggregated.items():
        base_proto = canonical_prototypes[c]
        if normalize:
            norm = torch.norm(agg_proto, p=2)
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5), (
                f"Prototype for class {c} is not unit norm: got {norm.item()}"
            )
            expected_norm = base_proto / torch.norm(base_proto, p=2)
            assert torch.allclose(agg_proto, expected_norm, atol=1e-5), (
                f"Normalized unanimity failed for class {c}"
            )
        else:
            assert torch.allclose(agg_proto, base_proto, atol=1e-5), (
                f"Unanimity failed for class {c}"
            )


@settings(max_examples=50, deadline=None)
@given(
    raw_floats=st.lists(
        st.floats(
            min_value=-1e5,
            max_value=1e5,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=0,
        max_size=256,
    )
)
def test_fuzz_vector_serialization_roundtrip(raw_floats: list[float]) -> None:
    """Verifies bit-exact serialization roundtrip for arbitrary 1D parameter vectors."""
    original = torch.tensor(raw_floats, dtype=torch.float32)
    payload = vector_to_bytes(original)
    reconstructed = bytes_to_vector(payload)

    if original.numel() == 0:
        assert reconstructed.numel() == 0
    else:
        assert original.shape == reconstructed.shape
        assert torch.equal(original, reconstructed)


@settings(max_examples=50, deadline=None)
@given(
    class_ids=st.sets(st.integers(min_value=0, max_value=99), min_size=0, max_size=10),
    dim=st.integers(min_value=1, max_value=64),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_fuzz_prototype_serialization_roundtrip(
    class_ids: set[int],
    dim: int,
    seed: int,
) -> None:
    """Verifies serialization roundtrip for prototype dictionary mappings."""
    rng = torch.Generator().manual_seed(seed)
    prototypes: dict[int, torch.Tensor] = {
        c: torch.randn(dim, generator=rng, dtype=torch.float32)
        for c in sorted(class_ids)
    }

    payload = prototypes_to_bytes(prototypes)
    reconstructed = bytes_to_prototypes(payload)

    if not prototypes:
        assert payload == b""
        assert reconstructed == {}
    else:
        assert set(reconstructed.keys()) == set(prototypes.keys())
        for c in prototypes:
            assert prototypes[c].shape == reconstructed[c].shape
            assert torch.allclose(prototypes[c], reconstructed[c], atol=1e-6)

