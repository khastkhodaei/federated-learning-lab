import random
from collections import deque

import pytest
import torch
from torch import nn

from src.baselines.dfedpgp import (
    DEFAULT_HEAD_PREFIXES,
    DFedPGPServerStrategy,
    ModelPartition,
    build_directed_mixing_matrix,
    partition_model,
    split_named_parameters,
)
from src.models.cnn import SimpleCNN
from src.utils.serialization import (
    bytes_to_vector,
    model_to_vector,
    vector_to_bytes,
    vector_to_model,
)


def test_partition_simple_cnn_isolates_backbone_and_head() -> None:
    model = SimpleCNN(seed=0)
    partition = partition_model(model)

    assert isinstance(partition, ModelPartition)
    assert DEFAULT_HEAD_PREFIXES == ("fc_out", "classifier", "head")
    assert partition.backbone_names == (
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "fc1.weight",
        "fc1.bias",
    )
    assert partition.head_names == ("fc_out.weight", "fc_out.bias")
    assert partition.backbone_dim == 543808
    assert partition.head_dim == 1290
    assert partition.total_dim == 545098
    assert partition.backbone_dim + partition.head_dim == partition.total_dim
    assert partition.head_mask.dtype == torch.bool
    assert partition.head_mask.device.type == "cpu"
    assert partition.head_mask.shape == (545098,)
    assert int(partition.head_mask.sum().item()) == 1290


class _NonContiguousHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 3, bias=False)
        self.head = nn.Linear(3, 2, bias=True)
        self.head_aux = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.stem.weight.copy_(torch.arange(1, 13, dtype=torch.float32).view(3, 4))
            self.head.weight.copy_(torch.arange(101, 107, dtype=torch.float32).view(2, 3))
            self.head.bias.copy_(torch.tensor([201.0, 202.0], dtype=torch.float32))
            self.head_aux.weight.copy_(torch.arange(301, 305, dtype=torch.float32).view(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_aux(self.head(self.stem(x)))


def test_split_and_merge_roundtrip_simple_cnn() -> None:
    model = SimpleCNN(seed=7)
    partition = partition_model(model)
    vec = model_to_vector(model)

    backbone_vec, head_vec = partition.split_vector(vec)

    assert backbone_vec.ndim == 1
    assert head_vec.ndim == 1
    assert backbone_vec.numel() == 543808
    assert head_vec.numel() == 1290
    expected_head = torch.cat(
        [model.fc_out.weight.detach().flatten(), model.fc_out.bias.detach().flatten()]
    )
    assert torch.equal(head_vec, expected_head)

    reconstructed = partition.merge_vectors(backbone_vec, head_vec)
    assert torch.equal(reconstructed, vec)


def test_split_and_merge_non_contiguous_head_preserves_personal_head() -> None:
    model = _NonContiguousHeadModel()
    partition = partition_model(model)

    assert partition.backbone_names == ("stem.weight", "head_aux.weight")
    assert partition.head_names == ("head.weight", "head.bias")
    assert partition.backbone_dim == 16
    assert partition.head_dim == 8
    assert partition.total_dim == 24

    vec = model_to_vector(model)
    backbone_vec, head_vec = partition.split_vector(vec)

    expected_backbone = torch.tensor(
        [*range(1, 13), *range(301, 305)],
        dtype=torch.float32,
    )
    expected_head = torch.tensor(
        [*range(101, 107), 201.0, 202.0],
        dtype=torch.float32,
    )
    assert torch.equal(backbone_vec, expected_backbone)
    assert torch.equal(head_vec, expected_head)
    assert torch.equal(partition.merge_vectors(backbone_vec, head_vec), vec)

    updated_backbone = torch.full((16,), -5.0, dtype=torch.float32)
    merged = partition.merge_vectors(updated_backbone, head_vec)
    vector_to_model(merged, model)

    assert torch.equal(model.stem.weight.detach(), torch.full((3, 4), -5.0))
    assert torch.equal(model.head_aux.weight.detach(), torch.full((2, 2), -5.0))
    assert torch.equal(
        model.head.weight.detach(),
        torch.arange(101, 107, dtype=torch.float32).view(2, 3),
    )
    assert torch.equal(model.head.bias.detach(), torch.tensor([201.0, 202.0]))


def test_split_named_parameters_returns_disjoint_live_parameters() -> None:
    model = SimpleCNN(seed=3)
    backbone_params, head_params = split_named_parameters(model)

    expected_backbone = [
        model.conv1.weight,
        model.conv1.bias,
        model.conv2.weight,
        model.conv2.bias,
        model.fc1.weight,
        model.fc1.bias,
    ]
    expected_head = [model.fc_out.weight, model.fc_out.bias]

    assert len(backbone_params) == 6
    assert len(head_params) == 2
    assert all(actual is expected for actual, expected in zip(backbone_params, expected_backbone))
    assert all(actual is expected for actual, expected in zip(head_params, expected_head))
    assert {id(p) for p in backbone_params}.isdisjoint({id(p) for p in head_params})
    assert sum(p.numel() for p in backbone_params) == 543808
    assert sum(p.numel() for p in head_params) == 1290

    with torch.no_grad():
        head_params[1].fill_(42.0)
    assert torch.equal(model.fc_out.bias.detach(), torch.full((10,), 42.0))


def test_partition_fails_on_empty_model() -> None:
    empty_model = nn.Identity()

    with pytest.raises(ValueError, match="no parameters"):
        partition_model(empty_model)

    with pytest.raises(ValueError, match="no parameters"):
        split_named_parameters(empty_model)


def test_partition_fails_when_head_prefixes_match_zero_or_all_parameters() -> None:
    model = SimpleCNN(seed=0)

    with pytest.raises(ValueError, match="personal_head"):
        partition_model(model, head_prefixes=("nonexistent",))

    with pytest.raises(ValueError, match="personal_head"):
        split_named_parameters(model, head_prefixes=("nonexistent",))

    all_prefixes = ("conv1", "conv2", "fc1", "fc_out")
    with pytest.raises(ValueError, match="shared_backbone"):
        partition_model(model, head_prefixes=all_prefixes)

    with pytest.raises(ValueError, match="shared_backbone"):
        split_named_parameters(model, head_prefixes=all_prefixes)


def test_split_and_merge_vectors_reject_invalid_rank_and_length() -> None:
    model = _NonContiguousHeadModel()
    partition = partition_model(model)
    backbone_vec = torch.zeros(partition.backbone_dim, dtype=torch.float32)
    head_vec = torch.zeros(partition.head_dim, dtype=torch.float32)

    with pytest.raises(ValueError):
        partition.split_vector(torch.zeros((1, partition.total_dim), dtype=torch.float32))

    with pytest.raises(ValueError):
        partition.split_vector(torch.zeros(partition.total_dim - 1, dtype=torch.float32))

    with pytest.raises(ValueError):
        partition.merge_vectors(backbone_vec.unsqueeze(0), head_vec)

    with pytest.raises(ValueError):
        partition.merge_vectors(torch.zeros(partition.backbone_dim - 1, dtype=torch.float32), head_vec)

    with pytest.raises(ValueError):
        partition.merge_vectors(backbone_vec, head_vec.unsqueeze(0))

    with pytest.raises(ValueError):
        partition.merge_vectors(backbone_vec, torch.zeros(partition.head_dim + 1, dtype=torch.float32))


def test_partition_matches_exact_parameter_name_and_custom_prefix() -> None:
    class _DirectHeadParamModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(3, 2, bias=False)
            self.head = nn.Parameter(torch.tensor([7.0, 8.0]))
            self.head_scale = nn.Parameter(torch.tensor([1.0]))

    model = _DirectHeadParamModel()
    partition = partition_model(model, head_prefixes=("head",))

    assert partition.backbone_names == ("head_scale", "encoder.weight")
    assert partition.head_names == ("head",)
    assert partition.backbone_dim == 7
    assert partition.head_dim == 2
    assert partition.total_dim == 9
    assert torch.equal(
        partition.head_mask,
        torch.tensor([True, True, False, False, False, False, False, False, False]),
    )


@pytest.mark.parametrize("num_clients", [1, 2, 3, 5])
@pytest.mark.parametrize("out_degree", [0, 1, 2])
def test_build_directed_mixing_matrix_column_stochastic_and_self_loops(
    num_clients: int,
    out_degree: int,
) -> None:
    matrix = build_directed_mixing_matrix(
        num_clients=num_clients,
        round_id=1,
        out_degree=out_degree,
        seed=42,
    )

    assert matrix.shape == (num_clients, num_clients)
    assert matrix.dtype == torch.float32
    assert matrix.device.type == "cpu"
    assert torch.all(matrix >= 0.0)
    assert torch.all(torch.diag(matrix) > 0.0)
    assert torch.allclose(matrix.sum(dim=0), torch.ones(num_clients, dtype=torch.float32), atol=1e-6)


def _is_strongly_connected(support: torch.Tensor) -> bool:
    num_nodes = support.shape[0]
    for start in range(num_nodes):
        visited = {start}
        queue: deque[int] = deque([start])
        while queue:
            src = queue.popleft()
            for dst in range(num_nodes):
                if bool(support[dst, src].item()) and dst not in visited:
                    visited.add(dst)
                    queue.append(dst)
        if len(visited) != num_nodes:
            return False
    return True


@pytest.mark.parametrize("num_clients", [3, 4, 6])
def test_build_directed_mixing_matrix_asymmetric_and_strongly_connected(num_clients: int) -> None:
    matrix = build_directed_mixing_matrix(
        num_clients=num_clients,
        round_id=1,
        out_degree=1,
        seed=42,
    )

    assert not torch.allclose(matrix, matrix.T)
    for col in range(num_clients):
        nonzero = matrix[matrix[:, col] > 0.0, col]
        assert nonzero.numel() == 2
        assert torch.allclose(nonzero, torch.full((2,), 0.5, dtype=torch.float32))
    assert _is_strongly_connected(matrix > 0.0)


def test_build_directed_mixing_matrix_multi_neighbor_weights_and_connectivity() -> None:
    matrix = build_directed_mixing_matrix(
        num_clients=5,
        round_id=1,
        out_degree=2,
        seed=7,
    )

    expected_weight = 1.0 / 3.0
    for col in range(5):
        nonzero = matrix[matrix[:, col] > 0.0, col]
        assert nonzero.numel() == 3
        assert torch.allclose(nonzero, torch.full((3,), expected_weight, dtype=torch.float32))
    assert _is_strongly_connected(matrix > 0.0)


def test_build_directed_mixing_matrix_deterministic_and_isolated_rng() -> None:
    random_state_before = random.getstate()
    torch_state_before = torch.get_rng_state()

    m1 = build_directed_mixing_matrix(
        num_clients=5,
        round_id=3,
        out_degree=2,
        seed=42,
        time_varying=True,
    )

    assert random.getstate() == random_state_before
    assert torch.equal(torch.get_rng_state(), torch_state_before)

    m2 = build_directed_mixing_matrix(
        num_clients=5,
        round_id=3,
        out_degree=2,
        seed=42,
        time_varying=True,
    )
    assert torch.equal(m1, m2)

    m_other_seed = build_directed_mixing_matrix(
        num_clients=5,
        round_id=3,
        out_degree=2,
        seed=99,
        time_varying=True,
    )
    assert not torch.equal(m1, m_other_seed)


def test_build_directed_mixing_matrix_time_varying_vs_static_rounds() -> None:
    tv_r1 = build_directed_mixing_matrix(
        num_clients=6,
        round_id=1,
        out_degree=1,
        seed=42,
        time_varying=True,
    )
    tv_r2 = build_directed_mixing_matrix(
        num_clients=6,
        round_id=2,
        out_degree=1,
        seed=42,
        time_varying=True,
    )
    assert not torch.equal(tv_r1, tv_r2)

    static_r1 = build_directed_mixing_matrix(
        num_clients=6,
        round_id=1,
        out_degree=1,
        seed=42,
        time_varying=False,
    )
    static_r2 = build_directed_mixing_matrix(
        num_clients=6,
        round_id=2,
        out_degree=1,
        seed=42,
        time_varying=False,
    )
    assert torch.equal(static_r1, static_r2)


def test_build_directed_mixing_matrix_edge_cases() -> None:
    assert torch.equal(
        build_directed_mixing_matrix(num_clients=1, round_id=0, out_degree=3),
        torch.ones((1, 1), dtype=torch.float32),
    )
    assert torch.equal(
        build_directed_mixing_matrix(num_clients=4, round_id=1, out_degree=0),
        torch.eye(4, dtype=torch.float32),
    )
    assert torch.equal(
        build_directed_mixing_matrix(num_clients=4, round_id=1, out_degree=4),
        torch.full((4, 4), 0.25, dtype=torch.float32),
    )
    assert torch.equal(
        build_directed_mixing_matrix(num_clients=4, round_id=1, out_degree=10),
        torch.full((4, 4), 0.25, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("num_clients", "round_id", "out_degree"),
    [
        (0, 1, 1),
        (-2, 1, 1),
        (3, -1, 1),
        (3, 1, -1),
    ],
)
def test_build_directed_mixing_matrix_rejects_invalid_arguments(
    num_clients: int,
    round_id: int,
    out_degree: int,
) -> None:
    with pytest.raises(ValueError):
        build_directed_mixing_matrix(
            num_clients=num_clients,
            round_id=round_id,
            out_degree=out_degree,
        )


class _TwoLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def test_dfedpgp_server_strategy_partial_push_sum_and_head_privacy() -> None:
    strategy = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)
    global_model = _TwoLayerNet()
    partition = partition_model(global_model)

    backbones = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )
    heads = torch.tensor(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=torch.float32,
    )
    sample_counts = [10, 20, 30]
    client_ids = ["client_0", "client_1", "client_2"]
    full_vectors = [partition.merge_vectors(backbones[i], heads[i]) for i in range(3)]
    client_updates = [
        (vector_to_bytes(full_vectors[i]), sample_counts[i]) for i in range(3)
    ]

    global_bytes = strategy.aggregate(
        global_model=global_model,
        client_updates=client_updates,
        round_id=1,
        client_ids=client_ids,
    )

    expected_p = build_directed_mixing_matrix(
        num_clients=3,
        round_id=1,
        out_degree=1,
        seed=42,
        time_varying=True,
    )
    expected_mu = expected_p @ torch.ones(3, dtype=torch.float32)
    expected_z = (expected_p @ backbones) / expected_mu.unsqueeze(1)

    assert strategy.mixing_matrix is not None
    assert torch.equal(strategy.mixing_matrix, expected_p)
    assert strategy.push_sum_weights is not None
    assert torch.allclose(strategy.push_sum_weights, expected_mu, atol=1e-6)
    assert torch.isclose(strategy.push_sum_weights.sum(), torch.tensor(3.0), atol=1e-6)

    for i, cid in enumerate(client_ids):
        pers_vec = bytes_to_vector(strategy.client_weights[cid])
        pers_backbone, pers_head = partition.split_vector(pers_vec)
        assert torch.allclose(pers_backbone, expected_z[i], atol=1e-6)
        assert torch.equal(pers_head, heads[i])

    expected_global = (
        10.0 * full_vectors[0] + 20.0 * full_vectors[1] + 30.0 * full_vectors[2]
    ) / 60.0
    assert torch.allclose(bytes_to_vector(global_bytes), expected_global, atol=1e-6)
    assert torch.allclose(model_to_vector(global_model), expected_global, atol=1e-6)


def test_dfedpgp_multi_round_non_uniform_mu_and_consensus_fixed_point() -> None:
    strategy = DFedPGPServerStrategy(out_degree=2, seed=42, time_varying=True)
    global_model = _TwoLayerNet()
    partition = partition_model(global_model)
    client_ids = ["client_0", "client_1", "client_2", "client_3"]

    heads = [
        torch.tensor([10.0 + i, 20.0 + i, 30.0 + i], dtype=torch.float32)
        for i in range(4)
    ]
    expected_mu = torch.ones(4, dtype=torch.float32)

    for round_id in (1, 2, 3):
        updates = [
            (
                vector_to_bytes(
                    partition.merge_vectors(
                        torch.full((4,), float(round_id * 10 + i), dtype=torch.float32),
                        heads[i],
                    )
                ),
                10,
            )
            for i in range(4)
        ]
        strategy.aggregate(
            global_model=global_model,
            client_updates=updates,
            round_id=round_id,
            client_ids=client_ids,
        )
        p_round = build_directed_mixing_matrix(
            num_clients=4,
            round_id=round_id,
            out_degree=2,
            seed=42,
            time_varying=True,
        )
        expected_mu = p_round @ expected_mu

    assert strategy.push_sum_weights is not None
    assert torch.allclose(strategy.push_sum_weights, expected_mu, atol=1e-6)
    assert not torch.allclose(strategy.push_sum_weights, torch.ones(4), atol=1e-4)
    assert torch.isclose(strategy.push_sum_weights.sum(), torch.tensor(4.0), atol=1e-5)

    consensus_backbone = torch.tensor([3.5, -1.25, 0.75, 4.0], dtype=torch.float32)
    consensus_updates = [
        (
            vector_to_bytes(partition.merge_vectors(consensus_backbone, heads[i])),
            10,
        )
        for i in range(4)
    ]
    strategy.aggregate(
        global_model=global_model,
        client_updates=consensus_updates,
        round_id=4,
        client_ids=client_ids,
    )

    assert strategy.push_sum_weights is not None
    assert torch.isclose(strategy.push_sum_weights.sum(), torch.tensor(4.0), atol=1e-5)
    for i, cid in enumerate(client_ids):
        pers_vec = bytes_to_vector(strategy.client_weights[cid])
        pers_backbone, pers_head = partition.split_vector(pers_vec)
        assert torch.allclose(pers_backbone, consensus_backbone, atol=1e-5)
        assert torch.equal(pers_head, heads[i])


def test_dfedpgp_client_arrival_order_permutation_invariance() -> None:
    model_a = _TwoLayerNet()
    model_b = _TwoLayerNet()
    partition = partition_model(model_a)

    client_ids = ["client_0", "client_1", "client_2", "client_3"]
    updates_ordered = [
        (
            vector_to_bytes(
                partition.merge_vectors(
                    torch.full((4,), float(i + 1), dtype=torch.float32),
                    torch.full((3,), float((i + 1) * 10), dtype=torch.float32),
                )
            ),
            (i + 1) * 5,
        )
        for i in range(4)
    ]

    strat_a = DFedPGPServerStrategy(out_degree=2, seed=42, time_varying=True)
    strat_b = DFedPGPServerStrategy(out_degree=2, seed=42, time_varying=True)

    global_a_r1 = strat_a.aggregate(
        global_model=model_a,
        client_updates=updates_ordered,
        round_id=1,
        client_ids=client_ids,
    )
    perm_r1 = [3, 1, 0, 2]
    global_b_r1 = strat_b.aggregate(
        global_model=model_b,
        client_updates=[updates_ordered[idx] for idx in perm_r1],
        round_id=1,
        client_ids=[client_ids[idx] for idx in perm_r1],
    )

    assert strat_a.client_ids == client_ids
    assert strat_b.client_ids == client_ids
    assert strat_a.mixing_matrix is not None and strat_b.mixing_matrix is not None
    assert torch.equal(strat_a.mixing_matrix, strat_b.mixing_matrix)
    assert strat_a.push_sum_weights is not None and strat_b.push_sum_weights is not None
    assert torch.equal(strat_a.push_sum_weights, strat_b.push_sum_weights)
    assert strat_a.client_weights == strat_b.client_weights
    assert global_a_r1 == global_b_r1

    perm_r2 = [2, 0, 3, 1]
    global_a_r2 = strat_a.aggregate(
        global_model=model_a,
        client_updates=updates_ordered,
        round_id=2,
        client_ids=client_ids,
    )
    global_b_r2 = strat_b.aggregate(
        global_model=model_b,
        client_updates=[updates_ordered[idx] for idx in perm_r2],
        round_id=2,
        client_ids=[client_ids[idx] for idx in perm_r2],
    )

    assert strat_a.client_ids == strat_b.client_ids
    assert torch.equal(strat_a.push_sum_weights, strat_b.push_sum_weights)
    assert strat_a.client_weights == strat_b.client_weights
    assert global_a_r2 == global_b_r2


def test_dfedpgp_server_strategy_global_model_none_full_vector_push_sum() -> None:
    strategy = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)
    vectors = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=torch.float32,
    )
    sample_counts = [10, 20, 30]
    updates = [(vector_to_bytes(vectors[i]), sample_counts[i]) for i in range(3)]

    global_bytes = strategy.aggregate(
        global_model=None,
        client_updates=updates,
        round_id=1,
    )

    expected_p = build_directed_mixing_matrix(
        num_clients=3,
        round_id=1,
        out_degree=1,
        seed=42,
        time_varying=True,
    )
    expected_mu = expected_p @ torch.ones(3, dtype=torch.float32)
    expected_z = (expected_p @ vectors) / expected_mu.unsqueeze(1)

    assert strategy.client_ids == ["client_0", "client_1", "client_2"]
    for i, cid in enumerate(strategy.client_ids):
        assert torch.allclose(bytes_to_vector(strategy.client_weights[cid]), expected_z[i], atol=1e-6)

    expected_global = (10.0 * vectors[0] + 20.0 * vectors[1] + 30.0 * vectors[2]) / 60.0
    assert torch.allclose(bytes_to_vector(global_bytes), expected_global, atol=1e-6)


def test_dfedpgp_server_strategy_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="out_degree"):
        DFedPGPServerStrategy(out_degree=-1)

    strategy = DFedPGPServerStrategy()
    with pytest.raises(ValueError, match="empty"):
        strategy.aggregate(global_model=None, client_updates=[], round_id=1)

    dummy = vector_to_bytes(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="client_ids"):
        strategy.aggregate(
            global_model=None,
            client_updates=[(dummy, 5), (dummy, 5)],
            round_id=1,
            client_ids=["client_0"],
        )


def test_dfedpgp_cohort_change_resets_push_sum_weights_and_uses_custom_prefixes() -> None:
    class _CustomHeadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(2, 2, bias=False)
            self.custom_head = nn.Linear(2, 1, bias=False)

    model = _CustomHeadNet()
    strategy = DFedPGPServerStrategy(out_degree=2, seed=42, head_prefixes=("custom_head",))
    partition = partition_model(model, head_prefixes=("custom_head",))
    assert partition.head_names == ("custom_head.weight",)

    cohort_a = ["c0", "c1", "c2", "c3"]
    updates_a = [
        (vector_to_bytes(torch.full((6,), float(i + 1), dtype=torch.float32)), 10)
        for i in range(4)
    ]
    strategy.aggregate(global_model=model, client_updates=updates_a, round_id=1, client_ids=cohort_a)
    assert strategy.push_sum_weights is not None
    assert not torch.allclose(strategy.push_sum_weights, torch.ones(4))

    cohort_b = ["c0", "c1", "c2"]
    updates_b = [
        (vector_to_bytes(torch.full((6,), float(i + 10), dtype=torch.float32)), 10)
        for i in range(3)
    ]
    strategy.aggregate(global_model=model, client_updates=updates_b, round_id=2, client_ids=cohort_b)
    expected_p = build_directed_mixing_matrix(num_clients=3, round_id=2, out_degree=2, seed=42)
    assert strategy.push_sum_weights is not None
    assert torch.allclose(strategy.push_sum_weights, expected_p @ torch.ones(3), atol=1e-6)
    for i, cid in enumerate(cohort_b):
        _, head_vec = partition.split_vector(bytes_to_vector(strategy.client_weights[cid]))
        assert torch.equal(head_vec, torch.full((2,), float(i + 10), dtype=torch.float32))
