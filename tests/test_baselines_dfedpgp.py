import random
from collections import deque
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import create_server_strategy, get_baseline
from src.baselines.dfedpgp import (
    DEFAULT_HEAD_PREFIXES,
    DFedPGPClientTrainer,
    DFedPGPServerStrategy,
    DFedPGPTrainer,
    ModelPartition,
    build_directed_mixing_matrix,
    partition_model,
    split_named_parameters,
)
from src.client.config import (
    ClientConfig,
    DFedPGPClientConfig,
    create_trainer,
    create_trainer_from_config,
    load_client_config,
)
from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.server.dispatch import dispatch_round
from src.server.server import FederatedServer
from src.transport.client import FederatedClient
from src.transport.federated_pb2 import TrainRequest
from src.utils.config import DFedPGPConfig, UnifiedConfig, load_unified_config
from src.utils.serialization import (
    bytes_to_vector,
    model_to_bytes,
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


class _TinyClassifierNet(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.backbone = nn.Linear(4, 3, bias=True)
        self.head = nn.Linear(3, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))


def _make_tiny_loader(seed: int = 0, num_samples: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(num_samples, 4, generator=generator)
    targets = torch.randint(0, 2, (num_samples,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)


def test_dfedpgp_trainer_alternating_training_and_private_head_persistence() -> None:
    assert DFedPGPTrainer is DFedPGPClientTrainer

    model = _TinyClassifierNet(seed=1)
    partition = partition_model(model)
    loader = _make_tiny_loader(seed=1, num_samples=8)
    criterion = nn.CrossEntropyLoss()
    trainer = DFedPGPClientTrainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    initial_vector = model_to_vector(model)
    updated_bytes, total_samples, metrics = trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    updated_vec = bytes_to_vector(updated_bytes)
    assert not torch.equal(updated_vec, initial_vector)
    assert total_samples == 8
    assert {"loss", "head_loss", "backbone_loss"}.issubset(metrics.keys())
    assert trainer.private_head_vector is not None
    assert trainer.personalized_model is not None

    round1_head = trainer.private_head_vector.clone()
    _, model_head_r1 = partition.split_vector(model_to_vector(model))
    assert torch.equal(round1_head, model_head_r1)

    # Simulate round 2 where Node overwrote model with incoming payload containing a poisoned head.
    incoming_backbone = torch.full((partition.backbone_dim,), 0.25, dtype=torch.float32)
    poisoned_head = torch.full((partition.head_dim,), 999.0, dtype=torch.float32)
    incoming_payload = partition.merge_vectors(incoming_backbone, poisoned_head)
    vector_to_model(incoming_payload, model)
    incoming_bytes = vector_to_bytes(incoming_payload)

    updated_r2_bytes, samples_r2, _ = trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        device=torch.device("cpu"),
        global_weights=incoming_bytes,
        head_epochs=0,
        backbone_epochs=1,
    )

    assert samples_r2 == 8
    r2_backbone, r2_head = partition.split_vector(bytes_to_vector(updated_r2_bytes))
    assert torch.equal(r2_head, round1_head)
    assert torch.equal(trainer.private_head_vector, round1_head)
    assert not torch.equal(r2_backbone, incoming_backbone)
    assert torch.all((r2_backbone - incoming_backbone).abs() < 0.5)


def test_dfedpgp_trainer_phase_isolation_and_requires_grad_restoration() -> None:
    criterion = nn.CrossEntropyLoss()
    loader = _make_tiny_loader(seed=3, num_samples=8)

    # Head-only phase: shared_backbone must stay bit-exact, personal_head updates.
    model_head_only = _TinyClassifierNet(seed=5)
    part_head = partition_model(model_head_only)
    init_backbone_h, init_head_h = part_head.split_vector(model_to_vector(model_head_only))
    trainer_head_only = DFedPGPClientTrainer(head_epochs=1, backbone_epochs=0)

    out_h_bytes, samples_h, metrics_h = trainer_head_only.train(
        model=model_head_only,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=torch.optim.SGD(model_head_only.parameters(), lr=0.05),
        device=torch.device("cpu"),
    )
    out_backbone_h, out_head_h = part_head.split_vector(bytes_to_vector(out_h_bytes))
    assert samples_h == 0
    assert metrics_h["loss"] == metrics_h["head_loss"]
    assert metrics_h["backbone_loss"] == 0.0
    assert torch.equal(out_backbone_h, init_backbone_h)
    assert not torch.equal(out_head_h, init_head_h)
    assert all(param.requires_grad for param in model_head_only.parameters())

    # Backbone-only phase: personal_head must stay bit-exact, shared_backbone updates.
    model_backbone_only = _TinyClassifierNet(seed=5)
    part_backbone = partition_model(model_backbone_only)
    init_backbone_b, init_head_b = part_backbone.split_vector(model_to_vector(model_backbone_only))
    trainer_backbone_only = DFedPGPClientTrainer(head_epochs=0, backbone_epochs=1)

    out_b_bytes, samples_b, metrics_b = trainer_backbone_only.train(
        model=model_backbone_only,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=torch.optim.SGD(model_backbone_only.parameters(), lr=0.05),
        device=torch.device("cpu"),
    )
    out_backbone_b, out_head_b = part_backbone.split_vector(bytes_to_vector(out_b_bytes))
    assert samples_b == 8
    assert metrics_b["loss"] == metrics_b["backbone_loss"]
    assert metrics_b["head_loss"] == 0.0
    assert not torch.equal(out_backbone_b, init_backbone_b)
    assert torch.equal(out_head_b, init_head_b)
    assert all(param.requires_grad for param in model_backbone_only.parameters())

    # Exception safety & initially frozen parameter restoration + lr_head override precedence.
    model_exc = _TinyClassifierNet(seed=7)
    model_exc.backbone.bias.requires_grad = False
    trainer_exc = DFedPGPClientTrainer()

    class _FailingCriterion(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        trainer_exc.train(
            model=model_exc,
            dataloader=loader,
            epochs=1,
            criterion=_FailingCriterion(),
            optimizer=torch.optim.SGD(model_exc.parameters(), lr=0.0),
            device=torch.device("cpu"),
        )
    assert model_exc.backbone.bias.requires_grad is False
    assert model_exc.backbone.weight.requires_grad is True
    assert all(param.requires_grad for param in model_exc.head.parameters())

    model_lr_override = _TinyClassifierNet(seed=8)
    part_lr = partition_model(model_lr_override)
    init_b_lr, init_h_lr = part_lr.split_vector(model_to_vector(model_lr_override))
    out_lr_bytes, _, _ = DFedPGPClientTrainer().train(
        model=model_lr_override,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=torch.optim.SGD(model_lr_override.parameters(), lr=0.0),
        device=torch.device("cpu"),
        lr_head=0.1,
        lr_backbone=0.2,
    )
    out_b_lr, out_h_lr = part_lr.split_vector(bytes_to_vector(out_lr_bytes))
    assert not torch.equal(out_b_lr, init_b_lr)
    assert not torch.equal(out_h_lr, init_h_lr)


def test_dfedpgp_trainer_node_on_train_round_and_evaluate_local_seam() -> None:
    train_loader = _make_tiny_loader(seed=11, num_samples=12)
    test_loader = _make_tiny_loader(seed=12, num_samples=12)
    model = _TinyClassifierNet(seed=11)
    initial_bytes = model_to_bytes(model)

    trainer = DFedPGPClientTrainer()
    node = Node(
        node_id="dfedpgp_client_0",
        data=train_loader,
        test_data=test_loader,
        model=model,
        lr=0.05,
        trainer=trainer,
    )

    response = node.on_train_round(
        TrainRequest(round_id=1, weights=initial_bytes, epochs=1)
    )

    assert response.success is True
    assert response.num_samples == 12
    assert "test_acc=" in response.message

    eval_before = node.evaluate_local()
    with torch.no_grad():
        for param in node.model.parameters():
            param.zero_()
    eval_after = node.evaluate_local()

    assert eval_after["accuracy"] == eval_before["accuracy"]
    assert eval_after["loss"] == eval_before["loss"]


def test_dfedpgp_trainer_determinism_across_rounds() -> None:
    criterion = nn.CrossEntropyLoss()
    loader_a = _make_tiny_loader(seed=21, num_samples=8)
    loader_b = _make_tiny_loader(seed=21, num_samples=8)

    model_a = _TinyClassifierNet(seed=21)
    model_b = _TinyClassifierNet(seed=21)
    trainer_a = DFedPGPClientTrainer(lr_backbone=0.04, lr_head=0.08)
    trainer_b = DFedPGPClientTrainer(lr_backbone=0.04, lr_head=0.08)

    for round_id in (1, 2):
        payload_a = model_to_bytes(model_a)
        payload_b = model_to_bytes(model_b)
        out_a, samples_a, metrics_a = trainer_a.train(
            model=model_a,
            dataloader=loader_a,
            epochs=1,
            criterion=criterion,
            optimizer=torch.optim.SGD(model_a.parameters(), lr=0.01, momentum=0.9),
            device=torch.device("cpu"),
            global_weights=payload_a if round_id > 1 else None,
        )
        out_b, samples_b, metrics_b = trainer_b.train(
            model=model_b,
            dataloader=loader_b,
            epochs=1,
            criterion=criterion,
            optimizer=torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9),
            device=torch.device("cpu"),
            global_weights=payload_b if round_id > 1 else None,
        )
        assert out_a == out_b
        assert samples_a == samples_b
        assert metrics_a == metrics_b
        assert trainer_a.private_head_vector is not None
        assert trainer_b.private_head_vector is not None
        assert torch.equal(trainer_a.private_head_vector, trainer_b.private_head_vector)


@pytest.mark.parametrize(
    ("lr_backbone", "lr_head", "head_epochs", "backbone_epochs"),
    [
        (0.0, None, None, None),
        (-0.1, None, None, None),
        (None, 0.0, None, None),
        (None, -0.05, None, None),
        (None, None, -1, None),
        (None, None, None, -1),
    ],
)
def test_dfedpgp_trainer_init_rejects_invalid_hyperparams(
    lr_backbone: float | None,
    lr_head: float | None,
    head_epochs: int | None,
    backbone_epochs: int | None,
) -> None:
    with pytest.raises(ValueError):
        DFedPGPClientTrainer(
            lr_backbone=lr_backbone,
            lr_head=lr_head,
            head_epochs=head_epochs,
            backbone_epochs=backbone_epochs,
        )


def test_dfedpgp_trainer_train_rejects_invalid_overrides_and_head_dim_mismatch() -> None:
    model = _TinyClassifierNet(seed=1)
    loader = _make_tiny_loader(seed=1, num_samples=8)
    criterion = nn.CrossEntropyLoss()
    trainer = DFedPGPClientTrainer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for bad_kwargs in (
        {"lr_backbone": 0.0},
        {"lr_head": -0.1},
        {"head_epochs": -1},
        {"backbone_epochs": -2},
    ):
        with pytest.raises(ValueError):
            trainer.train(
                model=model,
                dataloader=loader,
                epochs=1,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device("cpu"),
                **bad_kwargs,
            )

    trainer.train(
        model=model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    class _WiderHeadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 3, bias=True)
            self.head = nn.Linear(3, 5, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(torch.relu(self.backbone(x)))

    mismatched_model = _WiderHeadNet()
    with pytest.raises(ValueError, match="head_dim"):
        trainer.train(
            model=mismatched_model,
            dataloader=loader,
            epochs=1,
            criterion=criterion,
            optimizer=torch.optim.SGD(mismatched_model.parameters(), lr=0.05),
            device=torch.device("cpu"),
        )


def test_dfedpgp_baseline_registry_and_server_strategy_factory() -> None:
    import src.baselines as baselines_pkg

    strategy_cls, trainer_cls = get_baseline("dfedpgp")
    assert strategy_cls is DFedPGPServerStrategy
    assert trainer_cls is DFedPGPClientTrainer
    assert baselines_pkg.DFedPGPServerStrategy is DFedPGPServerStrategy
    assert baselines_pkg.DFedPGPClientTrainer is DFedPGPClientTrainer
    assert baselines_pkg.DFedPGPTrainer is DFedPGPTrainer

    default_strategy = create_server_strategy("dfedpgp")
    assert isinstance(default_strategy, DFedPGPServerStrategy)
    assert default_strategy.out_degree == 1
    assert default_strategy.seed == 42
    assert default_strategy.time_varying is True

    cfg = UnifiedConfig(seed=77)
    cfg.baselines.dfedpgp = DFedPGPConfig(out_degree=2, time_varying=False)

    cfg_strategy = create_server_strategy(
        "dfedpgp",
        config=cfg,
        out_degree=None,
        seed=None,
        time_varying=None,
    )
    assert isinstance(cfg_strategy, DFedPGPServerStrategy)
    assert cfg_strategy.out_degree == 2
    assert cfg_strategy.seed == 77
    assert cfg_strategy.time_varying is False

    override_strategy = create_server_strategy(
        "dfedpgp",
        config=cfg,
        out_degree=3,
        seed=123,
        time_varying=True,
    )
    assert isinstance(override_strategy, DFedPGPServerStrategy)
    assert override_strategy.out_degree == 3
    assert override_strategy.seed == 123
    assert override_strategy.time_varying is True


def test_dfedpgp_unified_config_yaml_loading_and_to_client_config(tmp_path: Path) -> None:
    from src.config import DFedPGPConfig as ConfigPkgDFedPGPConfig
    from src.config.unified import DFedPGPConfig as UnifiedModuleDFedPGPConfig

    assert ConfigPkgDFedPGPConfig is DFedPGPConfig
    assert UnifiedModuleDFedPGPConfig is DFedPGPConfig

    repo_config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(repo_config_path, "r", encoding="utf-8") as f:
        raw_repo_yaml = yaml.safe_load(f)
    assert raw_repo_yaml["baselines"]["dfedpgp"] == {"out_degree": 1, "time_varying": True}

    repo_cfg = load_unified_config(repo_config_path)
    assert repo_cfg.baselines.dfedpgp == DFedPGPConfig(
        out_degree=1,
        time_varying=True,
        lr_backbone=None,
        lr_head=None,
        head_epochs=None,
        backbone_epochs=None,
    )

    custom_yaml = tmp_path / "dfedpgp_custom.yaml"
    custom_yaml.write_text(
        yaml.safe_dump(
            {
                "algorithm": "dfedpgp",
                "seed": 19,
                "baselines": {
                    "dfedpgp": {
                        "out_degree": 2,
                        "time_varying": False,
                        "lr_backbone": 0.02,
                        "lr_head": 0.05,
                        "head_epochs": 3,
                        "backbone_epochs": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    custom_cfg = load_unified_config(custom_yaml)
    assert custom_cfg.algorithm == "dfedpgp"
    assert custom_cfg.baselines.dfedpgp == DFedPGPConfig(
        out_degree=2,
        time_varying=False,
        lr_backbone=0.02,
        lr_head=0.05,
        head_epochs=3,
        backbone_epochs=1,
    )

    client_cfg = custom_cfg.to_client_config(client_id=1)
    assert client_cfg.algorithm == "dfedpgp"
    assert client_cfg.dfedpgp.lr_backbone == 0.02
    assert client_cfg.dfedpgp.lr_head == 0.05
    assert client_cfg.dfedpgp.head_epochs == 3
    assert client_cfg.dfedpgp.backbone_epochs == 1

    null_yaml = tmp_path / "dfedpgp_nulls.yaml"
    null_yaml.write_text(
        yaml.safe_dump(
            {
                "algorithm": "dfedpgp",
                "baselines": {
                    "dfedpgp": {
                        "out_degree": 2,
                        "time_varying": False,
                        "lr_backbone": None,
                        "lr_head": None,
                        "head_epochs": None,
                        "backbone_epochs": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    null_cfg = load_unified_config(null_yaml)
    assert null_cfg.baselines.dfedpgp.lr_backbone is None
    assert null_cfg.baselines.dfedpgp.lr_head is None
    assert null_cfg.baselines.dfedpgp.head_epochs is None
    assert null_cfg.baselines.dfedpgp.backbone_epochs is None


def test_dfedpgp_client_config_yaml_loading_and_trainer_factory(tmp_path: Path) -> None:
    top_level_yaml = tmp_path / "client_top_level.yaml"
    top_level_yaml.write_text(
        yaml.safe_dump(
            {
                "algorithm": "dfedpgp",
                "dfedpgp": {
                    "lr_backbone": 0.03,
                    "lr_head": 0.06,
                    "head_epochs": 2,
                    "backbone_epochs": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    top_cfg = load_client_config(top_level_yaml)
    assert top_cfg.algorithm == "dfedpgp"
    assert top_cfg.dfedpgp == DFedPGPClientConfig(
        lr_backbone=0.03,
        lr_head=0.06,
        head_epochs=2,
        backbone_epochs=4,
    )

    nested_yaml = tmp_path / "client_nested.yaml"
    nested_yaml.write_text(
        yaml.safe_dump(
            {
                "algorithm": "dfedpgp",
                "baselines": {
                    "dfedpgp": {
                        "lr_backbone": 0.015,
                        "lr_head": None,
                        "head_epochs": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    nested_cfg = load_client_config(nested_yaml)
    assert nested_cfg.dfedpgp == DFedPGPClientConfig(
        lr_backbone=0.015,
        lr_head=None,
        head_epochs=3,
        backbone_epochs=None,
    )

    trainer_direct = create_trainer(
        "dfedpgp",
        lr_backbone=0.02,
        lr_head=0.04,
        head_epochs=2,
        backbone_epochs=3,
    )
    assert isinstance(trainer_direct, DFedPGPClientTrainer)
    assert trainer_direct.lr_backbone == 0.02
    assert trainer_direct.lr_head == 0.04
    assert trainer_direct.head_epochs == 2
    assert trainer_direct.backbone_epochs == 3

    trainer_from_cfg = create_trainer_from_config(top_cfg)
    assert isinstance(trainer_from_cfg, DFedPGPClientTrainer)
    assert trainer_from_cfg.lr_backbone == 0.03
    assert trainer_from_cfg.lr_head == 0.06
    assert trainer_from_cfg.head_epochs == 2
    assert trainer_from_cfg.backbone_epochs == 4

    default_client_cfg = ClientConfig(algorithm="dfedpgp")
    default_trainer = create_trainer_from_config(default_client_cfg)
    assert isinstance(default_trainer, DFedPGPClientTrainer)
    assert default_trainer.lr_backbone is None
    assert default_trainer.lr_head is None
    assert default_trainer.head_epochs is None
    assert default_trainer.backbone_epochs is None


def test_dfedpgp_end_to_end_synthetic_simulation(tmp_path: Path) -> None:
    from run_simulation import run_dfedpgp_simulation, run_federated_simulation

    assert callable(run_dfedpgp_simulation)
    reports_dir = tmp_path / "reports"
    results = run_federated_simulation(
        algorithm="dfedpgp",
        num_clients=3,
        rounds=2,
        epochs=1,
        out_degree=1,
        time_varying=True,
        lr_backbone=0.02,
        lr_head=0.05,
        head_epochs=1,
        backbone_epochs=1,
        synthetic=True,
        reports_dir=reports_dir,
    )

    assert results["algorithm"] == "dfedpgp"
    assert results["rounds_completed"] == 2
    assert results["num_clients"] == 3
    assert results["final_model_weights"] is not None
    assert len(results["final_model_weights"]) > 0

    dfed_reports = reports_dir / "dfedpgp"
    assert (dfed_reports / "metrics.json").is_file()
    assert (dfed_reports / "loss_curve.png").is_file()
    assert (dfed_reports / "accuracy_curves.png").is_file()
    assert (dfed_reports / "client_dispersion.png").is_file()
    assert (dfed_reports / "round_duration.png").is_file()


def test_dfedpgp_live_multi_round_grpc_server_and_nodes() -> None:
    initial_model = _TinyClassifierNet(seed=0)
    partition = partition_model(initial_model)
    init_backbone, _ = partition.split_vector(model_to_vector(initial_model))

    strategy = DFedPGPServerStrategy(out_degree=1, seed=42)
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=initial_model,
        strategy=strategy,
    )
    transports: list[FederatedClient] = []
    nodes: list[Node] = []

    try:
        server.start()
        server_addr = f"127.0.0.1:{server.bound_port}"

        datasets = [
            TensorDataset(torch.full((8, 4), 1.5), torch.zeros(8, dtype=torch.long)),
            TensorDataset(torch.full((8, 4), -1.5), torch.ones(8, dtype=torch.long)),
            TensorDataset(
                torch.linspace(-2.0, 2.0, 32).view(8, 4),
                torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long),
            ),
        ]
        trainers = [DFedPGPClientTrainer() for _ in range(3)]
        for i in range(3):
            transport = FederatedClient(target=server_addr)
            transports.append(transport)
            nodes.append(
                Node(
                    node_id=f"client_{i}",
                    data=DataLoader(datasets[i], batch_size=4, shuffle=False),
                    test_data=DataLoader(datasets[i], batch_size=4, shuffle=False),
                    model=_TinyClassifierNet(seed=0),
                    transport=transport,
                    lr=0.05,
                    port=0,
                    trainer=trainers[i],
                )
            )

        for node in nodes:
            reg = node.register()
            assert reg is not None and reg.success is True

        for round_id in (1, 2):
            responses = dispatch_round(server, round_id=round_id, epochs=1, timeout=10.0)
            assert len(responses) == 3
            assert all(res.success and res.num_samples == 8 for res in responses)

        expected_cids = {"client_0", "client_1", "client_2"}
        assert set(server.client_weights.keys()) == expected_cids
        assert strategy.push_sum_weights is not None
        assert torch.isclose(strategy.push_sum_weights.sum(), torch.tensor(3.0), atol=1e-5)

        resolved_heads = [
            trainer.private_head_vector
            for trainer in trainers
            if trainer.private_head_vector is not None
        ]
        assert len(resolved_heads) == 3
        assert not torch.equal(resolved_heads[0], resolved_heads[1])
        assert not torch.equal(resolved_heads[1], resolved_heads[2])
        assert not torch.equal(resolved_heads[0], resolved_heads[2])

        for i, node in enumerate(nodes):
            cid = f"client_{i}"
            pushed_backbone, pushed_head = partition.split_vector(
                bytes_to_vector(server.client_weights[cid])
            )
            assert not torch.equal(pushed_backbone, init_backbone)
            assert torch.equal(pushed_head, resolved_heads[i])

            eval_res = node.evaluate_local()
            assert eval_res["total"] == 8
            assert 0.0 <= eval_res["accuracy"] <= 1.0
    finally:
        for node, transport in zip(nodes, transports, strict=True):
            node.stop(grace=0)
            transport.close()
        server.stop(grace=0)
