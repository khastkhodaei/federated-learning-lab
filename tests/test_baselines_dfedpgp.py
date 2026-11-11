import pytest
import torch
from torch import nn

from src.baselines.dfedpgp import (
    DEFAULT_HEAD_PREFIXES,
    ModelPartition,
    partition_model,
    split_named_parameters,
)
from src.models.cnn import SimpleCNN
from src.utils.serialization import model_to_vector, vector_to_model


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

