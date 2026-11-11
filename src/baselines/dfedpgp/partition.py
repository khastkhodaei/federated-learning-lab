"""Model parameter partitioning between shared backbone and personal head."""

from dataclasses import dataclass

import torch
from torch import nn

DEFAULT_HEAD_PREFIXES: tuple[str, ...] = ("fc_out", "classifier", "head")


def _is_head_parameter(name: str, head_prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in head_prefixes)


def _classify_named_parameters(
    model: nn.Module,
    head_prefixes: tuple[str, ...],
) -> tuple[list[tuple[str, nn.Parameter, bool]], int, int]:
    entries: list[tuple[str, nn.Parameter, bool]] = []
    backbone_dim = 0
    head_dim = 0

    for name, param in model.named_parameters():
        is_head = _is_head_parameter(name, head_prefixes)
        entries.append((name, param, is_head))
        if is_head:
            head_dim += param.numel()
        else:
            backbone_dim += param.numel()

    if not entries or backbone_dim + head_dim == 0:
        raise ValueError("Model has no parameters to partition.")
    if head_dim == 0:
        raise ValueError(
            f"head_prefixes={head_prefixes!r} matched 0 parameters for personal_head."
        )
    if backbone_dim == 0:
        raise ValueError(
            f"head_prefixes={head_prefixes!r} matched all parameters, "
            "leaving 0 parameters for shared_backbone."
        )
    return entries, backbone_dim, head_dim


def _validate_vector(vector: torch.Tensor, expected_dim: int, label: str) -> None:
    if vector.ndim != 1 or vector.numel() != expected_dim:
        raise ValueError(
            f"{label} must be a 1D tensor of length {expected_dim}, "
            f"got shape {tuple(vector.shape)}."
        )


@dataclass(frozen=True)
class ModelPartition:
    """Partition descriptor separating a model into shared_backbone and personal_head.

    Attributes:
        backbone_dim: Total scalar elements belonging to shared_backbone.
        head_dim: Total scalar elements belonging to personal_head.
        total_dim: Total scalar elements across the full parameter vector.
        backbone_names: Ordered parameter names assigned to shared_backbone.
        head_names: Ordered parameter names assigned to personal_head.
        head_mask: 1D CPU boolean tensor of length total_dim marking personal_head entries.
    """

    backbone_dim: int
    head_dim: int
    total_dim: int
    backbone_names: tuple[str, ...]
    head_names: tuple[str, ...]
    head_mask: torch.Tensor

    def split_vector(self, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits a flat 1D model parameter vector into (shared_backbone, personal_head).

        Args:
            vector: 1D tensor of length total_dim.

        Returns:
            Tuple of 1D tensors ``(shared_backbone_vec, personal_head_vec)``.

        Raises:
            ValueError: If ``vector`` is not 1D or does not have length ``total_dim``.
        """
        _validate_vector(vector, self.total_dim, "vector")
        mask = self.head_mask.to(device=vector.device)
        return vector[~mask], vector[mask]

    def merge_vectors(self, backbone_vec: torch.Tensor, head_vec: torch.Tensor) -> torch.Tensor:
        """Reconstructs a flat 1D parameter vector in original parameter order.

        Args:
            backbone_vec: 1D tensor of length ``backbone_dim``.
            head_vec: 1D tensor of length ``head_dim``.

        Returns:
            Reconstructed 1D tensor of length ``total_dim``.

        Raises:
            ValueError: If either vector is not 1D or does not match its expected length.
        """
        _validate_vector(backbone_vec, self.backbone_dim, "backbone_vec")
        _validate_vector(head_vec, self.head_dim, "head_vec")
        dtype = torch.promote_types(backbone_vec.dtype, head_vec.dtype)
        mask = self.head_mask.to(device=backbone_vec.device)
        merged = torch.empty(self.total_dim, dtype=dtype, device=backbone_vec.device)
        merged[~mask] = backbone_vec.to(dtype=dtype)
        merged[mask] = head_vec.to(dtype=dtype)
        return merged


def partition_model(
    model: nn.Module,
    head_prefixes: tuple[str, ...] = DEFAULT_HEAD_PREFIXES,
) -> ModelPartition:
    """Partitions a model's parameter vector into shared_backbone and personal_head.

    Args:
        model: Neural network module whose named parameters will be partitioned.
        head_prefixes: Module/parameter prefixes assigned to ``personal_head``.

    Returns:
        Frozen ``ModelPartition`` descriptor with mask and dimension metadata.

    Raises:
        ValueError: If the model has zero parameters, or either partition is empty.
    """
    entries, backbone_dim, head_dim = _classify_named_parameters(model, head_prefixes)
    total_dim = backbone_dim + head_dim
    head_mask = torch.zeros(total_dim, dtype=torch.bool)

    backbone_names: list[str] = []
    head_names: list[str] = []
    offset = 0

    for name, param, is_head in entries:
        numel = param.numel()
        if is_head:
            head_mask[offset : offset + numel] = True
            head_names.append(name)
        else:
            backbone_names.append(name)
        offset += numel

    return ModelPartition(
        backbone_dim=backbone_dim,
        head_dim=head_dim,
        total_dim=total_dim,
        backbone_names=tuple(backbone_names),
        head_names=tuple(head_names),
        head_mask=head_mask,
    )


def split_named_parameters(
    model: nn.Module,
    head_prefixes: tuple[str, ...] = DEFAULT_HEAD_PREFIXES,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Splits a model's live parameters into shared_backbone and personal_head groups.

    Args:
        model: Neural network module to inspect.
        head_prefixes: Module/parameter prefixes assigned to ``personal_head``.

    Returns:
        Tuple ``(backbone_params, head_params)`` referencing ``model`` parameters.

    Raises:
        ValueError: If the model has zero parameters, or either partition is empty.
    """
    entries, _, _ = _classify_named_parameters(model, head_prefixes)
    backbone_params = [param for _, param, is_head in entries if not is_head]
    head_params = [param for _, param, is_head in entries if is_head]
    return backbone_params, head_params
