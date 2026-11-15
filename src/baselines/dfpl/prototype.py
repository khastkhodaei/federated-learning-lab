import io
from typing import Any, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_local_prototypes(
    model: nn.Module,
    dataloader: Iterable[Any],
    device: Union[torch.device, str] = "cpu",
    normalize: bool = False,
) -> Tuple[dict[int, torch.Tensor], dict[int, int]]:
    """Compute mean feature representation embeddings per class from a local dataloader.

    Args:
        model: Model containing feature extraction capability.
        dataloader: Iterable providing batches of (inputs, targets).
        device: Target device for running model inference.
        normalize: If True, L2-normalizes each computed prototype vector.

    Returns:
        Tuple of (prototypes, sample_counts) where prototypes maps class index to
        a 1D feature tensor on CPU, and sample_counts maps class index to sample count.
    """
    was_training = model.training
    model.eval()

    class_sums: dict[int, torch.Tensor] = {}
    class_counts: dict[int, int] = {}

    try:
        with torch.no_grad():
            for batch in dataloader:
                if not batch or len(batch) < 2:
                    continue
                inputs, targets = batch[0], batch[1]
                if inputs.size(0) == 0:
                    continue

                inputs = inputs.to(device)
                if hasattr(model, "extract_features"):
                    features = model.extract_features(inputs)
                else:
                    out = model(inputs)
                    features = out[1] if isinstance(out, tuple) else out

                features = features.detach().cpu().to(torch.float32)
                targets = targets.detach().cpu()

                for c in torch.unique(targets):
                    c_idx = int(c.item())
                    mask = targets == c
                    sum_feat = features[mask].sum(dim=0)
                    count = int(mask.sum().item())

                    if c_idx not in class_sums:
                        class_sums[c_idx] = sum_feat
                        class_counts[c_idx] = count
                    else:
                        class_sums[c_idx] += sum_feat
                        class_counts[c_idx] += count
    finally:
        model.train(was_training)

    prototypes: dict[int, torch.Tensor] = {}
    for c_idx, feat_sum in class_sums.items():
        count = class_counts[c_idx]
        proto = feat_sum / float(max(count, 1))
        if normalize:
            norm = torch.norm(proto, p=2)
            if norm > 1e-12:
                proto = proto / norm
        prototypes[c_idx] = proto

    return prototypes, class_counts


def aggregate_prototypes(
    client_prototypes: list[dict[int, torch.Tensor]],
    client_sample_counts: Optional[list[dict[int, int]]] = None,
    normalize: bool = False,
) -> dict[int, torch.Tensor]:
    """Aggregate prototypes across multiple clients with optional sample-count weighting.

    Handles non-IID label skew robustly when clients hold disjoint or partial subsets of classes.

    Args:
        client_prototypes: List of client prototype dictionaries.
        client_sample_counts: Optional list of class-wise sample count dictionaries.
        normalize: If True, L2-normalizes each aggregated prototype.

    Returns:
        Dictionary mapping class index to aggregated prototype tensor on CPU.
    """
    if not client_prototypes:
        return {}

    all_classes = sorted({c for p_dict in client_prototypes for c in p_dict})
    aggregated: dict[int, torch.Tensor] = {}

    for c in all_classes:
        matching_clients: list[int] = [
            i for i, p_dict in enumerate(client_prototypes) if c in p_dict
        ]
        if not matching_clients:
            continue

        if client_sample_counts is not None and len(client_sample_counts) == len(client_prototypes):
            weights = [
                float(client_sample_counts[i].get(c, 0)) for i in matching_clients
            ]
            total_weight = sum(weights)
            if total_weight > 0:
                proto_c = sum(
                    w * client_prototypes[i][c].detach().cpu().to(torch.float32)
                    for w, i in zip(weights, matching_clients)
                ) / total_weight
            else:
                proto_c = sum(
                    client_prototypes[i][c].detach().cpu().to(torch.float32)
                    for i in matching_clients
                ) / float(len(matching_clients))
        else:
            proto_c = sum(
                client_prototypes[i][c].detach().cpu().to(torch.float32)
                for i in matching_clients
            ) / float(len(matching_clients))

        if normalize:
            norm = torch.norm(proto_c, p=2)
            if norm > 1e-12:
                proto_c = proto_c / norm

        aggregated[c] = proto_c

    return aggregated


def prototypes_to_tensor(
    prototypes: dict[int, torch.Tensor],
    num_classes: int,
    feature_dim: int,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse dictionary of prototypes into a dense matrix and presence mask.

    Args:
        prototypes: Dictionary mapping class index to 1D feature tensor.
        num_classes: Total number of classes.
        feature_dim: Dimensionality of feature embeddings.
        device: Target device for the output tensors.

    Returns:
        Tuple of (proto_matrix, mask) where proto_matrix has shape (num_classes, feature_dim)
        and mask has shape (num_classes,) indicating valid present prototypes.
    """
    proto_matrix = torch.zeros(
        (num_classes, feature_dim), dtype=torch.float32, device=device
    )
    mask = torch.zeros((num_classes,), dtype=torch.bool, device=device)

    for c, proto in prototypes.items():
        if 0 <= c < num_classes:
            proto_matrix[c] = proto.detach().to(device=device, dtype=torch.float32)
            mask[c] = True

    return proto_matrix, mask


def prototypes_to_bytes(prototypes: dict[int, torch.Tensor]) -> bytes:
    """Serialize dictionary of prototypes to raw bytes for network transport.

    Args:
        prototypes: Dictionary mapping class index to feature tensor.

    Returns:
        Serialized bytes representation, or b"" if empty.
    """
    if not prototypes:
        return b""

    cleaned = {
        int(k): v.detach().cpu().to(torch.float32)
        for k, v in prototypes.items()
    }
    buffer = io.BytesIO()
    torch.save(cleaned, buffer)
    return buffer.getvalue()


def bytes_to_prototypes(
    data: bytes,
    device: Union[torch.device, str] = "cpu",
) -> dict[int, torch.Tensor]:
    """Deserialize raw bytes back into a dictionary of class prototypes.

    Args:
        data: Serialized bytes.
        device: Target device for deserialized prototype tensors.

    Returns:
        Dictionary mapping class index to prototype tensor.
    """
    if not data:
        return {}

    buffer = io.BytesIO(data)
    loaded = torch.load(buffer, map_location=device, weights_only=True)
    return {
        int(k): v.to(device=device, dtype=torch.float32)
        for k, v in loaded.items()
    }
