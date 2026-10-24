from typing import List, Sequence, Tuple
import torch

from src.utils.serialization import bytes_to_vector, vector_to_bytes


def fedavg_vectors(updates: Sequence[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    """Computes the sample-weighted average of 1D parameter vectors (FedAvg).

    Args:
        updates: Sequence of tuples (weight_vector, num_samples).

    Returns:
        Aggregated 1D parameter vector.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates.")

    total_samples = sum(num_samples for _, num_samples in updates)

    if total_samples == 0:
        # Uniform average fallback if no client reported samples
        stacked = torch.stack([vec for vec, _ in updates])
        return torch.mean(stacked, dim=0)

    # Weighted sum: sum_k (n_k / total_n) * w_k
    aggregated = torch.zeros_like(updates[0][0])
    for vec, num_samples in updates:
        weight = num_samples / total_samples
        aggregated += weight * vec

    return aggregated


def fedavg_bytes(updates: Sequence[Tuple[bytes, int]]) -> bytes:
    """Computes the sample-weighted average of serialized parameter byte payloads.

    Args:
        updates: Sequence of tuples (serialized_weights, num_samples).

    Returns:
        Serialized aggregated parameter vector bytes.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates.")

    vector_updates = [
        (bytes_to_vector(raw_bytes), num_samples) for raw_bytes, num_samples in updates
    ]
    aggregated_vector = fedavg_vectors(vector_updates)
    return vector_to_bytes(aggregated_vector)
