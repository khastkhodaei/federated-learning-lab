"""Deterministic directed asymmetric topology generation for DFedPGP."""

import random

import torch


def build_directed_mixing_matrix(
    num_clients: int,
    round_id: int = 1,
    out_degree: int = 1,
    seed: int = 42,
    time_varying: bool = True,
) -> torch.Tensor:
    """Builds a column-stochastic directed mixing matrix on CPU.

    Entry ``matrix[i, j]`` is the push weight sent from peer ``j`` to peer ``i``.
    When ``num_clients >= 2`` and ``out_degree >= 1``, a directed Hamiltonian cycle
    over a seeded node permutation guarantees strong connectivity and asymmetry for
    ``num_clients >= 3``, and any extra peer out-neighbors are sampled without
    replacement using an isolated RNG.

    Args:
        num_clients: Number of participating clients (must be ``>= 1``).
        round_id: Communication round index used when ``time_varying=True`` (``>= 0``).
        out_degree: Number of non-self outgoing peers per client (``>= 0``).
        seed: Deterministic base seed for the local RNG.
        time_varying: Whether ``round_id`` varies the topology across rounds.

    Returns:
        CPU ``torch.float32`` tensor of shape ``(num_clients, num_clients)`` whose
        columns sum to ``1.0`` with strictly positive diagonal self-loops.

    Raises:
        ValueError: If ``num_clients < 1``, ``round_id < 0``, or ``out_degree < 0``.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}.")
    if round_id < 0:
        raise ValueError(f"round_id must be >= 0, got {round_id}.")
    if out_degree < 0:
        raise ValueError(f"out_degree must be >= 0, got {out_degree}.")

    if num_clients == 1:
        return torch.ones((1, 1), dtype=torch.float32)
    if out_degree == 0:
        return torch.eye(num_clients, dtype=torch.float32)

    effective_deg = min(out_degree, num_clients - 1)
    rng_seed: int | str = f"{seed}:{round_id}" if time_varying else seed
    rng = random.Random(rng_seed)
    nodes = list(range(num_clients))
    rng.shuffle(nodes)

    matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)
    weight = 1.0 / (1 + effective_deg)

    for idx, src in enumerate(nodes):
        cycle_target = nodes[(idx + 1) % num_clients]
        targets = [src, cycle_target]
        if effective_deg > 1:
            candidates = [node for node in range(num_clients) if node not in (src, cycle_target)]
            targets.extend(rng.sample(candidates, effective_deg - 1))
        matrix[targets, src] = weight

    return matrix
