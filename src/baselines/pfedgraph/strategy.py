"""pFedGraph Server Strategy for Personalized Federated Learning via Graph.

Maintains an adaptive client collaboration graph (adjacency matrix W) optimized based on
pairwise parameter distances and regularization. Aggregates personalized models for each client.
Reference: Chen et al., 'pFedGraph: Personalized Federated Learning via Graph'.
"""

import copy
import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from src.baselines.base import ServerStrategy
from src.utils.serialization import bytes_to_vector, load_bytes_into_model, model_to_bytes, vector_to_bytes

logger = logging.getLogger("pfedgraph.strategy")


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Projects a 1D vector v onto the probability simplex:

    min_{x} ||x - v||_2^2 subject to sum(x) = 1 and x >= 0.
    Exact O(K log K) Euclidean projection algorithm (Duchi et al., 2008).
    """
    k = v.shape[0]
    if k == 1:
        return torch.tensor([1.0], dtype=v.dtype, device=v.device)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, k + 1, dtype=v.dtype, device=v.device)
    cond = u - cssv / ind > 0
    valid_indices = torch.nonzero(cond)
    if len(valid_indices) == 0:
        return torch.full_like(v, 1.0 / k)
    rho = valid_indices[-1].item() + 1
    theta = cssv[rho - 1] / rho
    w = torch.clamp(v - theta, min=0.0)
    # Ensure exact sum=1 due to floating point rounding
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = torch.full_like(v, 1.0 / k)
    return w


class PFedGraphServerStrategy(ServerStrategy):
    """Server-side aggregation strategy for pFedGraph.

    Learns an adjacency matrix W (K x K) representing pairwise client similarity
    and aggregates a personalized model for each client based on graph edge weights.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        lam: float = 0.01,
        step_mode: str = "projected_gd",
    ) -> None:
        """Initializes pFedGraph strategy parameters.

        Args:
            alpha: Graph learning rate / interpolation factor (default: 0.1).
            lam: Sparsity / L2 regularization penalty parameter lambda (default: 0.01).
            step_mode: Optimization mode ('projected_gd' or 'target_interp').
        """
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.step_mode = step_mode
        self.adjacency_matrix: torch.Tensor | None = None
        self.client_ids: list[str] = []
        self.client_weights: dict[str, bytes] = {}

    def update_graph(self, client_vectors: torch.Tensor) -> torch.Tensor:
        """Updates the adjacency matrix W given client parameter vectors.

        Args:
            client_vectors: Tensor of shape (K, P) where K is client count and P is parameter count.

        Returns:
            Updated adjacency matrix W of shape (K, K) where each row lies on the probability simplex.
        """
        k = client_vectors.shape[0]
        if k <= 1:
            self.adjacency_matrix = torch.ones((1, 1), dtype=torch.float32, device=client_vectors.device)
            return self.adjacency_matrix

        # Initialize uniform graph if not present or resized
        if self.adjacency_matrix is None or self.adjacency_matrix.shape != (k, k):
            self.adjacency_matrix = torch.full(
                (k, k), 1.0 / k, dtype=torch.float32, device=client_vectors.device
            )

        # 1. Compute pairwise squared L2 distance matrix D (K x K)
        # D_ij = ||w_i - w_j||^2
        diff = client_vectors.unsqueeze(1) - client_vectors.unsqueeze(0)  # (K, K, P)
        dist_sq = torch.sum(diff**2, dim=-1)  # (K, K)

        # Scale distances to make optimization robust to parameter dimension P
        scale = dist_sq.mean().item()
        if scale > 1e-12:
            norm_dist = dist_sq / scale
        else:
            norm_dist = dist_sq

        # 2. Update each row W_i
        # Objective: min_{W_i in Simplex} W_i^T D_i + lambda ||W_i||^2
        new_w = torch.zeros_like(self.adjacency_matrix)
        for i in range(k):
            d_i = norm_dist[i]
            w_curr = self.adjacency_matrix[i]

            if self.step_mode == "projected_gd":
                # Gradient of objective w.r.t W_i: grad = d_i + 2 * lam * w_curr
                grad = d_i + 2.0 * self.lam * w_curr
                unprojected = w_curr - self.alpha * grad
                new_w[i] = project_to_simplex(unprojected)
            else:
                # Target unconstrained minimizer is -d_i / (2 * lam)
                target = project_to_simplex(-d_i / (2.0 * max(1e-6, self.lam)))
                new_w[i] = project_to_simplex((1.0 - self.alpha) * w_curr + self.alpha * target)

        self.adjacency_matrix = new_w
        return self.adjacency_matrix

    def aggregate(
        self,
        global_model: nn.Module | None,
        client_updates: Sequence[tuple[bytes, int]],
        round_id: int,
        client_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Computes graph-based personalized models for each client and updates global model.

        Args:
            global_model: Global model instance (updated with weighted mean).
            client_updates: Sequence of tuples (weights_bytes, num_samples).
            round_id: Current federated round number.
            client_ids: Optional list of client node IDs corresponding to client_updates.

        Returns:
            Serialized global model parameter bytes.
        """
        if not client_updates:
            logger.warning("pFedGraph received empty client updates in round %d.", round_id)
            return model_to_bytes(global_model) if global_model is not None else b""

        k = len(client_updates)
        if client_ids is not None and len(client_ids) == k:
            self.client_ids = [str(cid) for cid in client_ids]
        else:
            self.client_ids = [f"client_{i}" for i in range(k)]

        # 1. Unpack client updates to flattened parameter vectors
        vectors: list[torch.Tensor] = []
        sample_counts: list[int] = []
        for weights_bytes, num_samples in client_updates:
            vec = bytes_to_vector(weights_bytes)
            vectors.append(vec)
            sample_counts.append(max(1, num_samples))

        stacked_vectors = torch.stack(vectors, dim=0)  # (K, P)

        # 2. Update graph adjacency matrix
        w_matrix = self.update_graph(stacked_vectors)

        # 3. Aggregate personalized models for each client: w_i_agg = sum_j W_ij * w_j
        self.client_weights = {}
        for i in range(k):
            # Convex combination of client vectors using row i of W
            row_weights = w_matrix[i].unsqueeze(1)  # (K, 1)
            pers_vector = torch.sum(row_weights * stacked_vectors, dim=0)  # (P,)
            pers_bytes = vector_to_bytes(pers_vector)
            self.client_weights[self.client_ids[i]] = pers_bytes

        # 4. Compute global benchmark model via standard sample-weighted FedAvg
        total_samples = sum(sample_counts)
        global_vec = torch.zeros_like(vectors[0])
        for vec, n in zip(vectors, sample_counts):
            global_vec.add_(vec, alpha=n / total_samples)

        global_bytes = vector_to_bytes(global_vec)
        if global_model is not None:
            load_bytes_into_model(global_bytes, global_model)

        logger.info(
            "pFedGraph Round %d: Graph W updated (%d clients). Adjacency:\n%s",
            round_id,
            k,
            w_matrix.cpu().numpy().round(4),
        )

        return global_bytes
