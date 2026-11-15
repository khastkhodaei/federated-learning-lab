"""DFPL client local trainer with intermediate representation prototype alignment."""

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.baselines.dfpl.prototype import (
    bytes_to_prototypes,
    compute_local_prototypes,
    prototypes_to_bytes,
)
from src.utils.serialization import load_bytes_into_model, model_to_bytes


class DFPLClientTrainer(ClientTrainer):
    """Local client trainer for Decentralized Federated Prototype Learning (DFPL).

    Trains a local model by optimizing classification task loss along with an alignment
    regularizer pulling sample feature embeddings toward target class prototypes.
    """

    def __init__(
        self,
        lambda_proto: float = 0.1,
        normalize_prototypes: bool = False,
        loss_type: str = "mse",
    ) -> None:
        """Initializes DFPL client trainer hyperparameters and state.

        Args:
            lambda_proto: Regularization weight for prototype alignment loss.
            normalize_prototypes: If True, normalizes cached prototypes to unit L2 norm.
            loss_type: Distance metric for alignment ('mse' or 'cosine').

        Raises:
            ValueError: If lambda_proto is negative or loss_type is unsupported.
        """
        if lambda_proto < 0:
            raise ValueError(f"lambda_proto must be non-negative, got {lambda_proto}.")
        if loss_type not in ("mse", "cosine"):
            raise ValueError(
                f"loss_type must be 'mse' or 'cosine', got '{loss_type}'."
            )

        self.lambda_proto: float = float(lambda_proto)
        self.normalize_prototypes: bool = bool(normalize_prototypes)
        self.loss_type: str = str(loss_type)

        self.last_local_prototypes: dict[int, torch.Tensor] = {}
        self.last_sample_counts: dict[int, int] = {}
        self.last_prototypes_bytes: bytes = b""

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: Optional[bytes] = None,
        global_prototypes: Optional[Union[dict[int, torch.Tensor], bytes]] = None,
        lambda_proto: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes local client training with prototype alignment regularizer.

        Args:
            model: Client local neural network model.
            dataloader: DataLoader delivering local client data batches.
            epochs: Number of local training epochs.
            criterion: Classification loss function (e.g. CrossEntropyLoss).
            optimizer: Local optimizer.
            device: Computing device (CPU or GPU).
            global_weights: Optional serialized global model weights for synchronization.
            global_prototypes: Target class prototypes as dictionary or serialized bytes.
            lambda_proto: Optional per-round override for prototype alignment weight.
            **kwargs: Additional baseline-specific arguments.

        Returns:
            Tuple of (serialized_model_weights, total_samples, metrics_dict).
        """
        effective_lambda = float(
            lambda_proto if lambda_proto is not None else self.lambda_proto
        )
        if effective_lambda < 0:
            raise ValueError(
                f"lambda_proto must be non-negative, got {effective_lambda}."
            )

        if global_weights is not None and len(global_weights) > 0:
            load_bytes_into_model(global_weights, model)

        if epochs <= 0:
            return model_to_bytes(model), 0, {"loss": 0.0, "loss_ce": 0.0, "loss_proto": 0.0}

        target_prototypes: dict[int, torch.Tensor] = {}
        if isinstance(global_prototypes, bytes):
            target_prototypes = bytes_to_prototypes(global_prototypes, device=device)
        elif isinstance(global_prototypes, dict):
            target_prototypes = {
                int(k): v.to(device=device, dtype=torch.float32)
                for k, v in global_prototypes.items()
            }

        model.train()
        total_samples = 0
        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0

        for _ in range(epochs):
            for batch in dataloader:
                if not batch or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                batch_size = targets.size(0)
                if batch_size == 0:
                    continue

                optimizer.zero_grad()

                try:
                    out = model(inputs, return_features=True)
                    if isinstance(out, tuple) and len(out) == 2:
                        logits, features = out
                    else:
                        logits = out
                        features = (
                            model.extract_features(inputs)
                            if hasattr(model, "extract_features")
                            else None
                        )
                except TypeError:
                    if hasattr(model, "extract_features"):
                        features = model.extract_features(inputs)
                        logits = (
                            model.fc_out(features)
                            if hasattr(model, "fc_out")
                            else model(inputs)
                        )
                    else:
                        logits = model(inputs)
                        features = None

                loss_ce = criterion(logits, targets)

                loss_proto = torch.tensor(0.0, device=device)
                if features is not None and target_prototypes and effective_lambda > 0.0:
                    valid_indices = [
                        i
                        for i, y in enumerate(targets)
                        if int(y.item()) in target_prototypes
                    ]
                    if valid_indices:
                        batch_z = features[valid_indices]
                        batch_p = torch.stack(
                            [
                                target_prototypes[int(targets[i].item())]
                                for i in valid_indices
                            ],
                            dim=0,
                        ).detach()

                        if self.loss_type == "cosine":
                            cos_sim = F.cosine_similarity(batch_z, batch_p, dim=-1)
                            loss_proto = (1.0 - cos_sim).mean()
                        else:  # "mse"
                            loss_proto = ((batch_z - batch_p) ** 2).sum(dim=-1).mean()

                loss_total = loss_ce + effective_lambda * loss_proto
                loss_total.backward()
                optimizer.step()

                total_samples += batch_size
                total_loss += float(loss_total.item()) * batch_size
                total_ce_loss += float(loss_ce.item()) * batch_size
                total_proto_loss += float(loss_proto.item()) * batch_size

        local_prototypes, sample_counts = compute_local_prototypes(
            model,
            dataloader,
            device=device,
            normalize=self.normalize_prototypes,
        )
        self.last_local_prototypes = local_prototypes
        self.last_sample_counts = sample_counts
        self.last_prototypes_bytes = prototypes_to_bytes(local_prototypes)

        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        avg_ce_loss = (total_ce_loss / total_samples) if total_samples > 0 else 0.0
        avg_proto_loss = (total_proto_loss / total_samples) if total_samples > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "loss_ce": avg_ce_loss,
            "loss_proto": avg_proto_loss,
        }

        return model_to_bytes(model), total_samples, metrics


DFPLTrainer = DFPLClientTrainer

__all__ = [
    "DFPLClientTrainer",
    "DFPLTrainer",
]
