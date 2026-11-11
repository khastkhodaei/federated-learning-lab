"""DFedPGP client trainer with alternating personal_head and shared_backbone updates."""

from collections.abc import Iterable
import copy
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.baselines.dfedpgp.partition import (
    DEFAULT_HEAD_PREFIXES,
    partition_model,
    split_named_parameters,
)
from src.utils.serialization import (
    bytes_to_vector,
    model_to_bytes,
    model_to_vector,
    vector_to_model,
)


def _validate_hyperparams(
    lr_backbone: float | None,
    lr_head: float | None,
    head_epochs: int | None,
    backbone_epochs: int | None,
) -> None:
    if lr_backbone is not None and lr_backbone <= 0:
        raise ValueError(f"lr_backbone must be > 0, got {lr_backbone}.")
    if lr_head is not None and lr_head <= 0:
        raise ValueError(f"lr_head must be > 0, got {lr_head}.")
    if head_epochs is not None and head_epochs < 0:
        raise ValueError(f"head_epochs must be >= 0, got {head_epochs}.")
    if backbone_epochs is not None and backbone_epochs < 0:
        raise ValueError(f"backbone_epochs must be >= 0, got {backbone_epochs}.")


def _resolve_override(
    overrides: dict[str, Any],
    key: str,
    default: Any,
    caster: Any,
) -> Any:
    value = overrides.get(key)
    return caster(value) if value is not None else default


def _run_phase(
    model: nn.Module,
    dataloader: Iterable[Any],
    epochs: int,
    criterion: nn.Module,
    active_params: list[nn.Parameter],
    frozen_params: list[nn.Parameter],
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[float, int]:
    """Runs local SGD epochs over ``active_params`` while freezing ``frozen_params``."""
    for param in frozen_params:
        param.requires_grad = False
    for param in active_params:
        param.requires_grad = True

    if epochs <= 0:
        return 0.0, 0

    phase_optimizer = torch.optim.SGD(
        active_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    model.train()
    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            phase_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            phase_optimizer.step()

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    return avg_loss, total_samples


class DFedPGPClientTrainer(ClientTrainer):
    """Local client trainer for DFedPGP partial personalization.

    Alternates local SGD updates over the client's private ``personal_head`` and
    the peer-shared ``shared_backbone``, keeping ``personal_head`` local across
    rounds and exposing ``personalized_model`` for local evaluation.
    """

    def __init__(
        self,
        lr_backbone: float | None = None,
        lr_head: float | None = None,
        head_epochs: int | None = None,
        backbone_epochs: int | None = None,
        head_prefixes: tuple[str, ...] = DEFAULT_HEAD_PREFIXES,
    ) -> None:
        """Initializes DFedPGP client trainer hyperparameters and state.

        Args:
            lr_backbone: Optional learning rate override for ``shared_backbone`` updates.
            lr_head: Optional learning rate override for ``personal_head`` updates.
            head_epochs: Optional epoch count override for ``personal_head`` phase.
            backbone_epochs: Optional epoch count override for ``shared_backbone`` phase.
            head_prefixes: Parameter/module prefixes identifying ``personal_head``.

        Raises:
            ValueError: If learning rates are non-positive or phase epochs are negative.
        """
        _validate_hyperparams(lr_backbone, lr_head, head_epochs, backbone_epochs)
        self.lr_backbone: float | None = float(lr_backbone) if lr_backbone is not None else None
        self.lr_head: float | None = float(lr_head) if lr_head is not None else None
        self.head_epochs: int | None = int(head_epochs) if head_epochs is not None else None
        self.backbone_epochs: int | None = (
            int(backbone_epochs) if backbone_epochs is not None else None
        )
        self.head_prefixes: tuple[str, ...] = tuple(head_prefixes)
        self.private_head_vector: torch.Tensor | None = None
        self.personalized_model: nn.Module | None = None

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: bytes | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes alternating local updates for personal_head and shared_backbone.

        Args:
            model: Local neural network model.
            dataloader: DataLoader yielding ``(inputs, targets)`` batches.
            epochs: Base local epoch count used when phase-specific counts are unset.
            criterion: Training loss function.
            optimizer: Base local optimizer supplying default learning rate, momentum,
                and weight decay.
            device: Target computation device.
            global_weights: Optional serialized peer/consensus parameters whose
                ``shared_backbone`` replaces the model's backbone while preserving
                this client's persisted ``personal_head``.
            **kwargs: Optional per-round overrides for ``lr_backbone``, ``lr_head``,
                ``head_epochs``, and ``backbone_epochs``.

        Returns:
            Tuple ``(serialized_weights, total_samples, metrics)`` with ``loss``,
            ``head_loss``, and ``backbone_loss``.

        Raises:
            ValueError: If overridden hyperparameters are invalid, ``model`` cannot be
                partitioned, or ``private_head_vector`` dimension mismatches ``head_dim``.
        """
        raw_lr_backbone: float | None = _resolve_override(
            kwargs, "lr_backbone", self.lr_backbone, float
        )
        raw_lr_head: float | None = _resolve_override(
            kwargs, "lr_head", self.lr_head, float
        )
        raw_head_epochs: int | None = _resolve_override(
            kwargs, "head_epochs", self.head_epochs, int
        )
        raw_backbone_epochs: int | None = _resolve_override(
            kwargs, "backbone_epochs", self.backbone_epochs, int
        )
        _validate_hyperparams(
            raw_lr_backbone,
            raw_lr_head,
            raw_head_epochs,
            raw_backbone_epochs,
        )

        partition = partition_model(model, head_prefixes=self.head_prefixes)
        backbone_params, head_params = split_named_parameters(
            model, head_prefixes=self.head_prefixes
        )

        incoming_vec = (
            bytes_to_vector(global_weights)
            if (global_weights is not None and len(global_weights) > 0)
            else model_to_vector(model)
        )
        incoming_backbone, incoming_head = partition.split_vector(incoming_vec)

        if self.private_head_vector is None:
            private_head = incoming_head.detach().cpu().clone()
        else:
            if self.private_head_vector.numel() != partition.head_dim:
                raise ValueError(
                    f"Persisted personal_head dimension ({self.private_head_vector.numel()}) "
                    f"does not match model head_dim ({partition.head_dim})."
                )
            private_head = self.private_head_vector

        vector_to_model(partition.merge_vectors(incoming_backbone, private_head), model)

        group0 = optimizer.param_groups[0] if optimizer.param_groups else {}
        opt_lr = float(group0.get("lr", 0.01))
        opt_momentum = float(group0.get("momentum", 0.0))
        opt_wd = float(group0.get("weight_decay", 0.0))

        eff_lr_head = raw_lr_head if raw_lr_head is not None else opt_lr
        eff_lr_backbone = raw_lr_backbone if raw_lr_backbone is not None else opt_lr
        eff_head_epochs = raw_head_epochs if raw_head_epochs is not None else max(0, int(epochs))
        eff_backbone_epochs = (
            raw_backbone_epochs if raw_backbone_epochs is not None else max(0, int(epochs))
        )

        orig_requires_grad = [(param, param.requires_grad) for param in model.parameters()]
        try:
            head_loss, _ = _run_phase(
                model=model,
                dataloader=dataloader,
                epochs=eff_head_epochs,
                criterion=criterion,
                active_params=head_params,
                frozen_params=backbone_params,
                lr=eff_lr_head,
                momentum=opt_momentum,
                weight_decay=opt_wd,
                device=device,
            )
            backbone_loss, total_samples = _run_phase(
                model=model,
                dataloader=dataloader,
                epochs=eff_backbone_epochs,
                criterion=criterion,
                active_params=backbone_params,
                frozen_params=head_params,
                lr=eff_lr_backbone,
                momentum=opt_momentum,
                weight_decay=opt_wd,
                device=device,
            )
        finally:
            for param, was_trainable in orig_requires_grad:
                param.requires_grad = was_trainable

        _, updated_head = partition.split_vector(model_to_vector(model))
        self.private_head_vector = updated_head.detach().cpu().clone()

        if self.personalized_model is None:
            self.personalized_model = copy.deepcopy(model).to(device)
        else:
            self.personalized_model.load_state_dict(copy.deepcopy(model.state_dict()))
            self.personalized_model.to(device)

        primary_loss = backbone_loss if total_samples > 0 else head_loss
        metrics = {
            "loss": primary_loss,
            "head_loss": head_loss,
            "backbone_loss": backbone_loss,
        }
        return model_to_bytes(model), total_samples, metrics


DFedPGPTrainer = DFedPGPClientTrainer

__all__ = [
    "DFedPGPClientTrainer",
    "DFedPGPTrainer",
]
