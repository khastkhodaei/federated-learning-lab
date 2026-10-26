"""Per-FedAvg Client Local Trainer (FO-MAML).

Implements the First-Order Model-Agnostic Meta-Learning (FO-MAML) client optimization
algorithm for Personalized Federated Learning (Fallah et al., 2020).
"""

from collections.abc import Iterable
import copy
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.baselines.base import ClientTrainer
from src.utils.evaluation import evaluate_model
from src.utils.serialization import model_to_bytes


class PerFedAvgClientTrainer(ClientTrainer):
    """Local client trainer implementing First-Order Per-FedAvg (FO-MAML)."""

    def __init__(
        self,
        alpha_lr: float = 0.01,
        beta_lr: float = 0.005,
    ) -> None:
        self.alpha_lr = float(alpha_lr)
        self.beta_lr = float(beta_lr)

    def train(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        global_weights: bytes | None = None,
        alpha_lr: float | None = None,
        beta_lr: float | None = None,
        **kwargs: Any,
    ) -> tuple[bytes, int, dict[str, float]]:
        """Executes First-Order MAML meta-learning steps on local client batches.

        1. Sample batch D1 -> compute inner gradient grad_w L(w; D1).
        2. Compute temporary adapted weights w' = w - alpha * grad1.
        3. Sample batch D2 -> compute outer gradient g = grad_{w'} L(w'; D2).
        4. Update actual model parameters w_new = w - beta * g.
        """
        model.train()
        effective_alpha = float(alpha_lr if alpha_lr is not None else self.alpha_lr)
        effective_beta = float(beta_lr if beta_lr is not None else self.beta_lr)

        total_samples = 0
        total_meta_loss = 0.0
        batches_count = 0

        for _ in range(epochs):
            iterator = iter(dataloader)
            for batch1 in iterator:
                try:
                    batch2 = next(iterator)
                except StopIteration:
                    batch2 = batch1

                x1, y1 = batch1[0].to(device), batch1[1].to(device)
                x2, y2 = batch2[0].to(device), batch2[1].to(device)

                # a) Inner step: Compute gradient on D1
                model.zero_grad()
                out1 = model(x1)
                loss1 = criterion(out1, y1)
                grads1 = torch.autograd.grad(
                    loss1, model.parameters(), retain_graph=False, create_graph=False
                )

                # b) Temporary adapted parameters: w' = w - alpha * grad1
                adapted_params = {
                    name: (param - effective_alpha * g).detach().requires_grad_(True)
                    for (name, param), g in zip(model.named_parameters(), grads1)
                }

                # c) Outer step: Compute loss on D2 evaluated at w'
                out2 = torch.func.functional_call(model, adapted_params, x2)
                loss2 = criterion(out2, y2)
                grads2 = torch.autograd.grad(
                    loss2,
                    list(adapted_params.values()),
                    retain_graph=False,
                    create_graph=False,
                )

                # d) Update actual model parameters: w_new = w - beta * grads2
                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads2):
                        p.sub_(effective_beta * g)

                batch_samples = y1.size(0) + y2.size(0)
                total_samples += batch_samples
                total_meta_loss += loss2.item() * batch_samples
                batches_count += 2

        avg_loss = (total_meta_loss / total_samples) if total_samples > 0 else 0.0
        updated_weights = model_to_bytes(model)

        metrics = {
            "loss": avg_loss,
            "alpha_lr": effective_alpha,
            "beta_lr": effective_beta,
        }

        return updated_weights, total_samples, metrics

    def adapt_and_evaluate(
        self,
        model: nn.Module,
        train_dataloader: Any,
        eval_dataloader: Any = None,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
        adaptation_steps: int = 1,
        val_dataloader: Any = None,
        num_adaptation_steps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Performs 1-step local SGD adaptation on local training data, then evaluates on validation data."""
        target_eval_loader = eval_dataloader if eval_dataloader is not None else val_dataloader
        steps = num_adaptation_steps if num_adaptation_steps is not None else adaptation_steps
        if train_dataloader is None or target_eval_loader is None or model is None:
            return {"loss": 0.0, "accuracy": 0.0, "correct": 0, "total": 0}

        target_device = torch.device(device) if isinstance(device, str) else device
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Clone model to prevent mutating the global meta-model
        adapted_model = copy.deepcopy(model).to(target_device)
        adapted_model.train()

        train_iter = iter(train_dataloader)
        for _ in range(max(1, steps)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            inputs, targets = batch[0].to(target_device), batch[1].to(target_device)
            adapted_model.zero_grad()
            outputs = adapted_model(inputs)
            loss = criterion(outputs, targets)
            grads = torch.autograd.grad(loss, adapted_model.parameters())
            with torch.no_grad():
                for p, g in zip(adapted_model.parameters(), grads):
                    p.sub_(self.alpha_lr * g)

        # Evaluate fine-tuned model on local validation split
        return evaluate_model(
            model=adapted_model,
            dataloader=target_eval_loader,
            criterion=criterion,
            device=target_device,
        )


# Factory alias
PerFedAvgTrainer = PerFedAvgClientTrainer

__all__ = [
    "PerFedAvgClientTrainer",
    "PerFedAvgTrainer",
]
