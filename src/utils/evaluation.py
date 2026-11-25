"""Model evaluation and global benchmark dataset loading utilities.

Provides baseline-agnostic evaluation routines for calculating loss,
top-1 accuracy, and sample counts on centralized or client test sets.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets.transforms import get_default_transform

logger = logging.getLogger(__name__)


def parse_client_metrics(message: str) -> dict[str, float]:
    """Parses key-value metric pairs formatted as 'key1=val1;key2=val2' from TrainResponse.message.

    Args:
        message: Semi-colon separated key-value string.

    Returns:
        Dictionary mapping metric names to parsed float values.
    """
    metrics: dict[str, float] = {}
    if not message or not isinstance(message, str):
        return metrics

    for part in message.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            metrics[key] = float(val)
        except ValueError:
            pass
    return metrics


def aggregate_client_round_metrics(
    responses: Sequence[Any],
    round_id: int | None = None,
) -> dict[str, Any]:
    """Aggregates parsed client metrics across successful responses without zero-imputation.

    Excludes clients omitting expected keys ('loss', 'test_acc') from mean denominators
    and logs an explicit warning when any successful response lacks expected metric keys.
    """
    client_losses: list[float] = []
    client_accs: list[float] = []
    client_metrics: dict[str, dict[str, Any]] = {}
    weighted_loss_sum = 0.0
    loss_samples = 0

    for res in responses:
        if not getattr(res, "success", True):
            continue

        node_id = str(getattr(res, "node_id", "unknown"))
        num_samples = int(getattr(res, "num_samples", 0))
        message = str(getattr(res, "message", ""))
        parsed = parse_client_metrics(message)

        missing_keys = [k for k in ("loss", "test_acc") if k not in parsed]
        if missing_keys:
            round_label = f" (round {round_id})" if round_id is not None else ""
            logger.warning(
                "Client '%s'%s missing expected metric keys: %s; excluding from mean calculations.",
                node_id,
                round_label,
                ", ".join(missing_keys),
            )

        entry: dict[str, Any] = {"samples": num_samples}
        if "loss" in parsed:
            c_loss = parsed["loss"]
            client_losses.append(c_loss)
            entry["loss"] = round(c_loss, 4)
            if num_samples > 0:
                weighted_loss_sum += c_loss * num_samples
                loss_samples += num_samples
        if "test_acc" in parsed:
            c_acc = parsed["test_acc"]
            client_accs.append(c_acc)
            entry["test_accuracy"] = round(c_acc, 4)
        if "test_loss" in parsed:
            entry["test_loss"] = round(parsed["test_loss"], 4)

        client_metrics[node_id] = entry

    mean_loss = (sum(client_losses) / len(client_losses)) if client_losses else None
    weighted_loss = (weighted_loss_sum / loss_samples) if loss_samples > 0 else mean_loss
    mean_local_acc = (sum(client_accs) / len(client_accs)) if client_accs else None

    return {
        "client_losses": client_losses,
        "client_accs": client_accs,
        "client_metrics": client_metrics,
        "mean_loss": mean_loss,
        "weighted_loss": weighted_loss,
        "mean_local_acc": mean_local_acc,
    }


def compute_fairness_metrics(client_accuracies: list[float]) -> dict[str, float]:
    """Calculates dispersion and fairness statistics for client local test accuracies.

    Args:
        client_accuracies: List of client test accuracies (in [0.0, 1.0]).

    Returns:
        Dictionary containing 'mean', 'std', 'worst_10', and 'best_10'.
    """
    if not client_accuracies:
        return {"mean": 0.0, "std": 0.0, "worst_10": 0.0, "best_10": 0.0}

    arr = np.array(client_accuracies, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
    worst_10 = float(np.percentile(arr, 10))
    best_10 = float(np.percentile(arr, 90))

    return {
        "mean": mean_val,
        "std": std_val,
        "worst_10": worst_10,
        "best_10": best_10,
    }


def compute_majority_class_accuracy(
    train_dataloader: Iterable[Any] | None,
    eval_dataloader: Iterable[Any] | None,
) -> float:
    """Computes accuracy of a constant majority-class predictor on eval_dataloader.

    Selects the mode class from ``train_dataloader`` (or ``eval_dataloader`` if training
    data is absent) and measures its accuracy on ``eval_dataloader``.
    """
    if eval_dataloader is None:
        return 0.0

    eval_counts: Counter[int] = Counter()
    total_eval = 0
    for batch in eval_dataloader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            continue
        for lbl in batch[1].view(-1).tolist():
            eval_counts[int(lbl)] += 1
            total_eval += 1

    if total_eval == 0:
        return 0.0

    train_counts: Counter[int] = Counter()
    if train_dataloader is not None:
        for batch in train_dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            for lbl in batch[1].view(-1).tolist():
                train_counts[int(lbl)] += 1

    source_counts = train_counts if train_counts else eval_counts
    majority_label = max(source_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return float(eval_counts[majority_label] / total_eval)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluates a PyTorch model on a given DataLoader.

    Args:
        model: PyTorch neural network to evaluate.
        dataloader: DataLoader containing evaluation batches.
        criterion: Optional loss function (defaults to CrossEntropyLoss with sum reduction).
        device: Device to run evaluation on (defaults to model's parameter device or CPU).

    Returns:
        Dictionary containing:
            - 'loss': Average loss per sample (float).
            - 'accuracy': Top-1 accuracy in [0.0, 1.0] (float).
            - 'correct': Count of correctly predicted samples (int).
            - 'total': Total number of samples evaluated (int).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = targets.size(0)
                total += batch_size
                total_loss += float(loss.item())

                _, preds = torch.max(outputs, 1)
                correct += int((preds == targets).sum().item())
    finally:
        model.train(was_training)

    avg_loss = (total_loss / total) if total > 0 else 0.0
    acc = (correct / total) if total > 0 else 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "correct": int(correct),
        "total": int(total),
    }


def get_global_test_loader(
    dataset_name: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = False,
) -> DataLoader:
    """Returns a DataLoader for the standard held-out evaluation test set.

    Args:
        dataset_name: Name of the benchmark dataset ('cifar10', 'mnist', 'fashionmnist').
        data_dir: Root directory where datasets are cached.
        batch_size: Evaluation batch size.
        num_workers: Number of DataLoader worker subprocesses.
        download: Whether to trigger torchvision download if not locally present.

    Returns:
        PyTorch DataLoader instance for the test split.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")
    transform = get_default_transform(dataset_name)

    if key in ("cifar10",):
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    elif key in ("cifar100",):
        test_dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    elif key in ("mnist",):
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    elif key in ("fashionmnist",):
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported benchmark dataset for global evaluation: '{dataset_name}'")

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
