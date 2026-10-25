"""Model evaluation and global benchmark dataset loading utilities.

Provides baseline-agnostic evaluation routines for calculating loss,
top-1 accuracy, and sample counts on centralized or client test sets.
"""

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets.transforms import get_default_transform


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

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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
