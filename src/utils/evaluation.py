"""Model evaluation and global benchmark dataset loading utilities.

Provides baseline-agnostic evaluation routines for calculating loss,
top-1 accuracy, and sample counts on centralized or client test sets.
"""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets.transforms import get_default_transform


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
