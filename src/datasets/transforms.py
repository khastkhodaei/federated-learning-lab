from typing import Callable, Optional
from torchvision import transforms


def get_default_transform(dataset_name: str) -> Optional[Callable]:
    """Returns standard normalization and ToTensor transforms for torchvision datasets.

    Args:
        dataset_name: Identifier of the dataset (e.g. 'cifar10', 'mnist').

    Returns:
        Compose transform or ToTensor callable.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")
    if key in ("cifar10", "cifar100"):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )
    elif key in ("mnist", "fashionmnist"):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
    return transforms.ToTensor()
