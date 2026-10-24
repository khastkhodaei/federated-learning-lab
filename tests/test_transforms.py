import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import get_default_transform


class MockPilImageDataset(Dataset):
    """Mock dataset returning PIL Images, mimicking torchvision raw datasets."""

    def __init__(self, size: int = 10, transform=None):
        self.size = size
        self.transform = transform

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # Create 32x32 RGB PIL image
        img = Image.new("RGB", (32, 32), color=(idx * 10, idx * 20, idx * 5))
        label = idx % 10
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def test_dataloader_fails_without_transform() -> None:
    # Reproduces the exact error reported by user: PIL.Image cannot be collated
    raw_dataset = MockPilImageDataset(size=4, transform=None)
    loader = DataLoader(raw_dataset, batch_size=2)

    with pytest.raises(TypeError, match="default_collate: batch must contain tensors.*PIL\\.Image\\.Image"):
        for _ in loader:
            pass


def test_dataloader_succeeds_with_default_cifar_transform() -> None:
    transform = get_default_transform("cifar10")
    transformed_dataset = MockPilImageDataset(size=6, transform=transform)
    loader = DataLoader(transformed_dataset, batch_size=3)

    batches = list(loader)
    assert len(batches) == 2

    inputs, targets = batches[0]
    assert isinstance(inputs, torch.Tensor)
    assert inputs.shape == (3, 3, 32, 32)
    assert inputs.dtype == torch.float32
    assert isinstance(targets, torch.Tensor)
    assert targets.shape == (3,)
