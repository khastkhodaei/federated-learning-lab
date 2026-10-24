from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple 2-layer Convolutional Neural Network with 1 hidden FC layer for CIFAR-10."""

    def __init__(
        self, in_channels: int = 3, num_classes: int = 10, seed: Optional[int] = None
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 max-poolings: 32x32 -> 16x16 -> 8x8 with 64 channels
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc_out(x)
