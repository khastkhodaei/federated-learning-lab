"""Local-Only non-federated reference baseline."""

from .strategy import LocalOnlyServerStrategy
from .trainer import LocalOnlyClientTrainer, LocalOnlyTrainer

__all__ = [
    "LocalOnlyClientTrainer",
    "LocalOnlyServerStrategy",
    "LocalOnlyTrainer",
]
