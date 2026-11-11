"""DFedPGP baseline package."""

from src.baselines.dfedpgp.partition import (
    DEFAULT_HEAD_PREFIXES,
    ModelPartition,
    partition_model,
    split_named_parameters,
)
from src.baselines.dfedpgp.strategy import DFedPGPServerStrategy
from src.baselines.dfedpgp.topology import build_directed_mixing_matrix
from src.baselines.dfedpgp.trainer import DFedPGPClientTrainer, DFedPGPTrainer

__all__ = [
    "DEFAULT_HEAD_PREFIXES",
    "DFedPGPClientTrainer",
    "DFedPGPServerStrategy",
    "DFedPGPTrainer",
    "ModelPartition",
    "build_directed_mixing_matrix",
    "partition_model",
    "split_named_parameters",
]
