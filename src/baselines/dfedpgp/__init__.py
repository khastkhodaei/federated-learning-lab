"""DFedPGP baseline package."""

from src.baselines.dfedpgp.partition import (
    DEFAULT_HEAD_PREFIXES,
    ModelPartition,
    partition_model,
    split_named_parameters,
)
from src.baselines.dfedpgp.strategy import DFedPGPServerStrategy
from src.baselines.dfedpgp.topology import build_directed_mixing_matrix

__all__ = [
    "DEFAULT_HEAD_PREFIXES",
    "DFedPGPServerStrategy",
    "ModelPartition",
    "build_directed_mixing_matrix",
    "partition_model",
    "split_named_parameters",
]
