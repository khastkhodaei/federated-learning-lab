"""DFedPGP baseline package."""

from src.baselines.dfedpgp.partition import (
    DEFAULT_HEAD_PREFIXES,
    ModelPartition,
    partition_model,
    split_named_parameters,
)

__all__ = [
    "DEFAULT_HEAD_PREFIXES",
    "ModelPartition",
    "partition_model",
    "split_named_parameters",
]
