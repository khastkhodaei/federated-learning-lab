"""Ditto Server Strategy.

Implements the central coordinator aggregation for Ditto (Li et al., 2021).
Since clients only transmit their updated global model weights to the server,
aggregation is performed using standard sample-weighted averaging (FedAvg).
"""

from src.baselines.fedavg.strategy import FedAvgServerStrategy


class DittoServerStrategy(FedAvgServerStrategy):
    """Canonical Ditto server aggregation strategy."""


__all__ = ["DittoServerStrategy"]
