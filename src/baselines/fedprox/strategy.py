"""Federated Proximal (FedProx) Server Strategy.

Implements the server-side aggregation for FedProx (Li et al., 2020).
In canonical FedProx, the central coordinator applies sample-weighted parameter averaging (FedAvg).
"""

from src.baselines.fedavg.strategy import FedAvgServerStrategy


class FedProxServerStrategy(FedAvgServerStrategy):
    """Canonical FedProx server aggregation strategy (sample-weighted averaging)."""


__all__ = ["FedProxServerStrategy"]
