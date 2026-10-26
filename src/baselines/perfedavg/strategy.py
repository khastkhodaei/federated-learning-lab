"""Per-FedAvg Server Strategy.

Implements the coordinator aggregation for Per-FedAvg (Fallah et al., 2020).
In Per-FedAvg, client meta-model parameter updates are aggregated using standard
sample-weighted parameter averaging (FedAvg).
"""

from src.baselines.fedavg.strategy import FedAvgServerStrategy


class PerFedAvgServerStrategy(FedAvgServerStrategy):
    """Canonical Per-FedAvg server aggregation strategy."""


__all__ = ["PerFedAvgServerStrategy"]
