from .base import ClientTrainer, ServerStrategy
from .ditto import DittoClientTrainer, DittoServerStrategy, DittoTrainer
from .fedavg import FedAvgClientTrainer, FedAvgServerStrategy
from .fedprox import FedProxClientTrainer, FedProxServerStrategy, FedProxTrainer
from .perfedavg import PerFedAvgClientTrainer, PerFedAvgServerStrategy, PerFedAvgTrainer

BASELINES: dict[str, tuple[type[ServerStrategy], type[ClientTrainer]]] = {
    "fedavg": (FedAvgServerStrategy, FedAvgClientTrainer),
    "fedprox": (FedProxServerStrategy, FedProxClientTrainer),
    "ditto": (DittoServerStrategy, DittoClientTrainer),
    "perfedavg": (PerFedAvgServerStrategy, PerFedAvgClientTrainer),
}


def get_baseline(name: str) -> tuple[type[ServerStrategy], type[ClientTrainer]]:
    """Resolves a baseline name to its corresponding (ServerStrategy, ClientTrainer) classes."""
    key = name.lower().strip()
    if key not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{name}'. Available baselines: {sorted(BASELINES.keys())}"
        )
    return BASELINES[key]


__all__ = [
    "BASELINES",
    "ClientTrainer",
    "DittoClientTrainer",
    "DittoServerStrategy",
    "DittoTrainer",
    "FedAvgClientTrainer",
    "FedAvgServerStrategy",
    "FedProxClientTrainer",
    "FedProxServerStrategy",
    "FedProxTrainer",
    "PerFedAvgClientTrainer",
    "PerFedAvgServerStrategy",
    "PerFedAvgTrainer",
    "ServerStrategy",
    "get_baseline",
]
