from .base import ClientTrainer, ServerStrategy
from .fedavg import FedAvgClientTrainer, FedAvgServerStrategy
from .fedprox import FedProxClientTrainer, FedProxServerStrategy, FedProxTrainer

BASELINES: dict[str, tuple[type[ServerStrategy], type[ClientTrainer]]] = {
    "fedavg": (FedAvgServerStrategy, FedAvgClientTrainer),
    "fedprox": (FedProxServerStrategy, FedProxClientTrainer),
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
    "FedAvgClientTrainer",
    "FedAvgServerStrategy",
    "FedProxClientTrainer",
    "FedProxServerStrategy",
    "FedProxTrainer",
    "ServerStrategy",
    "get_baseline",
]
