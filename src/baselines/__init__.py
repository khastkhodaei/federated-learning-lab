from typing import Any

from .base import ClientTrainer, ServerStrategy
from .dfedpgp import DFedPGPClientTrainer, DFedPGPServerStrategy, DFedPGPTrainer
from .ditto import DittoClientTrainer, DittoServerStrategy, DittoTrainer
from .fedavg import FedAvgClientTrainer, FedAvgServerStrategy
from .fedprox import FedProxClientTrainer, FedProxServerStrategy, FedProxTrainer
from .perfedavg import PerFedAvgClientTrainer, PerFedAvgServerStrategy, PerFedAvgTrainer
from .pfedgraph import PFedGraphClientTrainer, PFedGraphServerStrategy, PFedGraphTrainer

BASELINES: dict[str, tuple[type[ServerStrategy], type[ClientTrainer]]] = {
    "fedavg": (FedAvgServerStrategy, FedAvgClientTrainer),
    "fedprox": (FedProxServerStrategy, FedProxClientTrainer),
    "ditto": (DittoServerStrategy, DittoClientTrainer),
    "perfedavg": (PerFedAvgServerStrategy, PerFedAvgClientTrainer),
    "pfedgraph": (PFedGraphServerStrategy, PFedGraphClientTrainer),
    "dfedpgp": (DFedPGPServerStrategy, DFedPGPClientTrainer),
}


def get_baseline(name: str) -> tuple[type[ServerStrategy], type[ClientTrainer]]:
    """Resolves a baseline name to its corresponding (ServerStrategy, ClientTrainer) classes."""
    key = name.lower().strip()
    if key not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{name}'. Available baselines: {sorted(BASELINES.keys())}"
        )
    return BASELINES[key]


def _resolve_strategy_param(
    kwargs: dict[str, Any],
    key: str,
    source: Any | None,
    default: Any,
) -> Any:
    """Resolves a hyperparameter with precedence kwargs > source attribute > default."""
    if kwargs.get(key) is not None:
        return kwargs[key]
    if source is not None and getattr(source, key, None) is not None:
        return getattr(source, key)
    return default


def create_server_strategy(
    name: str,
    config: Any | None = None,
    **kwargs: Any,
) -> ServerStrategy:
    """Instantiates a ServerStrategy configured for the given baseline.

    Args:
        name: Baseline algorithm name ('fedavg', 'fedprox', 'ditto', 'perfedavg', 'pfedgraph', 'dfedpgp').
        config: Optional UnifiedConfig instance containing baseline-specific configurations.
        **kwargs: Optional explicit overrides for strategy hyperparameters (e.g. alpha, lam, out_degree).

    Returns:
        Configured ServerStrategy instance.
    """
    key = name.lower().strip()
    strategy_cls, _ = get_baseline(key)
    if key == "dfedpgp":
        dfed_cfg = getattr(config.baselines, "dfedpgp", None) if hasattr(config, "baselines") else None
        out_degree = _resolve_strategy_param(kwargs, "out_degree", dfed_cfg, 1)
        time_varying = _resolve_strategy_param(kwargs, "time_varying", dfed_cfg, True)
        seed = _resolve_strategy_param(kwargs, "seed", config, 42)
        return strategy_cls(
            out_degree=int(out_degree),
            seed=int(seed),
            time_varying=bool(time_varying),
        )
    if key == "pfedgraph":
        alpha = kwargs.get("alpha")
        if alpha is None and "graph_alpha" in kwargs:
            alpha = kwargs.get("graph_alpha")
        if alpha is None and config is not None:
            pfed_cfg = getattr(config.baselines, "pfedgraph", None) if hasattr(config, "baselines") else None
            if pfed_cfg is not None:
                alpha = getattr(pfed_cfg, "alpha", 0.1)
        if alpha is None:
            alpha = 0.1

        lam = kwargs.get("lam")
        if lam is None and "graph_lam" in kwargs:
            lam = kwargs.get("graph_lam")
        if lam is None and config is not None:
            pfed_cfg = getattr(config.baselines, "pfedgraph", None) if hasattr(config, "baselines") else None
            if pfed_cfg is not None:
                lam = getattr(pfed_cfg, "lam", 0.01)
        if lam is None:
            lam = 0.01

        step_mode = kwargs.get("step_mode", "projected_gd")
        return strategy_cls(alpha=alpha, lam=lam, step_mode=step_mode)

    return strategy_cls()


__all__ = [
    "BASELINES",
    "ClientTrainer",
    "DFedPGPClientTrainer",
    "DFedPGPServerStrategy",
    "DFedPGPTrainer",
    "DittoClientTrainer",
    "DittoServerStrategy",
    "DittoTrainer",
    "FedAvgClientTrainer",
    "FedAvgServerStrategy",
    "FedProxClientTrainer",
    "FedProxServerStrategy",
    "FedProxTrainer",
    "PFedGraphClientTrainer",
    "PFedGraphServerStrategy",
    "PFedGraphTrainer",
    "PerFedAvgClientTrainer",
    "PerFedAvgServerStrategy",
    "PerFedAvgTrainer",
    "ServerStrategy",
    "create_server_strategy",
    "get_baseline",
]
