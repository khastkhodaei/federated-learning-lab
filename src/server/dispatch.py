"""Centralized federated round dispatch coordinator."""

from typing import Any

from src.server.server import FederatedServer
from src.transport.federated_pb2 import TrainResponse


def dispatch_round(
    server: FederatedServer,
    round_id_or_settings: Any = 1,
    epochs: int | None = None,
    timeout: float | None = None,
    *,
    round_id: int | None = None,
    **kwargs: Any,
) -> list[TrainResponse]:
    """Dispatches a federated training round to all registered clients and aggregates responses.

    Preserves server.client_weights for personalized algorithms by omitting explicit weights overrides.
    """
    if hasattr(round_id_or_settings, "epochs"):
        settings = round_id_or_settings
        r_id = round_id if round_id is not None else (epochs if isinstance(epochs, int) else 1)
        ep = getattr(settings, "epochs", 1)
        to = getattr(settings, "round_timeout", None)
        return server.trigger_round(
            round_id=r_id,
            epochs=ep,
            timeout=to,
            aggregate=True,
        )

    actual_round_id = round_id if round_id is not None else round_id_or_settings
    actual_epochs = epochs if epochs is not None else 1
    return server.trigger_round(
        round_id=actual_round_id,
        epochs=actual_epochs,
        timeout=timeout,
        aggregate=True,
    )
