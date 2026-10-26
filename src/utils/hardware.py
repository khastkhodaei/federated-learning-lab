"""Hardware detection and diagnostic utilities for Federated Learning Lab."""

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str | None = None) -> str:
    """Resolves target compute device with safe CPU fallback.

    Semantics:
        - None or empty string: returns "cuda" if CUDA is available else "cpu".
        - "cuda" or "cuda:N": returns requested string if CUDA is available;
          if CUDA is unavailable, logs a warning and falls back to "cpu".
        - Any other value (e.g. "cpu"): returns the requested string.
    """
    if not requested or not str(requested).strip():
        return "cuda" if torch.cuda.is_available() else "cpu"

    req_str = str(requested).strip()
    req_lower = req_str.lower()
    if req_lower.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "Requested device '%s' but CUDA is not available. Falling back to 'cpu'.",
                req_str,
            )
            return "cpu"
        return req_str

    return req_str


def get_hardware_summary(configured_device: str = "cpu") -> str:
    """Returns a concise, human-readable summary of compute hardware (GPU / CPU)."""
    device_lower = configured_device.lower().strip()
    is_cuda_requested = device_lower.startswith("cuda") or (device_lower != "cpu" and torch.cuda.is_available())
    if is_cuda_requested and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        return f"GPU ({gpu_name})" if count == 1 else f"GPU ({count}x {gpu_name})"

    cores = os.cpu_count() or 1
    return f"CPU ({cores} logical cores)"


def get_hardware_details(configured_device: str = "cpu") -> dict[str, Any]:
    """Returns structured hardware information for JSON report generation."""
    cuda_avail = torch.cuda.is_available()
    details: dict[str, Any] = {
        "configured_device": configured_device,
        "cuda_available": cuda_avail,
        "cpu_count": os.cpu_count() or 1,
    }
    if cuda_avail:
        details["device_count"] = torch.cuda.device_count()
        details["device_name"] = torch.cuda.get_device_name(0)
    return details
