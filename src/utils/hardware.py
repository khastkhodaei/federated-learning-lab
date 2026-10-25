"""Hardware detection and diagnostic utilities for Federated Learning Lab."""

import os
from typing import Any

import torch


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
