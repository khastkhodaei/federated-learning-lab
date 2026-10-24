import numpy as np
import torch
import torch.nn as nn


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Flattens all parameters of a model into a continuous 1D float32 tensor on CPU."""
    vec = nn.utils.parameters_to_vector(model.parameters())
    return vec.detach().cpu().to(torch.float32)


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    """Unpacks a 1D parameter vector back into a model's layers in-place."""
    device = next(model.parameters()).device
    vec_device = vector.to(device=device, dtype=torch.float32)
    nn.utils.vector_to_parameters(vec_device, model.parameters())


def vector_to_bytes(vector: torch.Tensor) -> bytes:
    """Converts a 1D float32 tensor into raw bytes."""
    return vector.detach().cpu().to(torch.float32).numpy().tobytes()


def bytes_to_vector(data: bytes) -> torch.Tensor:
    """Reconstructs a 1D float32 PyTorch tensor from raw bytes."""
    array = np.frombuffer(data, dtype=np.float32)
    return torch.from_numpy(array.copy())


def model_to_bytes(model: nn.Module) -> bytes:
    """Flattens model parameters to a 1D vector and serializes them to raw float32 bytes."""
    vector = model_to_vector(model)
    return vector_to_bytes(vector)


def load_bytes_into_model(data: bytes, model: nn.Module) -> None:
    """Deserializes raw float32 bytes into a 1D vector and loads it into the model."""
    vector = bytes_to_vector(data)
    vector_to_model(vector, model)
