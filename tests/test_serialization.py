import torch
from src.models.cnn import SimpleCNN
from src.utils.serialization import (
    bytes_to_vector,
    load_bytes_into_model,
    model_to_bytes,
    model_to_vector,
    vector_to_bytes,
    vector_to_model,
)


def test_model_to_vector_shape_and_dtype() -> None:
    model = SimpleCNN()
    vec = model_to_vector(model)

    assert vec.ndim == 1
    assert vec.dtype == torch.float32

    # Calculate expected parameter count
    expected_params = sum(p.numel() for p in model.parameters())
    assert vec.shape[0] == expected_params


def test_serialization_roundtrip_simple_cnn() -> None:
    source_model = SimpleCNN()
    target_model = SimpleCNN()

    # Ensure source and target start with different random weights
    with torch.no_grad():
        for p in source_model.parameters():
            p.add_(1.5)

    raw_bytes = model_to_bytes(source_model)

    # 4 bytes per float32 parameter
    expected_params = sum(p.numel() for p in source_model.parameters())
    assert len(raw_bytes) == expected_params * 4

    # Load bytes into target model
    load_bytes_into_model(raw_bytes, target_model)

    # Verify all layers have exact matching parameters
    for p_src, p_tgt in zip(source_model.parameters(), target_model.parameters()):
        assert torch.allclose(p_src, p_tgt)


def test_vector_to_bytes_roundtrip() -> None:
    original = torch.tensor([1.0, -2.5, 3.14159, 42.0], dtype=torch.float32)
    raw = vector_to_bytes(original)
    reconstructed = bytes_to_vector(raw)

    assert reconstructed.dtype == torch.float32
    assert torch.allclose(original, reconstructed)
