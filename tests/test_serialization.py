import torch
from torch import nn

from src.models.cnn import SimpleCNN
from src.utils.serialization import (
    bytes_to_vector,
    load_bytes_into_model,
    model_to_bytes,
    model_to_vector,
    vector_to_bytes,
)


def test_model_to_vector_shape_and_dtype() -> None:
    model = SimpleCNN()
    vec = model_to_vector(model)

    assert vec.ndim == 1
    assert vec.dtype == torch.float32

    expected_params = sum(p.numel() for p in model.parameters())
    assert vec.shape[0] == expected_params


def test_serialization_roundtrip_simple_cnn() -> None:
    source_model = SimpleCNN()
    target_model = SimpleCNN()

    with torch.no_grad():
        for p in source_model.parameters():
            p.add_(1.5)

    raw_bytes = model_to_bytes(source_model)

    expected_params = sum(p.numel() for p in source_model.parameters())
    assert len(raw_bytes) == expected_params * 4

    load_bytes_into_model(raw_bytes, target_model)

    for p_src, p_tgt in zip(source_model.parameters(), target_model.parameters()):
        assert torch.allclose(p_src, p_tgt)


def test_serialization_roundtrip_bit_exact() -> None:
    source_model = SimpleCNN(seed=42)
    target_model = SimpleCNN(seed=99)

    with torch.no_grad():
        for p in source_model.parameters():
            p.copy_(torch.randn_like(p))

    raw_bytes = model_to_bytes(source_model)
    load_bytes_into_model(raw_bytes, target_model)

    for p_src, p_tgt in zip(source_model.parameters(), target_model.parameters()):
        assert torch.equal(p_src, p_tgt)


def test_vector_to_bytes_roundtrip() -> None:
    original = torch.tensor([1.0, -2.5, 3.14159, 42.0], dtype=torch.float32)
    raw = vector_to_bytes(original)
    reconstructed = bytes_to_vector(raw)

    assert reconstructed.dtype == torch.float32
    assert torch.allclose(original, reconstructed)


def test_vector_to_bytes_bit_exact() -> None:
    original = torch.tensor(
        [
            1e-38,
            1.4012985e-45,
            3.4028235e38,
            -3.4028235e38,
            -0.0,
            0.0,
            1.0,
            -1.0,
            1e-7,
            float("inf"),
            float("-inf"),
        ],
        dtype=torch.float32,
    )
    raw = vector_to_bytes(original)
    reconstructed = bytes_to_vector(raw)

    assert reconstructed.dtype == torch.float32
    assert torch.equal(original, reconstructed)


def test_serialization_preserves_layer_shapes_and_dtypes() -> None:
    source_model = SimpleCNN(seed=7)
    target_model = SimpleCNN(seed=13)

    raw_bytes = model_to_bytes(source_model)
    load_bytes_into_model(raw_bytes, target_model)

    for (name_src, p_src), (name_tgt, p_tgt) in zip(
        source_model.named_parameters(), target_model.named_parameters()
    ):
        assert name_src == name_tgt
        assert p_src.shape == p_tgt.shape
        assert p_src.dtype == torch.float32
        assert p_tgt.dtype == torch.float32
        assert torch.equal(p_src, p_tgt)


def test_serialization_buffers_characterization() -> None:
    cnn = SimpleCNN()
    assert len(list(cnn.buffers())) == 0

    class BufferedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3)
            self.bn = nn.BatchNorm2d(8)
            self.register_buffer("epoch_count", torch.tensor(17, dtype=torch.int64))

    m_src = BufferedModel()
    m_tgt = BufferedModel()

    with torch.no_grad():
        m_src.bn.running_mean.fill_(2.5)
        m_src.bn.running_var.fill_(0.8)
        m_tgt.epoch_count.fill_(99)

    raw_bytes = model_to_bytes(m_src)
    load_bytes_into_model(raw_bytes, m_tgt)

    for p_src, p_tgt in zip(m_src.parameters(), m_tgt.parameters()):
        assert torch.equal(p_src, p_tgt)

    assert m_tgt.epoch_count.item() == 99
