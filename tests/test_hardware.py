from unittest.mock import patch

from src.utils.hardware import get_hardware_summary


def test_get_hardware_summary_cpu_default() -> None:
    summary = get_hardware_summary("cpu")
    assert isinstance(summary, str)
    assert summary.startswith("CPU")
    assert "cores" in summary


def test_get_hardware_summary_cuda_mocked() -> None:
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
        summary = get_hardware_summary("cuda")
        assert "GPU" in summary
        assert "NVIDIA RTX 4090" in summary


def test_get_hardware_summary_cuda_multi_gpu() -> None:
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4), \
         patch("torch.cuda.get_device_name", return_value="A100"):
        summary = get_hardware_summary("cuda")
        assert "4x A100" in summary
