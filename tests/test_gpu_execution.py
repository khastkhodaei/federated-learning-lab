"""Tests for verifying CUDA GPU device propagation and execution."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from launch_terminals import build_launch_commands
from src.client.config import ClientConfig, create_client_node
from src.client.node import Node
from src.models.cnn import SimpleCNN


def test_node_device_parameter_propagation() -> None:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN()
    node = Node(node_id="test_gpu_prop", model=model, device=device_name)

    assert node.device == torch.device(device_name)
    param_device = next(node.model.parameters()).device.type
    assert param_device == torch.device(device_name).type


def test_create_client_node_respects_cuda_device() -> None:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ClientConfig(node_id="cfg_gpu_node")
    cfg.training.device = device_name

    node = create_client_node(cfg, load_dataset=False)
    try:
        assert node.device == torch.device(device_name)
        assert (
            next(node.model.parameters()).device.type
            == torch.device(device_name).type
        )
    finally:
        node.stop(grace=0)


def test_node_train_local_on_cuda_updates_parameters() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    node = Node(
        node_id="cuda_train_node", model=model, data=loader, device="cuda"
    )
    assert next(node.model.parameters()).is_cuda

    updated_weights, samples = node.train_local(epochs=1)
    assert samples == 4
    assert len(updated_weights) > 0
    assert next(node.model.parameters()).is_cuda


def test_launch_terminals_build_commands_passes_device_flag() -> None:
    cfg = {
        "host": "127.0.0.1",
        "port": 50051,
        "num_clients": 2,
        "rounds": 3,
        "epochs": 1,
        "baseline": "fedavg",
        "device": "cuda",
        "config_path": "configs/config.yaml",
    }
    cmds = build_launch_commands(cfg)
    assert "--device cuda" in cmds["server"]
    for client_info in cmds["clients"]:
        assert "--device cuda" in client_info["command"]
