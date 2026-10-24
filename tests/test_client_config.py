import os
import tempfile
import torch
import torch.nn as nn
import yaml

from src.client.config import ClientConfig, load_client_config, create_client_node
from src.models.cnn import SimpleCNN


def test_load_client_config_from_yaml() -> None:
    yaml_content = """
node_id: "test_client_1"
server_address: "127.0.0.1:50051"
client_host: "127.0.0.1"
client_port: 50052

model:
  name: "simple_cnn"
  seed: 42

training:
  learning_rate: 0.05
  loss: "cross_entropy"
  device: "cpu"
  optimizer:
    name: "sgd"
    momentum: 0.95
    weight_decay: 0.0005

dataset:
  name: "cifar10"
  batch_size: 16
  client_id: 0
  num_clients: 2
  partition: "dirichlet"
  alpha: 0.5
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = load_client_config(temp_path)
        assert config.node_id == "test_client_1"
        assert config.server_address == "127.0.0.1:50051"
        assert config.client_port == 50052
        assert config.training.learning_rate == 0.05
        assert config.training.loss == "cross_entropy"
        assert config.training.optimizer.name == "sgd"
        assert config.training.optimizer.momentum == 0.95
        assert config.training.optimizer.weight_decay == 0.0005
        assert config.model.name == "simple_cnn"
        assert config.dataset.name == "cifar10"
        assert config.dataset.batch_size == 16
    finally:
        os.remove(temp_path)


def test_create_client_node_from_config_components() -> None:
    # Build node from config without downloading full CIFAR dataset (pass dummy or None)
    config = ClientConfig(
        node_id="client_cfg_node",
        server_address="127.0.0.1:50051",
        client_port=0,
    )
    config.training.learning_rate = 0.03
    config.training.optimizer.momentum = 0.8
    config.training.loss = "cross_entropy"
    config.model.name = "simple_cnn"
    config.model.seed = 100

    node = create_client_node(config, load_dataset=False)

    assert node.node_id == "client_cfg_node"
    assert node.lr == 0.03
    assert isinstance(node.model, SimpleCNN)
    assert isinstance(node.criterion, nn.CrossEntropyLoss)
    assert node.listening_port is not None and node.listening_port > 0
    node.stop(grace=0)
    if node.transport is not None:
        node.transport.close()
