import pytest
from pathlib import Path
import tempfile
import yaml

from launch_terminals import load_launch_config, build_launch_commands


def test_load_launch_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        server_yaml = Path(tmpdir) / "server.yaml"
        server_yaml.write_text(
            yaml.dump({
                "host": "127.0.0.1",
                "port": 50055,
                "num_clients": 3,
                "rounds": 4,
                "epochs_per_round": 2,
            }),
            encoding="utf-8"
        )
        cfg = load_launch_config(server_config_path=str(server_yaml))
        assert cfg["host"] == "127.0.0.1"
        assert cfg["port"] == 50055
        assert cfg["num_clients"] == 3
        assert cfg["rounds"] == 4
        assert cfg["epochs"] == 2


def test_build_launch_commands():
    cfg = {
        "host": "127.0.0.1",
        "port": 50051,
        "num_clients": 2,
        "rounds": 3,
        "epochs": 1,
        "server_config": "configs/server.yaml",
        "client_config": "configs/client.yaml",
    }
    commands = build_launch_commands(cfg)
    assert len(commands["clients"]) == 2
    assert "run_server.py" in commands["server"]
    assert "--port 50051" in commands["server"]
    assert "--clients 2" in commands["server"]

    assert "client_0" in commands["clients"][0]["name"]
    assert "--id 0" in commands["clients"][0]["command"]
    assert "--server 127.0.0.1:50051" in commands["clients"][0]["command"]

    assert "client_1" in commands["clients"][1]["name"]
    assert "--id 1" in commands["clients"][1]["command"]
