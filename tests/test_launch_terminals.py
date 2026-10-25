import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from launch_terminals import (
    build_launch_commands,
    generate_distribution_chart,
    launch_windows,
    load_launch_config,
)


def test_load_launch_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server_yaml = Path(tmpdir) / "server.yaml"
        server_yaml.write_text(
            yaml.dump({
                "host": "127.0.0.1",
                "port": 50055,
                "num_clients": 3,
                "rounds": 4,
                "epochs_per_round": 2,
                "baseline": "fedprox",
            }),
            encoding="utf-8",
        )
        cfg = load_launch_config(server_config_path=str(server_yaml))
        assert cfg["host"] == "127.0.0.1"
        assert cfg["port"] == 50055
        assert cfg["num_clients"] == 3
        assert cfg["rounds"] == 4
        assert cfg["epochs"] == 2
        assert cfg["baseline"] == "fedprox"


def test_build_launch_commands() -> None:
    cfg = {
        "host": "127.0.0.1",
        "port": 50051,
        "num_clients": 2,
        "rounds": 3,
        "epochs": 1,
        "baseline": "fedavg",
        "server_config": "configs/server.yaml",
        "client_config": "configs/client.yaml",
    }
    commands = build_launch_commands(cfg)
    assert len(commands["clients"]) == 2
    assert "run_server.py" in commands["server"]
    assert "--port 50051" in commands["server"]
    assert "--clients 2" in commands["server"]
    assert "--baseline fedavg" in commands["server"]

    assert "client_0" in commands["clients"][0]["name"]
    assert "--id 0" in commands["clients"][0]["command"]
    assert "--server 127.0.0.1:50051" in commands["clients"][0]["command"]
    assert "--baseline fedavg" in commands["clients"][0]["command"]

    assert "client_1" in commands["clients"][1]["name"]
    assert "--id 1" in commands["clients"][1]["command"]
    assert "--baseline fedavg" in commands["clients"][1]["command"]


def test_generate_distribution_chart_defaults_to_baseline_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "num_clients": 2,
            "dataset_name": "cifar10",
            "partition": "iid",
            "alpha": 0.5,
            "seed": 42,
            "baseline": "fedavg",
        }
        expected_path = Path(tmpdir) / "fedavg" / "data_distribution.png"
        out_file = generate_distribution_chart(cfg, output_path=str(expected_path))
        assert out_file.is_file()
        assert out_file == expected_path


def test_generate_distribution_chart_does_not_create_distributions_dir() -> None:
    with tempfile.TemporaryDirectory():
        cfg = {
            "num_clients": 2,
            "dataset_name": "cifar10",
            "partition": "iid",
            "alpha": 0.5,
            "seed": 42,
            "baseline": "fedavg",
        }
        out_file = generate_distribution_chart(cfg)
        assert out_file.is_file()
        assert out_file == Path("reports/fedavg/data_distribution.png")
        assert not Path("reports/distributions").exists()
        assert not Path("reports/data_distribution.png").exists()


def test_launch_windows_banner_displays_baseline_method(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = {
        "host": "127.0.0.1",
        "port": 50051,
        "num_clients": 2,
        "rounds": 3,
        "epochs": 1,
        "baseline": "fedavg",
        "server_config": "configs/server.yaml",
        "client_config": "configs/client.yaml",
    }
    launch_windows(cfg, dry_run=True, plot=False)
    captured = capsys.readouterr().out
    assert "Baseline Method:" in captured
    assert "FedAvg" in captured
    assert "Hardware:" in captured


def test_launch_windows_uses_create_new_console_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    mock_popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    cfg = {
        "host": "127.0.0.1",
        "port": 50051,
        "num_clients": 2,
        "rounds": 1,
        "epochs": 1,
        "baseline": "fedavg",
        "server_config": "configs/server.yaml",
        "client_config": "configs/client.yaml",
    }
    launch_windows(cfg, dry_run=False, plot=False)

    assert mock_popen.call_count == 3
    for call in mock_popen.call_args_list:
        kwargs = call[1]
        assert "creationflags" in kwargs
        if sys.platform == "win32":
            assert kwargs["creationflags"] == subprocess.CREATE_NEW_CONSOLE
