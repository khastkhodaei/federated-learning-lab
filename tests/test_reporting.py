import json
import tempfile
from pathlib import Path

import pytest

from src.utils.reporting import (
    parse_client_message_metrics,
    save_baseline_report,
)


def test_parse_client_message_metrics_json() -> None:
    msg = json.dumps({"loss": 1.4523, "accuracy": 0.65})
    metrics = parse_client_message_metrics(msg)
    assert metrics["loss"] == pytest.approx(1.4523)
    assert metrics["accuracy"] == pytest.approx(0.65)


def test_parse_client_message_metrics_text_fallback() -> None:
    msg = "Client finished round 1. loss=0.8765 epochs=2"
    metrics = parse_client_message_metrics(msg)
    assert metrics["loss"] == pytest.approx(0.8765)

    msg_empty = "Just a text message"
    assert parse_client_message_metrics(msg_empty) == {}


def test_save_baseline_report_creates_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        round_history = [
            {
                "round": 1,
                "duration_seconds": 25.4,
                "total_samples": 50000,
                "weighted_loss": 2.15,
                "client_metrics": {
                    "client_0": {"samples": 25000, "loss": 2.20},
                    "client_1": {"samples": 25000, "loss": 2.10},
                },
            },
            {
                "round": 2,
                "duration_seconds": 24.8,
                "total_samples": 50000,
                "weighted_loss": 1.75,
                "client_metrics": {
                    "client_0": {"samples": 25000, "loss": 1.80},
                    "client_1": {"samples": 25000, "loss": 1.70},
                },
            },
        ]

        artifacts = save_baseline_report(
            baseline_name="fedavg",
            round_history=round_history,
            output_dir=tmp_dir,
        )

        target_dir = Path(tmp_dir) / "fedavg"
        assert target_dir.is_dir()

        json_path = target_dir / "metrics.json"
        loss_plot_path = target_dir / "loss_curve.png"
        duration_plot_path = target_dir / "round_duration.png"

        assert json_path.is_file()
        assert loss_plot_path.is_file()
        assert duration_plot_path.is_file()

        assert artifacts["metrics_json"] == str(json_path)
        assert artifacts["loss_curve"] == str(loss_plot_path)
        assert artifacts["round_duration"] == str(duration_plot_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["baseline"] == "fedavg"
        assert data["total_rounds"] == 2
        assert len(data["rounds"]) == 2
        assert data["rounds"][0]["weighted_loss"] == pytest.approx(2.15)


def test_save_baseline_report_supports_different_baselines() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for baseline in ["fedavg", "fedprox", "scaffold"]:
            artifacts = save_baseline_report(
                baseline_name=baseline,
                round_history=[{"round": 1, "duration_seconds": 10.0, "total_samples": 100, "weighted_loss": 0.5}],
                output_dir=tmp_dir,
            )
            assert Path(artifacts["metrics_json"]).is_file()
            assert Path(artifacts["loss_curve"]).is_file()
            assert Path(artifacts["round_duration"]).is_file()
            assert (Path(tmp_dir) / baseline / "metrics.json").is_file()
            assert (Path(tmp_dir) / baseline / "loss_curve.png").is_file()
            assert (Path(tmp_dir) / baseline / "round_duration.png").is_file()


def test_save_baseline_report_does_not_generate_training_curves() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts = save_baseline_report(
            baseline_name="fedavg",
            round_history=[{"round": 1, "duration_seconds": 1.0, "total_samples": 10, "weighted_loss": 0.5}],
            output_dir=tmp_dir,
        )
        assert "training_curves" not in artifacts
        assert not (Path(tmp_dir) / "fedavg" / "training_curves.png").exists()
