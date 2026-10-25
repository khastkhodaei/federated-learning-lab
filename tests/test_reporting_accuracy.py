"""Unit tests for publication-grade accuracy and dispersion plot reporting."""

import tempfile
from pathlib import Path

from src.utils.reporting import save_baseline_report


def test_save_baseline_report_generates_accuracy_and_dispersion_plots() -> None:
    history = [
        {
            "round": 1,
            "weighted_loss": 1.25,
            "duration_seconds": 2.1,
            "global_test_accuracy": 0.42,
            "global_test_loss": 1.30,
            "mean_local_test_accuracy": 0.65,
            "std_local_test_accuracy": 0.08,
            "worst_10_local_accuracy": 0.50,
            "client_metrics": {
                "client_0": {"samples": 500, "test_accuracy": 0.70},
                "client_1": {"samples": 500, "test_accuracy": 0.60},
            },
        },
        {
            "round": 2,
            "weighted_loss": 0.85,
            "duration_seconds": 1.9,
            "global_test_accuracy": 0.58,
            "global_test_loss": 0.95,
            "mean_local_test_accuracy": 0.78,
            "std_local_test_accuracy": 0.05,
            "worst_10_local_accuracy": 0.70,
            "client_metrics": {
                "client_0": {"samples": 500, "test_accuracy": 0.82},
                "client_1": {"samples": 500, "test_accuracy": 0.74},
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts = save_baseline_report(
            baseline_name="fedavg",
            round_history=history,
            output_dir=tmp_dir,
        )

        assert "accuracy_curves" in artifacts
        assert "client_dispersion" in artifacts
        assert Path(artifacts["accuracy_curves"]).is_file()
        assert Path(artifacts["client_dispersion"]).is_file()
        assert Path(artifacts["loss_curve"]).is_file()
        assert Path(artifacts["round_duration"]).is_file()
        assert Path(artifacts["metrics_json"]).is_file()
