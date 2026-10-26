import json
from pathlib import Path
import pytest

from src.utils.benchmark_plotting import (
    load_benchmark_metrics,
    plot_benchmark_summary,
    plot_global_accuracy_comparison,
    plot_local_accuracy_comparison,
)
from run_benchmark import run_full_benchmark


@pytest.fixture
def mock_benchmark_dir(tmp_path: Path) -> Path:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    algos = ["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph"]
    for i, algo in enumerate(algos):
        algo_dir = reports_dir / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        sample_metrics = {
            "baseline": algo,
            "total_rounds": 5,
            "rounds": [
                {
                    "round": r,
                    "global_test_accuracy": 0.20 + 0.05 * r + 0.02 * i,
                    "mean_local_test_accuracy": 0.50 + 0.06 * r + 0.03 * i,
                    "weighted_loss": 1.5 - 0.1 * r,
                    "duration_seconds": 12.0,
                }
                for r in range(1, 6)
            ],
        }
        with open(algo_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(sample_metrics, f)

    return reports_dir


def test_load_benchmark_metrics(mock_benchmark_dir: Path) -> None:
    data = load_benchmark_metrics(reports_dir=mock_benchmark_dir)
    assert len(data) == 5
    for algo in ["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph"]:
        assert algo in data
        assert len(data[algo]["rounds"]) == 5


def test_plot_benchmark_summary(mock_benchmark_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "plots"
    artifacts = plot_benchmark_summary(
        reports_dir=mock_benchmark_dir,
        output_dir=output_dir,
    )

    assert "global_accuracy_comparison" in artifacts
    assert "local_accuracy_comparison" in artifacts
    assert "benchmark_summary" in artifacts

    for path_str in artifacts.values():
        p = Path(path_str)
        assert p.is_file()
        assert p.stat().st_size > 0


def test_run_full_benchmark_synthetic(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark_plots"
    results = run_full_benchmark(
        rounds=1,
        clients=2,
        epochs=1,
        algorithms=["fedavg", "fedprox"],
        reports_dir=str(tmp_path / "reports"),
        output_dir=str(output_dir),
        synthetic=True,
    )

    assert "results" in results
    assert "plots" in results
    assert len(results["results"]) == 2
