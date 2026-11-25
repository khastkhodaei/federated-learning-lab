import json
from pathlib import Path
import pytest

from src.utils.benchmark_plotting import (
    ALGORITHM_STYLES,
    load_benchmark_metrics,
    plot_benchmark_summary,
)
from run_benchmark import (
    BENCHMARK_CONFIGS,
    parse_seeds,
    run_full_benchmark,
    summarize_multi_seed_runs,
)


@pytest.fixture
def mock_benchmark_dir(tmp_path: Path) -> Path:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    algos = ["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph", "dfedpgp", "dfpl"]
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
    assert len(data) == 7
    for algo in ["fedavg", "fedprox", "ditto", "perfedavg", "pfedgraph", "dfedpgp", "dfpl"]:
        assert algo in data
        assert len(data[algo]["rounds"]) == 5


def test_plot_benchmark_summary(mock_benchmark_dir: Path, tmp_path: Path) -> None:
    assert "dfedpgp" in ALGORITHM_STYLES
    assert ALGORITHM_STYLES["dfedpgp"]["label"] == "DFedPGP (Directed Push)"
    assert "dfpl" in ALGORITHM_STYLES
    assert ALGORITHM_STYLES["dfpl"]["label"] == "DFPL (Prototype Alignment)"

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
    assert "dfedpgp" in BENCHMARK_CONFIGS
    assert BENCHMARK_CONFIGS["dfedpgp"]["params"] == {"out_degree": 1, "time_varying": True}
    assert "dfpl" in BENCHMARK_CONFIGS
    assert BENCHMARK_CONFIGS["dfpl"]["params"] == {"lambda_proto": 0.1}

    output_dir = tmp_path / "benchmark_plots"
    results = run_full_benchmark(
        rounds=1,
        clients=2,
        epochs=1,
        algorithms=["fedavg", "dfedpgp"],
        reports_dir=str(tmp_path / "reports"),
        output_dir=str(output_dir),
        synthetic=True,
    )

    assert "results" in results
    assert "plots" in results
    assert set(results["results"].keys()) == {"fedavg", "dfedpgp"}
    assert (tmp_path / "reports" / "dfedpgp" / "metrics.json").is_file()


def test_multi_seed_mean_std_aggregation_and_cli_parsing(tmp_path: Path) -> None:
    """Verifies multi-seed Mean/Std summary aggregation and end-to-end --seeds benchmark mode."""
    assert parse_seeds("42,43,44") == [42, 43, 44]
    assert parse_seeds(["42, 43", "44"]) == [42, 43, 44]
    assert parse_seeds(None) is None

    seed_histories = [
        [
            {
                "global_test_accuracy": 0.40,
                "global_on_local_accuracy": 0.50,
                "local_test_accuracy": 0.70,
                "majority_class_accuracy": 0.60,
            },
            {
                "global_test_accuracy": 0.60,
                "global_on_local_accuracy": 0.70,
                "local_test_accuracy": 0.90,
                "majority_class_accuracy": 0.60,
            },
        ],
        [
            {
                "global_test_accuracy": 0.50,
                "global_on_local_accuracy": 0.60,
                "local_test_accuracy": 0.80,
                "majority_class_accuracy": 0.60,
            },
            {
                "global_test_accuracy": 0.70,
                "global_on_local_accuracy": 0.80,
                "local_test_accuracy": 0.90,
                "majority_class_accuracy": 0.60,
            },
        ],
    ]
    summary = summarize_multi_seed_runs(seed_histories, last_n=10)
    assert summary["num_seeds"] == 2
    assert summary["global_test_accuracy_mean"] == pytest.approx(65.0)
    assert summary["global_test_accuracy_std"] == pytest.approx(5.0)
    assert summary["last_10_global_accuracy_mean"] == pytest.approx(55.0)
    assert summary["last_10_global_accuracy_std"] == pytest.approx(5.0)
    assert summary["global_on_local_accuracy_mean"] == pytest.approx(75.0)
    assert summary["local_test_accuracy_mean"] == pytest.approx(90.0)
    assert summary["local_test_accuracy_std"] == pytest.approx(0.0)
    assert summary["majority_class_accuracy_mean"] == pytest.approx(60.0)

    # End-to-end 2-seed synthetic benchmark execution
    output_dir = tmp_path / "multi_plots"
    bench = run_full_benchmark(
        rounds=2,
        clients=2,
        epochs=1,
        algorithms=["fedavg", "local_only"],
        reports_dir=str(tmp_path / "multi_reports"),
        output_dir=str(output_dir),
        synthetic=True,
        device="cpu",
        seeds=[42, 43],
        partition_seed=42,
    )
    assert bench["seeds"] == [42, 43]
    assert (tmp_path / "multi_reports" / "seed_42" / "fedavg" / "metrics.json").is_file()
    assert (tmp_path / "multi_reports" / "seed_43" / "local_only" / "metrics.json").is_file()

    summary_json = output_dir / "benchmark_summary.json"
    assert summary_json.is_file()
    with open(summary_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for algo in ("fedavg", "local_only"):
        assert "last_10_global_accuracy" in payload[algo]
        assert "global_test_accuracy_mean" in payload[algo]
        assert "global_test_accuracy_std" in payload[algo]
        assert "global_on_local_accuracy_mean" in payload[algo]
        assert "local_test_accuracy_mean" in payload[algo]
        assert "majority_class_accuracy_mean" in payload[algo]
