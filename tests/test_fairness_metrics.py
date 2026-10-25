"""Unit tests for client accuracy dispersion and fairness metrics."""

from src.utils.evaluation import compute_fairness_metrics


def test_compute_fairness_metrics_basic() -> None:
    # 10 clients with accuracies from 0.1 to 1.0
    accs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metrics = compute_fairness_metrics(accs)

    assert round(metrics["mean"], 4) == 0.55
    assert metrics["worst_10"] <= 0.2
    assert metrics["best_10"] >= 0.9
    assert metrics["std"] > 0.0


def test_compute_fairness_metrics_empty() -> None:
    metrics = compute_fairness_metrics([])
    assert metrics["mean"] == 0.0
    assert metrics["std"] == 0.0
    assert metrics["worst_10"] == 0.0
    assert metrics["best_10"] == 0.0


def test_compute_fairness_metrics_single() -> None:
    metrics = compute_fairness_metrics([0.75])
    assert metrics["mean"] == 0.75
    assert metrics["std"] == 0.0
    assert metrics["worst_10"] == 0.75
    assert metrics["best_10"] == 0.75
