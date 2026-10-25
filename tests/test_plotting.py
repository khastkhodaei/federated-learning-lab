import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.utils.plotting import (
    compute_client_class_distribution,
    plot_client_distribution,
)


def test_compute_client_class_distribution():
    # 2 clients, 3 classes, total 6 samples
    # Client 0: indices [0, 1, 2], labels: [0, 0, 1] -> class 0: 2, class 1: 1, class 2: 0
    # Client 1: indices [3, 4, 5], labels: [1, 2, 2] -> class 0: 0, class 1: 1, class 2: 2
    labels = np.array([0, 0, 1, 1, 2, 2])
    client_dict = {
        0: np.array([0, 1, 2]),
        1: np.array([3, 4, 5]),
    }

    counts = compute_client_class_distribution(client_dict=client_dict, labels=labels, num_classes=3)
    assert counts.shape == (2, 3)

    # Client 0 counts
    assert counts[0, 0] == 2
    assert counts[0, 1] == 1
    assert counts[0, 2] == 0

    # Client 1 counts
    assert counts[1, 0] == 0
    assert counts[1, 1] == 1
    assert counts[1, 2] == 2


def test_plot_client_distribution_saves_file():
    counts = np.array([
        [10, 20, 5],
        [15, 5, 15],
    ])
    class_names = ["Class A", "Class B", "Class C"]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_distribution.png"
        fig = plot_client_distribution(
            counts=counts,
            class_names=class_names,
            title="Test Distribution",
            output_path=str(output_file),
            show=False,
        )
        assert output_file.is_file()
        assert output_file.stat().st_size > 1000
        import matplotlib.pyplot as plt
        plt.close(fig)
