"""Data distribution visualization utilities for Federated Learning.

Provides methods to compute per-client class counts and render stacked bar charts
depicting quantity skew (bar heights) and label skew (class color proportions).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure


# Standard human-readable class names for popular benchmark datasets
DATASET_CLASS_NAMES: Dict[str, List[str]] = {
    "cifar10": [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck",
    ],
    "mnist": [str(i) for i in range(10)],
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "fashionmnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
}


def compute_client_class_distribution(
    fed_dataset: Optional[Any] = None,
    client_dict: Optional[Dict[int, np.ndarray]] = None,
    labels: Optional[Union[np.ndarray, List[int]]] = None,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Computes a matrix of sample counts per client and per class.

    Args:
        fed_dataset: An initialized FederatedDataset instance.
        client_dict: Mapping of client_id -> array of sample indices.
        labels: 1D array of sample target labels.
        num_classes: Number of distinct classes (inferred if None).

    Returns:
        2D numpy array of shape (num_clients, num_classes) where cell [i, c]
        is the number of samples belonging to class c held by client i.
    """
    if fed_dataset is not None:
        c_dict = fed_dataset.client_dict
        lbls = np.asarray(fed_dataset.labels)
    elif client_dict is not None and labels is not None:
        c_dict = client_dict
        lbls = np.asarray(labels)
    else:
        raise ValueError("Either fed_dataset or both client_dict and labels must be provided.")

    num_clients = len(c_dict)
    if num_classes is None:
        num_classes = int(np.max(lbls)) + 1 if len(lbls) > 0 else 0

    counts = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in c_dict.items():
        if len(indices) == 0:
            continue
        client_labels = lbls[indices]
        unique, class_counts = np.unique(client_labels, return_counts=True)
        for u, count in zip(unique, class_counts):
            if u < num_classes:
                counts[client_id, int(u)] = count

    return counts


def plot_client_distribution(
    counts: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
) -> matplotlib.figure.Figure:
    """Renders a stacked bar chart of client sample counts and class distributions.

    Args:
        counts: 2D array of shape (num_clients, num_classes).
        class_names: Optional list of names for each class.
        title: Plot title.
        output_path: File path to save the generated figure.
        show: If True, calls plt.show() interactively.
        figsize: Optional figure size tuple (width, height).

    Returns:
        The generated matplotlib Figure.
    """
    num_clients, num_classes = counts.shape
    client_ids = [f"Client {i}" for i in range(num_clients)]

    if class_names is None or len(class_names) < num_classes:
        class_names = [f"Class {c}" for c in range(num_classes)]

    # Colormap selection
    if num_classes <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(num_classes)]

    fig_w = max(10, num_clients * 0.9)
    fig_size = figsize or (fig_w, 6.5)

    fig, ax = plt.subplots(figsize=fig_size)

    # Compute stacked bar positions
    bottom = np.zeros(num_clients, dtype=int)
    for c in range(num_classes):
        values = counts[:, c]
        ax.bar(
            client_ids,
            values,
            bottom=bottom,
            label=class_names[c],
            color=colors[c],
            edgecolor="white",
            linewidth=0.5,
            width=0.65,
        )
        bottom += values

    # Add sample count text on top of each bar
    max_height = np.max(bottom) if len(bottom) > 0 else 100
    for i, total in enumerate(bottom):
        ax.text(
            i,
            total + max_height * 0.015,
            f"{total:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2d3748",
        )

    ax.set_xlabel("Client Identifier", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Total Number of Samples", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylim(0, max_height * 1.12)

    chart_title = title or "Federated Client Data Distribution (Quantity & Class Skew)"
    ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=15)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e0")
    ax.spines["bottom"].set_color("#cbd5e0")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
    ax.set_axisbelow(True)

    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Legend placement outside the plot area
    ax.legend(
        title="Classes",
        title_fontsize="11",
        fontsize="9",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#e2e8f0",
    )

    plt.tight_layout()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
