from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, random_split

from fl_data_generator import dirichlet_partition, iid_partition
from src.utils.evaluation import get_global_test_loader


def test_client_train_val_split_strictly_disjoint() -> None:
    """Verifies that client train/val splits are strictly disjoint and conserve sample count."""
    total_samples = 100
    inputs = torch.randn(total_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (total_samples,))
    client_ds = TensorDataset(inputs, targets)

    for val_ratio in [0.1, 0.2, 0.25, 0.3]:
        val_len = max(1, int(len(client_ds) * val_ratio))
        train_len = len(client_ds) - val_len

        train_ds, val_ds = random_split(
            client_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        train_indices = set(train_ds.indices)
        val_indices = set(val_ds.indices)

        # 1. Strict disjointness (zero intersection)
        assert len(train_indices.intersection(val_indices)) == 0

        # 2. Complete conservation of samples
        assert len(train_ds) + len(val_ds) == len(client_ds)
        assert train_indices.union(val_indices) == set(range(total_samples))


def test_client_partitions_cover_data_exactly_once() -> None:
    """Verifies multi-client partitions are pairwise disjoint and cover all samples exactly once."""
    num_samples = 1000
    num_clients = 5
    rng = np.random.RandomState(42)
    labels = rng.randint(0, 10, size=num_samples)
    complete_universe = set(range(num_samples))

    # Test both IID and Dirichlet Non-IID partitioners
    partitioners = [
        ("iid", iid_partition(labels, num_clients=num_clients, seed=42)),
        (
            "dirichlet",
            dirichlet_partition(labels, num_clients=num_clients, alpha=0.5, seed=42),
        ),
    ]

    for name, partitions in partitioners:
        assert len(partitions) == num_clients, f"{name} must create {num_clients} client splits"

        # 1. Pairwise disjointness: indices_i & indices_j == set() for all i != j
        client_ids = list(partitions.keys())
        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                set_i = set(partitions[client_ids[i]])
                set_j = set(partitions[client_ids[j]])
                overlap = set_i.intersection(set_j)
                assert overlap == set(), (
                    f"{name}: Clients {client_ids[i]} and {client_ids[j]} share {len(overlap)} samples: {overlap}"
                )

        # 2. Total coverage without omission or duplication
        all_indices: set[int] = set()
        total_count = 0
        for cid, indices in partitions.items():
            idx_list = list(indices)
            assert len(idx_list) == len(set(idx_list)), f"{name}: Client {cid} contains duplicate indices"
            all_indices.update(idx_list)
            total_count += len(idx_list)

        assert total_count == num_samples, f"{name}: Total partitioned count {total_count} != {num_samples}"
        assert all_indices == complete_universe, f"{name}: Partition indices do not match complete universe"


def test_global_test_set_disjoint_from_client_partitions() -> None:
    """Verifies global test loaders exclusively use the held-out test split (train=False)."""
    # Verify that get_global_test_loader requests train=False from torchvision datasets
    with patch("src.utils.evaluation.datasets.CIFAR10") as mock_cifar10:
        mock_cifar10.return_value = TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,)))
        _ = get_global_test_loader(dataset_name="cifar10", data_dir="./data", batch_size=32, download=False)
        mock_cifar10.assert_called_once()
        _, kwargs = mock_cifar10.call_args
        assert kwargs.get("train") is False, "Global test loader must load train=False split"

    with patch("src.utils.evaluation.datasets.MNIST") as mock_mnist:
        mock_mnist.return_value = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))
        _ = get_global_test_loader(dataset_name="mnist", data_dir="./data", batch_size=32, download=False)
        mock_mnist.assert_called_once()
        _, kwargs = mock_mnist.call_args
        assert kwargs.get("train") is False, "Global test loader must load train=False split"


def test_partitioning_seed_determinism() -> None:
    """Verifies deterministic partitioning given identical seeds and variation with differing seeds."""
    num_samples = 500
    num_clients = 4
    rng = np.random.RandomState(123)
    labels = rng.randint(0, 10, size=num_samples)

    # 1. IID determinism
    iid_run1 = iid_partition(labels, num_clients=num_clients, seed=100)
    iid_run2 = iid_partition(labels, num_clients=num_clients, seed=100)
    iid_diff = iid_partition(labels, num_clients=num_clients, seed=200)

    for c in range(num_clients):
        np.testing.assert_array_equal(iid_run1[c], iid_run2[c])
    assert not np.array_equal(iid_run1[0], iid_diff[0])

    # 2. Dirichlet determinism
    dir_run1 = dirichlet_partition(labels, num_clients=num_clients, alpha=0.3, seed=100)
    dir_run2 = dirichlet_partition(labels, num_clients=num_clients, alpha=0.3, seed=100)
    dir_diff = dirichlet_partition(labels, num_clients=num_clients, alpha=0.3, seed=200)

    for c in range(num_clients):
        np.testing.assert_array_equal(dir_run1[c], dir_run2[c])
    assert not np.array_equal(dir_run1[0], dir_diff[0])

    # 3. Client train/val split determinism
    client_ds = TensorDataset(torch.randn(100, 2), torch.zeros(100))
    s1_train, s1_val = random_split(client_ds, [80, 20], generator=torch.Generator().manual_seed(99))
    s2_train, s2_val = random_split(client_ds, [80, 20], generator=torch.Generator().manual_seed(99))
    sd_train, sd_val = random_split(client_ds, [80, 20], generator=torch.Generator().manual_seed(101))

    assert s1_train.indices == s2_train.indices
    assert s1_val.indices == s2_val.indices
    assert s1_train.indices != sd_train.indices
