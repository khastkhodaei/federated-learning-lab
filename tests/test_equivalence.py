import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.baselines.fedavg.trainer import FedAvgClientTrainer
from src.models.cnn import SimpleCNN


def test_single_client_one_round_fedavg_equals_centralized() -> None:
    device = torch.device("cpu")
    torch.manual_seed(123)

    # 1. Deterministic synthetic dataset
    inputs = torch.randn(64, 3, 32, 32)
    targets = torch.randint(0, 10, (64,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    m_central = SimpleCNN(seed=42)
    m_fed = SimpleCNN(seed=42)

    # Verify bit-exact initialization
    for p_c, p_f in zip(m_central.parameters(), m_fed.parameters()):
        assert torch.equal(p_c, p_f)

    # 2. Centralized training for 1 epoch
    criterion = nn.CrossEntropyLoss()
    opt_central = torch.optim.SGD(m_central.parameters(), lr=0.01, momentum=0.9)
    m_central.train()
    central_total_samples = 0
    central_total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt_central.zero_grad()
        out = m_central(x)
        loss = criterion(out, y)
        loss.backward()
        opt_central.step()

        b_size = y.size(0)
        central_total_samples += b_size
        central_total_loss += loss.item() * b_size

    central_avg_loss = central_total_loss / central_total_samples

    # 3. Federated training: 1 client, 1 round
    client_model = SimpleCNN(seed=42)
    trainer = FedAvgClientTrainer()
    opt_fed = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)

    client_bytes, sample_count, metrics = trainer.train(
        model=client_model,
        dataloader=loader,
        epochs=1,
        criterion=criterion,
        optimizer=opt_fed,
        device=device,
    )

    # 4. Server aggregation with FedAvgServerStrategy
    strategy = FedAvgServerStrategy()
    strategy.aggregate(
        global_model=m_fed,
        client_updates=[(client_bytes, sample_count)],
        round_id=1,
    )

    # 5. Assert parameter bit-exact equality across all layers
    for (name_c, p_c), (name_f, p_f) in zip(
        m_central.named_parameters(), m_fed.named_parameters()
    ):
        assert name_c == name_f
        assert torch.allclose(p_f, p_c, atol=1e-6)
        assert torch.equal(p_f, p_c)

    # 6. Assert client trainer metric loss equals the centralized average loss
    assert pytest.approx(central_avg_loss, rel=1e-6) == metrics["loss"]


def test_single_client_multi_epoch_equivalence() -> None:
    device = torch.device("cpu")
    torch.manual_seed(999)

    inputs = torch.randn(32, 3, 32, 32)
    targets = torch.randint(0, 10, (32,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    m_central = SimpleCNN(seed=101)
    m_fed = SimpleCNN(seed=101)

    criterion = nn.CrossEntropyLoss()
    opt_central = torch.optim.SGD(m_central.parameters(), lr=0.02, momentum=0.85)

    # Centralized 3 epochs
    m_central.train()
    central_total_samples = 0
    central_total_loss = 0.0
    for _ in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_central.zero_grad()
            out = m_central(x)
            loss = criterion(out, y)
            loss.backward()
            opt_central.step()
            b_size = y.size(0)
            central_total_samples += b_size
            central_total_loss += loss.item() * b_size

    central_avg_loss = central_total_loss / central_total_samples

    # Federated 1 client, 3 local epochs
    client_model = SimpleCNN(seed=101)
    trainer = FedAvgClientTrainer()
    opt_fed = torch.optim.SGD(client_model.parameters(), lr=0.02, momentum=0.85)

    client_bytes, sample_count, metrics = trainer.train(
        model=client_model,
        dataloader=loader,
        epochs=3,
        criterion=criterion,
        optimizer=opt_fed,
        device=device,
    )

    strategy = FedAvgServerStrategy()
    strategy.aggregate(
        global_model=m_fed,
        client_updates=[(client_bytes, sample_count)],
        round_id=1,
    )

    for (name_c, p_c), (name_f, p_f) in zip(
        m_central.named_parameters(), m_fed.named_parameters()
    ):
        assert name_c == name_f
        assert torch.equal(p_f, p_c)

    assert pytest.approx(central_avg_loss, rel=1e-6) == metrics["loss"]


def test_multiclient_full_batch_single_step_equals_pooled_gradient() -> None:
    device = torch.device("cpu")
    torch.manual_seed(777)

    # 1. Pooled dataset: 100 CIFAR-10 shape samples
    total_samples = 100
    inputs = torch.randn(total_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (total_samples,))
    pooled_dataset = TensorDataset(inputs, targets)

    # 2. Partition into 3 clients with heterogeneous sizes: 20, 30, 50
    client_sizes = [20, 30, 50]
    subsets = torch.utils.data.random_split(
        pooled_dataset,
        client_sizes,
        generator=torch.Generator().manual_seed(42),
    )

    # 3. Models initialized with identical weights
    m_pooled = SimpleCNN(seed=77)
    m_fed = SimpleCNN(seed=77)

    for p_p, p_f in zip(m_pooled.parameters(), m_fed.parameters()):
        assert torch.equal(p_p, p_f)

    # 4. Centralized single full-batch step on pooled dataset (batch_size=100)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    opt_pooled = torch.optim.SGD(
        m_pooled.parameters(), lr=lr, momentum=0.0, weight_decay=0.0
    )

    combined_dataset = torch.utils.data.ConcatDataset(subsets)
    pooled_loader = DataLoader(
        combined_dataset,
        batch_size=total_samples,
        shuffle=False,
    )
    m_pooled.train()
    opt_pooled.zero_grad()
    for x_all, y_all in pooled_loader:
        x_all, y_all = x_all.to(device), y_all.to(device)
        out_all = m_pooled(x_all)
        loss_all = criterion(out_all, y_all)
        loss_all.backward()
    opt_pooled.step()

    # 5. Federated training: 3 clients, each training local replica for 1 full-batch epoch
    client_updates = []
    trainer = FedAvgClientTrainer()

    for idx, subset in enumerate(subsets):
        client_model = SimpleCNN(seed=77)
        client_opt = torch.optim.SGD(
            client_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0
        )
        client_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

        client_bytes, sample_count, _ = trainer.train(
            model=client_model,
            dataloader=client_loader,
            epochs=1,
            criterion=criterion,
            optimizer=client_opt,
            device=device,
        )
        assert sample_count == client_sizes[idx]
        client_updates.append((client_bytes, sample_count))

    # 6. Server aggregation via FedAvgServerStrategy
    strategy = FedAvgServerStrategy()
    strategy.aggregate(
        global_model=m_fed,
        client_updates=client_updates,
        round_id=1,
    )

    # 7. Assert parameter matching within numerical tolerance across all layers
    for (name_p, p_p), (name_f, p_f) in zip(
        m_pooled.named_parameters(), m_fed.named_parameters()
    ):
        assert name_p == name_f
        assert torch.allclose(p_f, p_p, atol=1e-5), (
            f"Layer {name_p} failed tolerance: max diff {torch.max(torch.abs(p_f - p_p))}"
        )

