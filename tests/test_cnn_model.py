import torch
from src.models.cnn import SimpleCNN


def test_simple_cnn_forward_cifar10() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    # CIFAR-10 input shape: (batch_size, channels, height, width) = (4, 3, 32, 32)
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_simple_cnn_backward_gradients() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([1, 7])

    criterion = torch.nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify gradients are computed
    assert model.conv1.weight.grad is not None
    assert model.conv2.weight.grad is not None
    assert model.fc1.weight.grad is not None
    assert model.fc_out.weight.grad is not None
