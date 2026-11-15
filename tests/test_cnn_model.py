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


def test_simple_cnn_extract_features_shape() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)
    features = model.extract_features(inputs)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (4, 128)


def test_simple_cnn_forward_with_return_features() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)

    # Calling with return_features=True returns (logits, features)
    outputs = model(inputs, return_features=True)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    logits, features = outputs

    assert logits.shape == (4, 10)
    assert features.shape == (4, 128)
    assert torch.allclose(logits, model.fc_out(features), atol=1e-6)

    # Calling with default return_features=False returns only logits (backward compatibility)
    default_logits = model(inputs)
    assert isinstance(default_logits, torch.Tensor)
    assert default_logits.shape == (4, 10)
    assert torch.allclose(default_logits, logits, atol=1e-6)


def test_simple_cnn_extract_features_gradient_flow() -> None:
    model = SimpleCNN(in_channels=3, num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)
    features = model.extract_features(inputs)

    features.sum().backward()

    # Backbone parameters receive gradients
    assert model.conv1.weight.grad is not None
    assert model.conv2.weight.grad is not None
    assert model.fc1.weight.grad is not None
    assert torch.count_nonzero(model.conv1.weight.grad) > 0
    assert torch.count_nonzero(model.conv2.weight.grad) > 0
    assert torch.count_nonzero(model.fc1.weight.grad) > 0

    # Classification head is not involved in feature extraction
    assert model.fc_out.weight.grad is None

