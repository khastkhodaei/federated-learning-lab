"""Stress and fault injection tests for gRPC communication and node lifecycle."""

import time
from typing import Any
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.fedavg.strategy import FedAvgServerStrategy
from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.transport.client import FederatedClient
from src.transport.federated_pb2 import TrainResponse
from src.utils.serialization import model_to_bytes, model_to_vector


def _create_synthetic_loader(num_samples: int = 16, batch_size: int = 8) -> DataLoader:
    """Generates a small synthetic DataLoader for node training."""
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def test_chaos_client_dropout_mid_round() -> None:
    """Verifies server tolerates sudden client disconnect mid-round and aggregates survivors."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"
    assert server.global_model is not None
    initial_weights = model_to_vector(server.global_model).clone()

    nodes: list[Node] = []
    transports: list[FederatedClient] = []
    try:
        for i in range(3):
            t = FederatedClient(target=server_addr)
            transports.append(t)
            n = Node(
                node_id=f"client_{i}",
                model=SimpleCNN(seed=i + 1),
                data=_create_synthetic_loader(8, 4),
                transport=t,
                port=0,
            )
            reg = n.register()
            assert reg and reg.success
            nodes.append(n)

        assert len(server.servicer.registered_nodes) == 3

        # Abruptly stop node 1 before round dispatch (simulating crash / network disconnect)
        nodes[1].stop(grace=0)
        transports[1].close()

        # Trigger round with aggregation enabled
        responses = server.trigger_round(round_id=1, epochs=1, timeout=5.0, aggregate=True)

        # Exactly 2 responses from surviving clients
        assert len(responses) == 2
        for r in responses:
            assert r.success is True
            assert r.node_id in ("client_0", "client_2")

        # Global model updated by surviving nodes
        updated_weights = model_to_vector(server.global_model)
        assert not torch.allclose(updated_weights, initial_weights)

    finally:
        for n in nodes:
            try:
                n.stop(grace=0)
            except Exception:
                pass
        for t in transports:
            try:
                t.close()
            except Exception:
                pass
        server.stop(grace=0)


def test_chaos_straggler_timeout_recovery() -> None:
    """Verifies that slow stragglers are cut off by timeout without blocking the round barrier."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    transports: list[FederatedClient] = []
    nodes: list[Node] = []
    try:
        # Fast node
        t0 = FederatedClient(target=server_addr)
        transports.append(t0)
        n0 = Node(
            node_id="fast_client",
            model=SimpleCNN(seed=1),
            data=_create_synthetic_loader(8, 4),
            transport=t0,
            port=0,
        )
        reg0 = n0.register()
        assert reg0 and reg0.success
        nodes.append(n0)

        # Straggler node with injected delay
        t1 = FederatedClient(target=server_addr)
        transports.append(t1)
        n1 = Node(
            node_id="straggler_client",
            model=SimpleCNN(seed=2),
            data=_create_synthetic_loader(8, 4),
            transport=t1,
            port=0,
        )
        reg1 = n1.register()
        assert reg1 and reg1.success
        nodes.append(n1)

        # Inject 1.5s sleep into n1's local training step
        original_train = n1.trainer.train

        def _delayed_train(*args: Any, **kwargs: Any) -> Any:
            time.sleep(1.5)
            return original_train(*args, **kwargs)

        n1.trainer.train = _delayed_train  # type: ignore[method-assign]

        # Trigger round with short timeout (0.5s)
        t_start = time.time()
        responses = server.trigger_round(round_id=1, epochs=1, timeout=0.5, aggregate=True)
        elapsed = time.time() - t_start

        # Verification: Completed without deadlocking or waiting for 1.5s
        assert elapsed < 1.3, f"Round timeout failed: took {elapsed:.2f}s"
        assert len(responses) == 1
        assert responses[0].node_id == "fast_client"
        assert responses[0].success is True

    finally:
        for n in nodes:
            try:
                n.stop(grace=0)
            except Exception:
                pass
        for t in transports:
            try:
                t.close()
            except Exception:
                pass
        server.stop(grace=0)


def test_chaos_corrupted_payload_graceful_rejection() -> None:
    """Verifies server.aggregate_responses rejects malformed/corrupted byte payloads."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    assert server.global_model is not None
    initial_weights = model_to_vector(server.global_model).clone()

    valid_payload = model_to_bytes(SimpleCNN(seed=10))

    # Construct responses with various malformed payloads
    responses = [
        TrainResponse(node_id="node_good", success=True, weights=valid_payload, num_samples=10),
        TrainResponse(node_id="node_empty", success=True, weights=b"", num_samples=10),
        TrainResponse(
            node_id="node_corrupt_unaligned",
            success=True,
            weights=b"corrupted_garbage_bytes",
            num_samples=10,
        ),
        TrainResponse(node_id="node_corrupt_size", success=True, weights=b"\x00" * 32, num_samples=10),
    ]

    # Aggregation must reject corrupted updates, aggregate valid ones, and update model
    result = server.aggregate_responses(responses, round_id=1)
    assert result is not None
    assert len(result) == len(valid_payload)

    updated_weights = model_to_vector(server.global_model)
    assert not torch.allclose(updated_weights, initial_weights)


def test_chaos_all_clients_drop_no_deadlock() -> None:
    """Verifies that total client disconnection does not hang the coordinator and returns safely."""
    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=FedAvgServerStrategy(),
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    transports: list[FederatedClient] = []
    nodes: list[Node] = []
    try:
        for i in range(2):
            t = FederatedClient(target=server_addr)
            transports.append(t)
            n = Node(
                node_id=f"client_{i}",
                model=SimpleCNN(seed=i + 1),
                data=_create_synthetic_loader(8, 4),
                transport=t,
                port=0,
            )
            reg = n.register()
            assert reg and reg.success
            nodes.append(n)

        # Terminate all nodes abruptly
        for n in nodes:
            n.stop(grace=0)
        for t in transports:
            t.close()

        # Trigger round: all RPCs fail, server should not deadlock or crash
        responses = server.trigger_round(round_id=1, epochs=1, timeout=1.0, aggregate=True)
        assert len(responses) == 0

        # Direct aggregation on empty responses returns None safely without exception
        agg = server.aggregate_responses(responses)
        assert agg is None

    finally:
        server.stop(grace=0)

