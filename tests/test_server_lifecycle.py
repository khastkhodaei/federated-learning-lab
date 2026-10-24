from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.transport.client import FederatedClient
from src.utils.serialization import model_to_bytes


def test_federated_server_lifecycle() -> None:
    # Use port 0 to allow OS to assign an available ephemeral port
    server = FederatedServer(host="127.0.0.1", port=0)
    assert server.bound_port > 0

    server.start()
    server.stop(grace=0)


def test_federated_server_trigger_round_empty() -> None:
    server = FederatedServer(host="127.0.0.1", port=0)
    server.start()
    try:
        results = server.trigger_round(round_id=1, weights=b"weights", epochs=1)
        assert results == []
    finally:
        server.stop(grace=0)


def test_federated_server_trigger_round_multiple_nodes() -> None:
    # 1. Start server on ephemeral port
    server = FederatedServer(host="127.0.0.1", port=0)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    # 2. Start two nodes with their own listeners on ephemeral ports
    transport1 = FederatedClient(target=server_addr)
    transport2 = FederatedClient(target=server_addr)

    node1 = Node(node_id="client_node_1", transport=transport1, port=0)
    node2 = Node(node_id="client_node_2", transport=transport2, port=0)

    try:
        # 3. Register both nodes
        res1 = node1.register()
        res2 = node2.register()
        assert res1 is not None and res1.success is True
        assert res2 is not None and res2.success is True
        assert len(server.servicer.registered_nodes) == 2

        # 4. Trigger round concurrently across both nodes
        round_results = server.trigger_round(
            round_id=1, weights=b"global_round_1_weights", epochs=2
        )

        assert len(round_results) == 2
        received_ids = {r.node_id for r in round_results}
        assert received_ids == {"client_node_1", "client_node_2"}
        for r in round_results:
            assert r.success is True
            assert r.weights == b"global_round_1_weights"
    finally:
        node1.stop(grace=0)
        node2.stop(grace=0)
        transport1.close()
        transport2.close()
        server.stop(grace=0)


def test_federated_server_initial_model_broadcast() -> None:
    initial_model = SimpleCNN()
    expected_initial_weights = model_to_bytes(initial_model)

    server = FederatedServer(host="127.0.0.1", port=0, model=initial_model)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    assert server.get_global_weights() == expected_initial_weights

    transport1 = FederatedClient(target=server_addr)
    transport2 = FederatedClient(target=server_addr)

    node1 = Node(node_id="cifar_client_1", transport=transport1, port=0)
    node2 = Node(node_id="cifar_client_2", transport=transport2, port=0)

    try:
        node1.register()
        node2.register()

        # Trigger round without passing explicit weights (server should use initial model)
        results = server.trigger_round(round_id=1, epochs=1)

        assert len(results) == 2
        # Verify both clients received the exact same initial weights w_0
        for r in results:
            assert r.weights == expected_initial_weights
    finally:
        node1.stop(grace=0)
        node2.stop(grace=0)
        transport1.close()
        transport2.close()
        server.stop(grace=0)
