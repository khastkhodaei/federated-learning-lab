import torch
from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.transport.client import FederatedClient
from src.utils.serialization import model_to_vector


def test_node_to_server_grpc_connection() -> None:
    # 1. Start live server on an ephemeral port
    server = FederatedServer(host="127.0.0.1", port=0)
    server.start()
    server_address = f"127.0.0.1:{server.bound_port}"

    try:
        # 2. Configure transport client pointing to the live server
        transport = FederatedClient(target=server_address)

        # 3. Create Node with transport attached
        node = Node(node_id="client_node_1", transport=transport)

        # 4. Test Registration over the wire
        reg_response = node.register()
        assert reg_response is not None
        assert reg_response.success is True
        assert "client_node_1" in server.servicer.registered_nodes

        # 5. Test Update submission over the wire
        dummy_weights = b"sample_model_tensor_bytes"
        update_response = node.send_update(weights=dummy_weights, num_samples=64)
        assert update_response is not None
        assert update_response.success is True

        # Verify update reached server servicer correctly
        assert "client_node_1" in server.servicer.updates
        received_update = server.servicer.updates["client_node_1"]
        assert received_update.num_samples == 64
        assert received_update.weights == dummy_weights

        transport.close()
    finally:
        # 6. Ensure server stops cleanly
        server.stop(grace=0)


def test_node_synchronizes_initial_model_from_server_over_grpc() -> None:
    """Verifies that at the start of Round 1, clients receive and unpack the

    server's seeded initial global model w_0 over live gRPC.
    """
    # 1. Server initialized with deterministic seed=42
    server = FederatedServer(host="127.0.0.1", port=0, seed=42)
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    # Extract the ground truth server initial parameter vector
    assert server.global_model is not None
    server_initial_vec = model_to_vector(server.global_model)

    # 2. Clients start with different random initializations
    client_model1 = SimpleCNN(seed=101)
    client_model2 = SimpleCNN(seed=202)

    # Confirm that clients initially have DIFFERENT weights from the server
    assert not torch.allclose(model_to_vector(client_model1), server_initial_vec)
    assert not torch.allclose(model_to_vector(client_model2), server_initial_vec)

    transport1 = FederatedClient(target=server_addr)
    transport2 = FederatedClient(target=server_addr)

    node1 = Node(node_id="client_1", model=client_model1, transport=transport1, port=0)
    node2 = Node(node_id="client_2", model=client_model2, transport=transport2, port=0)

    try:
        # 3. Register both nodes over gRPC
        node1.register()
        node2.register()
        assert len(server.servicer.registered_nodes) == 2

        # 4. Server triggers round 1 over gRPC, sending w_0 to both nodes
        responses = server.trigger_round(round_id=1, epochs=1)
        assert len(responses) == 2
        for r in responses:
            assert r.success is True

        # 5. Verify that both client models now have the EXACT same weights as server w_0
        node1_current_vec = model_to_vector(node1.model)
        node2_current_vec = model_to_vector(node2.model)

        assert torch.allclose(node1_current_vec, server_initial_vec)
        assert torch.allclose(node2_current_vec, server_initial_vec)
        assert torch.allclose(node1_current_vec, node2_current_vec)

    finally:
        node1.stop(grace=0)
        node2.stop(grace=0)
        transport1.close()
        transport2.close()
        server.stop(grace=0)
