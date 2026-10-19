from src.client.node import Node
from src.server.server import FederatedServer
from src.transport.client import FederatedClient


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
