from src.server.server import FederatedServer


def test_federated_server_lifecycle() -> None:
    # Use port 0 to allow OS to assign an available ephemeral port
    server = FederatedServer(host="127.0.0.1", port=0)
    assert server.bound_port > 0

    server.start()
    server.stop(grace=0)
