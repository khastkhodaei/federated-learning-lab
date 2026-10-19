from src.client.node import Node


def test_node_creation() -> None:
    node = Node(node_id="client_1", data=None, model=None)
    assert node.node_id == "client_1"
    assert node.data is None
    assert node.model is None
