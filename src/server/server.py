from concurrent import futures
from typing import Optional
import grpc

from src.server.service import FederatedServerServicer
from src.transport.federated_pb2_grpc import add_FederatedServiceServicer_to_server


class FederatedServer:
    """Manages the gRPC server lifecycle for the central federated coordinator."""

    def __init__(
        self,
        host: str = "[::]",
        port: int = 50051,
        servicer: Optional[FederatedServerServicer] = None,
        max_workers: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.servicer = servicer if servicer is not None else FederatedServerServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        add_FederatedServiceServicer_to_server(self.servicer, self.server)
        self.bound_port = self.server.add_insecure_port(f"{self.host}:{self.port}")

    def start(self) -> None:
        """Starts the gRPC server to accept incoming connections."""
        self.server.start()

    def stop(self, grace: int = 0) -> None:
        """Stops the gRPC server."""
        self.server.stop(grace)
