import subprocess
import sys


def compile_proto() -> None:
    """Compiles federated.proto into Python gRPC stubs."""
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I.",
        "--python_out=.",
        "--grpc_python_out=.",
        "src/transport/federated.proto",
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    compile_proto()
