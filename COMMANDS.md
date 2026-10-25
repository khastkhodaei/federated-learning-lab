# Federated Learning Lab - Command Reference Guide (`COMMANDS.md`)

This guide provides a centralized reference for running, orchestrating, and visualizing federated learning experiments in this repository.

---

## 1. Automated Multi-Terminal Deployment (Recommended)

Spawns the central coordinator server and all client nodes into their own live terminal windows, and automatically generates the dataset distribution chart.

```powershell
# Launch server and clients using settings from configs/server.yaml and configs/client.yaml
python launch_terminals.py

# PowerShell shortcut equivalent
.\launch_terminals.ps1
```

### Useful CLI Overrides
```powershell
# Override number of clients (e.g., 5 clients)
python launch_terminals.py --clients 5

# Override number of training rounds and server port
python launch_terminals.py --rounds 5 --port 50055

# Preview generated commands without opening terminal windows
python launch_terminals.py --dry-run

# Launch terminals without regenerating the data distribution plot
python launch_terminals.py --no-plot
```

---

## 2. Dataset Distribution Visualizer

Generates a stacked bar chart displaying sample volumes (**Quantity Skew**) and color-coded class proportions (**Label Skew**) across all participating clients.

```powershell
# Generate and save chart to reports/data_distribution.png (reads configs/server.yaml)
python plot_distribution.py

# Display chart in an interactive pop-up window
python plot_distribution.py --interactive

# Visualize different benchmark datasets or partitioning strategies
python plot_distribution.py --dataset mnist --partition dirichlet --alpha 0.1
python plot_distribution.py --dataset cifar10 --partition pathological
python plot_distribution.py --dataset fashion_mnist --partition iid

# Save to a custom output path
python plot_distribution.py --output figures/cifar10_dirichlet_skew.png
```

---

## 3. Manual Terminal Execution (Step-by-Step)

If you prefer to start each component in individual command line tabs or on separate machines over a network:

### Step 1: Start Central Server (Terminal 1)
```powershell
python run_server.py --config configs/server.yaml
```
*Options:*
- `--port 50051`: Listening port
- `--clients 10`: Number of registered clients required before triggering rounds
- `--rounds 3`: Number of training rounds
- `--epochs 1`: Local client epochs per round
- `--baseline fedavg`: Aggregation baseline strategy (e.g. `fedavg`, `fedprox`)

### Step 2: Start Client Nodes (Terminals 2, 3, ...)
```powershell
# Start Client 0 (supports --baseline fedavg)
python run_client.py --id 0 --server 127.0.0.1:50051 --baseline fedavg

# Start Client 1
python run_client.py --id 1 --server 127.0.0.1:50051

# Start Client N
python run_client.py --id <N> --server 127.0.0.1:50051
```

---

## 4. Single-Process Simulation Orchestrator

Runs the server and all client nodes asynchronously inside a single terminal process (useful for automated benchmarks and CI pipelines):

```powershell
# Standard simulation using settings from configs/
python run_simulation.py --clients 10 --rounds 3 --epochs 1

# Fast verification using synthetic in-memory tensors (skips dataset download)
python run_simulation.py --synthetic --clients 3 --rounds 2
```

---

## 5. Configuration Files

All runtime behaviors are config-driven via YAML:

- **`configs/server.yaml`**: Server IP, port, required client count, and rounds:
  ```yaml
  host: "127.0.0.1"
  port: 50051
  num_clients: 10
  baseline: "fedavg"
  rounds: 3
  epochs_per_round: 1
  ```
- **`configs/client.yaml`**: Dataset partition type, Dirichlet $lpha$, learning rate, batch size, and optimizer:
  ```yaml
  server_address: "127.0.0.1:50051"
  dataset:
    name: "cifar10"
    partition: "dirichlet"
    alpha: 0.5
    batch_size: 32
  ```

---

## 6. Testing & Quality Assurance

Run the test suite to verify gRPC communication, FedAvg aggregation, and launcher scripts:

```powershell
# Run full test suite
pytest

# Run tests quietly with summary
pytest -q

# Test individual components
pytest tests/test_launch_terminals.py
pytest tests/test_plotting.py
pytest tests/test_grpc_integration.py
pytest tests/test_aggregation.py
pytest tests/test_baselines_fedavg.py
```


---

## 7. Federated Learning Baselines Architecture

All federated learning algorithms implement narrow Strategy seams defined in `src/baselines/base.py`:
- **`ServerStrategy`**: Interface for coordinating server aggregation (`aggregate(global_model, client_updates, round_id, **kwargs)`).
- **`ClientTrainer`**: Interface for local client training (`train(model, dataloader, epochs, criterion, optimizer, device, global_weights, **kwargs)`).
- **Registry**: `src/baselines/` exports `get_baseline("fedavg")` to instantiate strategy and trainer pairs dynamically based on CLI or configuration.
