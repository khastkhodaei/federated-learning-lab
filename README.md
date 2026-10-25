# Federated Learning Lab (`federated-learning-lab`)

A modular and extensible playground for learning, experimenting with, and testing **Federated Learning (FL)** algorithms and decentralized architectures.

---

## 🎯 Objectives
- **Hands-on Experimentation**: Implement foundational FL aggregation strategies (e.g., FedAvg, FedProx).
- **Data Heterogeneity**: Simulate real-world Non-IID client data distributions (Dirichlet distribution, pathological non-IID).
- **Client Simulation**: Flexible simulation environment for multi-client participation, local training rounds, and model aggregation.
- **Evaluation & Benchmarking**: Track convergence speed, accuracy curves, and communication overhead.

---

## 📁 Repository Structure
```text
federated-learning-lab/
├── configs/            # Experiment and hyperparameter configurations
├── src/
│   ├── server/         # Server orchestration & aggregation algorithms (FedAvg, etc.)
│   ├── client/         # Client-side local training & state management
│   ├── models/         # Neural network model architectures (CNN, MLP, ResNet)
│   ├── datasets/       # Data loaders & IID / Non-IID partitioners
│   └── utils/          # Metrics calculation, logging, and plotting helpers
├── tests/              # Unit tests for components & aggregation algorithms
├── requirements.txt    # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository (if not already local)
git clone https://github.com/khastkhodaei/federated-learning-lab.git
cd federated-learning-lab

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### 2. Command Reference Guide
For the full guide of CLI commands, multi-terminal orchestration, and data visualizers, see [COMMANDS.md](COMMANDS.md).

`ash
# Launch server + client terminals with automatic distribution chart:
python launch_terminals.py

# Visualize client data distribution (stacked bar chart):
python plot_distribution.py
`
``

---

## 🔬 Roadmap
- [ ] Base server and client abstract classes
- [ ] Federated Averaging (`FedAvg`) baseline
- [ ] Non-IID Dirichlet data partitioner (MNIST / CIFAR-10)
- [ ] Federated Proximal (`FedProx`) for handling system heterogeneity
- [ ] Differential privacy and secure aggregation exploration
