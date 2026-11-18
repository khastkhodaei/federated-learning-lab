"""Stateful baseline resilience tests for permutation invariance, dropout/rejoin, and concurrency."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.base import ClientTrainer, ServerStrategy
from src.baselines.dfedpgp import DFedPGPClientTrainer, DFedPGPServerStrategy
from src.baselines.dfpl import DFPLClientTrainer, DFPLServerStrategy
from src.baselines.pfedgraph import PFedGraphClientTrainer, PFedGraphServerStrategy
from src.client.node import Node
from src.models.cnn import SimpleCNN
from src.server.server import FederatedServer
from src.transport.client import FederatedClient
from src.transport.federated_pb2 import TrainResponse
from src.utils.serialization import bytes_to_vector, model_to_bytes, model_to_vector


def _create_synthetic_loader(
    num_samples: int = 8,
    batch_size: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Generates a deterministic small synthetic DataLoader for client training."""
    gen = torch.Generator().manual_seed(seed)
    inputs = torch.randn(num_samples, 3, 32, 32, generator=gen)
    targets = torch.randint(0, 10, (num_samples,), generator=gen)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def test_stateful_strategies_permutation_invariance() -> None:
    """Verifies stateful strategies and server aggregation are invariant to client response order."""
    client_ids = ["client_0", "client_1", "client_2", "client_3"]
    perm_idx = [2, 0, 3, 1]
    perm_ids = [client_ids[i] for i in perm_idx]

    weights = [model_to_bytes(SimpleCNN(seed=10 + i)) for i in range(4)]
    samples = [16, 32, 24, 40]
    updates = list(zip(weights, samples, strict=True))
    perm_updates = [updates[i] for i in perm_idx]

    # 1. PFedGraphServerStrategy permutation invariance
    g_pfg_a = SimpleCNN(seed=42)
    g_pfg_b = SimpleCNN(seed=42)
    s_pfg_a = PFedGraphServerStrategy(alpha=0.2, lam=0.05)
    s_pfg_b = PFedGraphServerStrategy(alpha=0.2, lam=0.05)

    res_pfg_a = s_pfg_a.aggregate(g_pfg_a, updates, round_id=1, client_ids=client_ids)
    res_pfg_b = s_pfg_b.aggregate(g_pfg_b, perm_updates, round_id=1, client_ids=perm_ids)

    assert s_pfg_a.client_ids == s_pfg_b.client_ids == client_ids
    assert bool(res_pfg_a == res_pfg_b), "PFedGraph global weights differ across permutations"
    assert s_pfg_a.adjacency_matrix is not None and s_pfg_b.adjacency_matrix is not None
    assert torch.equal(s_pfg_a.adjacency_matrix, s_pfg_b.adjacency_matrix)
    for cid in client_ids:
        assert bool(s_pfg_a.client_weights[cid] == s_pfg_b.client_weights[cid]), (
            f"PFedGraph client_weights[{cid}] differs across permutations"
        )

    # 2. DFedPGPServerStrategy permutation invariance
    g_pgp_a = SimpleCNN(seed=42)
    g_pgp_b = SimpleCNN(seed=42)
    s_pgp_a = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)
    s_pgp_b = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)

    res_pgp_a = s_pgp_a.aggregate(g_pgp_a, updates, round_id=1, client_ids=client_ids)
    res_pgp_b = s_pgp_b.aggregate(g_pgp_b, perm_updates, round_id=1, client_ids=perm_ids)

    assert s_pgp_a.client_ids == s_pgp_b.client_ids == client_ids
    assert bool(res_pgp_a == res_pgp_b), "DFedPGP global weights differ across permutations"
    assert s_pgp_a.mixing_matrix is not None and s_pgp_b.mixing_matrix is not None
    assert torch.equal(s_pgp_a.mixing_matrix, s_pgp_b.mixing_matrix)
    assert s_pgp_a.push_sum_weights is not None and s_pgp_b.push_sum_weights is not None
    assert torch.equal(s_pgp_a.push_sum_weights, s_pgp_b.push_sum_weights)
    for cid in client_ids:
        assert bool(s_pgp_a.client_weights[cid] == s_pgp_b.client_weights[cid]), (
            f"DFedPGP client_weights[{cid}] differs across permutations"
        )

    # 3. DFPLServerStrategy permutation invariance (weights + prototypes, both aggregate_weights modes)
    gen = torch.Generator().manual_seed(99)
    protos = [
        {0: torch.randn(128, generator=gen), 1: torch.randn(128, generator=gen)},
        {1: torch.randn(128, generator=gen), 2: torch.randn(128, generator=gen)},
        {0: torch.randn(128, generator=gen), 2: torch.randn(128, generator=gen)},
        {0: torch.randn(128, generator=gen), 3: torch.randn(128, generator=gen)},
    ]
    proto_counts = [
        {0: 10, 1: 6},
        {1: 12, 2: 20},
        {0: 8, 2: 16},
        {0: 15, 3: 25},
    ]
    perm_protos = [protos[i] for i in perm_idx]
    perm_counts = [proto_counts[i] for i in perm_idx]

    for agg_weights in (True, False):
        g_dfpl_a = SimpleCNN(seed=42)
        g_dfpl_b = SimpleCNN(seed=42)
        s_dfpl_a = DFPLServerStrategy(normalize_prototypes=True, aggregate_weights=agg_weights)
        s_dfpl_b = DFPLServerStrategy(normalize_prototypes=True, aggregate_weights=agg_weights)

        res_dfpl_a = s_dfpl_a.aggregate(
            g_dfpl_a,
            updates,
            round_id=1,
            client_ids=client_ids,
            client_prototypes=protos,
            client_sample_counts=proto_counts,
        )
        res_dfpl_b = s_dfpl_b.aggregate(
            g_dfpl_b,
            perm_updates,
            round_id=1,
            client_ids=perm_ids,
            client_prototypes=perm_protos,
            client_sample_counts=perm_counts,
        )

        assert bool(res_dfpl_a == res_dfpl_b), (
            f"DFPL global weights differ across permutations (aggregate_weights={agg_weights})"
        )
        assert bool(s_dfpl_a.global_prototypes_bytes == s_dfpl_b.global_prototypes_bytes), (
            "DFPL prototype bytes differ across permutations"
        )
        assert set(s_dfpl_a.global_prototypes.keys()) == set(s_dfpl_b.global_prototypes.keys())
        for c in s_dfpl_a.global_prototypes:
            assert torch.equal(s_dfpl_a.global_prototypes[c], s_dfpl_b.global_prototypes[c])

    # 4. FederatedServer.aggregate_responses deterministic ordering across permuted TrainResponses
    srv_a = FederatedServer(host="127.0.0.1", port=0, model=SimpleCNN(seed=42))
    srv_b = FederatedServer(host="127.0.0.1", port=0, model=SimpleCNN(seed=42))
    try:
        responses = [
            TrainResponse(node_id=cid, success=True, weights=w, num_samples=n)
            for cid, (w, n) in zip(client_ids, updates, strict=True)
        ]
        perm_responses = [responses[i] for i in perm_idx]

        agg_a = srv_a.aggregate_responses(responses, round_id=1)
        agg_b = srv_b.aggregate_responses(perm_responses, round_id=1)
        assert agg_a is not None and agg_b is not None
        assert bool(agg_a == agg_b), "Server aggregate_responses differs across response permutations"
    finally:
        srv_a.stop(grace=0)
        srv_b.stop(grace=0)


def test_stateful_strategies_dropout_and_rejoin_across_rounds() -> None:
    """Verifies stateful strategies recover cleanly across dropout, participant swap, and rejoin."""
    all_ids = ["client_0", "client_1", "client_2", "client_3"]
    all_updates = {
        cid: (model_to_bytes(SimpleCNN(seed=100 + idx)), 20 + idx * 5)
        for idx, cid in enumerate(all_ids)
    }

    round_cohorts = [
        ["client_0", "client_1", "client_2", "client_3"],  # Round 1: full cohort (K=4)
        ["client_0", "client_2", "client_3"],  # Round 2: client_1 dropped (K=3)
        ["client_0", "client_1", "client_3"],  # Round 3: swap participant set at same K=3
        ["client_0", "client_1", "client_2", "client_3"],  # Round 4: full rejoin (K=4)
    ]

    # 1. PFedGraphServerStrategy across cohort changes
    pfg_model = SimpleCNN(seed=42)
    pfg_strat = PFedGraphServerStrategy(alpha=0.2, lam=0.05)
    for r_id, cohort in enumerate(round_cohorts, start=1):
        updates = [all_updates[cid] for cid in cohort]
        agg = pfg_strat.aggregate(pfg_model, updates, round_id=r_id, client_ids=cohort)
        vec = bytes_to_vector(agg)
        assert torch.isfinite(vec).all()
        assert pfg_strat.client_ids == sorted(cohort)
        assert pfg_strat.adjacency_matrix is not None
        assert pfg_strat.adjacency_matrix.shape == (len(cohort), len(cohort))
        assert torch.allclose(
            pfg_strat.adjacency_matrix.sum(dim=-1),
            torch.ones(len(cohort)),
            atol=1e-5,
        )
        assert set(pfg_strat.client_weights.keys()) == set(cohort)

    # Specifically verify Round 3 participant swap (same K=3, different client IDs)
    # does not reuse stale adjacency weights from Round 2's different client set.
    pfg_r2_r3 = PFedGraphServerStrategy(alpha=0.2, lam=0.05)
    pfg_r2_r3.aggregate(
        SimpleCNN(seed=42),
        [all_updates[c] for c in round_cohorts[1]],
        round_id=2,
        client_ids=round_cohorts[1],
    )
    pfg_r2_r3.aggregate(
        SimpleCNN(seed=42),
        [all_updates[c] for c in round_cohorts[2]],
        round_id=3,
        client_ids=round_cohorts[2],
    )
    pfg_fresh_r3 = PFedGraphServerStrategy(alpha=0.2, lam=0.05)
    pfg_fresh_r3.aggregate(
        SimpleCNN(seed=42),
        [all_updates[c] for c in round_cohorts[2]],
        round_id=3,
        client_ids=round_cohorts[2],
    )
    assert pfg_r2_r3.adjacency_matrix is not None and pfg_fresh_r3.adjacency_matrix is not None
    assert torch.allclose(pfg_r2_r3.adjacency_matrix, pfg_fresh_r3.adjacency_matrix, atol=1e-6)

    # 2. DFedPGPServerStrategy across cohort changes
    pgp_model = SimpleCNN(seed=42)
    pgp_strat = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)
    for r_id, cohort in enumerate(round_cohorts, start=1):
        updates = [all_updates[cid] for cid in cohort]
        agg = pgp_strat.aggregate(pgp_model, updates, round_id=r_id, client_ids=cohort)
        vec = bytes_to_vector(agg)
        assert torch.isfinite(vec).all()
        assert pgp_strat.client_ids == sorted(cohort)
        assert pgp_strat.mixing_matrix is not None
        assert pgp_strat.mixing_matrix.shape == (len(cohort), len(cohort))
        assert torch.allclose(
            pgp_strat.mixing_matrix.sum(dim=0),
            torch.ones(len(cohort)),
            atol=1e-5,
        )
        assert pgp_strat.push_sum_weights is not None
        assert pgp_strat.push_sum_weights.shape == (len(cohort),)
        assert set(pgp_strat.client_weights.keys()) == set(cohort)

    # 3. DFPLServerStrategy across cohort changes with disjoint class prototypes
    gen = torch.Generator().manual_seed(77)
    proto_map = {
        "client_0": ({0: torch.randn(128, generator=gen)}, {0: 10}),
        "client_1": ({1: torch.randn(128, generator=gen)}, {1: 15}),
        "client_2": ({2: torch.randn(128, generator=gen)}, {2: 12}),
        "client_3": ({0: torch.randn(128, generator=gen), 3: torch.randn(128, generator=gen)}, {0: 8, 3: 20}),
    }
    dfpl_model = SimpleCNN(seed=42)
    dfpl_strat = DFPLServerStrategy(normalize_prototypes=True, aggregate_weights=True)
    for r_id, cohort in enumerate(round_cohorts, start=1):
        updates = [all_updates[cid] for cid in cohort]
        protos = [proto_map[cid][0] for cid in cohort]
        counts = [proto_map[cid][1] for cid in cohort]
        agg = dfpl_strat.aggregate(
            dfpl_model,
            updates,
            round_id=r_id,
            client_ids=cohort,
            client_prototypes=protos,
            client_sample_counts=counts,
        )
        vec = bytes_to_vector(agg)
        assert torch.isfinite(vec).all()
        expected_classes = {c for p in protos for c in p}
        assert set(dfpl_strat.global_prototypes.keys()) == expected_classes
        for p_vec in dfpl_strat.global_prototypes.values():
            assert torch.isfinite(p_vec).all()
            assert torch.allclose(torch.norm(p_vec, p=2), torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize("algorithm", ["pfedgraph", "dfedpgp", "dfpl"])
def test_concurrent_multi_client_grpc_rounds_lifecycle(algorithm: str) -> None:
    """Spawns 6 live gRPC nodes for 3 rounds with mid-run dropout and rejoin recovery."""
    strategy: ServerStrategy
    if algorithm == "pfedgraph":
        strategy = PFedGraphServerStrategy(alpha=0.1, lam=0.01)
    elif algorithm == "dfedpgp":
        strategy = DFedPGPServerStrategy(out_degree=1, seed=42, time_varying=True)
    else:
        strategy = DFPLServerStrategy(normalize_prototypes=True, aggregate_weights=True)

    def _make_trainer() -> ClientTrainer:
        if algorithm == "pfedgraph":
            return PFedGraphClientTrainer()
        if algorithm == "dfedpgp":
            return DFedPGPClientTrainer(head_epochs=1, backbone_epochs=1)
        return DFPLClientTrainer(lambda_proto=0.1, normalize_prototypes=True)

    server = FederatedServer(
        host="127.0.0.1",
        port=0,
        model=SimpleCNN(seed=42),
        strategy=strategy,
        max_workers=10,
    )
    server.start()
    server_addr = f"127.0.0.1:{server.bound_port}"

    num_clients = 6
    nodes: dict[str, Node] = {}
    transports: dict[str, FederatedClient] = {}

    def _spawn_and_register(idx: int) -> None:
        cid = f"client_{idx}"
        t = FederatedClient(target=server_addr)
        transports[cid] = t
        n = Node(
            node_id=cid,
            model=SimpleCNN(seed=idx + 1),
            data=_create_synthetic_loader(num_samples=4, batch_size=4, seed=idx + 10),
            trainer=_make_trainer(),
            transport=t,
            port=0,
        )
        nodes[cid] = n
        reg = n.register()
        assert reg and reg.success

    try:
        assert server.global_model is not None
        for i in range(num_clients):
            _spawn_and_register(i)
        assert len(server.servicer.registered_nodes) == num_clients

        # Round 1: All 6 concurrent clients participate
        w_before = model_to_vector(server.global_model).clone()
        r1 = server.trigger_round(round_id=1, epochs=1, timeout=15.0, aggregate=True)
        assert len(r1) == num_clients
        assert all(res.success for res in r1)
        w_after_r1 = model_to_vector(server.global_model).clone()
        assert torch.isfinite(w_after_r1).all()
        assert not torch.allclose(w_after_r1, w_before)

        # Round 2: Drop client_2 abruptly before round dispatch
        nodes["client_2"].stop(grace=0)
        transports["client_2"].close()
        r2 = server.trigger_round(round_id=2, epochs=1, timeout=15.0, aggregate=True)
        assert len(r2) == num_clients - 1
        assert all(res.success for res in r2)
        assert "client_2" not in {res.node_id for res in r2}
        w_after_r2 = model_to_vector(server.global_model).clone()
        assert torch.isfinite(w_after_r2).all()

        # Round 3: Rejoin client_2 on a fresh ephemeral port and recover full cohort
        _spawn_and_register(2)
        r3 = server.trigger_round(round_id=3, epochs=1, timeout=15.0, aggregate=True)
        assert len(r3) == num_clients
        assert all(res.success for res in r3)
        assert "client_2" in {res.node_id for res in r3}
        w_after_r3 = model_to_vector(server.global_model).clone()
        assert torch.isfinite(w_after_r3).all()

    finally:
        for n in nodes.values():
            try:
                n.stop(grace=0)
            except Exception:
                pass
        for t in transports.values():
            try:
                t.close()
            except Exception:
                pass
        server.stop(grace=0)
