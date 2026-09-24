import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flow import SOLVERS, build_network, cut_neurons, solve  # noqa: E402


def csr(edges, n):
    r, c, w = zip(*edges)
    return sp.csr_matrix((w, (r, c)), shape=(n, n))


# 0 -> {1,2} -> {3,4} -> 5; min edge cut is {1->3 (4), 2->4 (9)} = 13
EDGE_CASE = csr([(0, 1, 10), (0, 2, 10), (1, 3, 4), (1, 2, 2), (2, 4, 9), (4, 3, 6), (3, 5, 10), (4, 5, 10)], 6)

# sources {0,1} funnel through neuron 4 to sink 8; 0 -> 8 is a direct edge that must be excluded
NODE_CASE = csr([(0, 2, 5), (1, 3, 5), (2, 4, 5), (3, 4, 5), (4, 5, 1), (4, 6, 5), (5, 8, 5), (6, 8, 5), (0, 8, 3)], 9)


@pytest.mark.parametrize("solver", SOLVERS)
def test_edge_cut(solver):
    net = build_network(EDGE_CASE, np.array([0]), np.array([5]), mode="edge")
    r = solve(net, solver)
    assert r["value"] == 13
    assert cut_neurons(net, r["side"]) == {1: 4, 2: 9}


@pytest.mark.parametrize("solver", SOLVERS)
def test_node_cut_unit(solver):
    net = build_network(NODE_CASE, np.array([0, 1]), np.array([8]), mode="node", node_cap=np.ones(9))
    r = solve(net, solver)
    assert net.direct_synapses == 3
    assert r["value"] == 1
    assert set(cut_neurons(net, r["side"])) == {4}


@pytest.mark.parametrize("solver", SOLVERS)
def test_node_cut_weighted(solver):
    out_strength = np.asarray(NODE_CASE.sum(1)).ravel()
    net = build_network(NODE_CASE, np.array([0, 1]), np.array([8]), mode="node", node_cap=out_strength)
    r = solve(net, solver)
    assert r["value"] == 6
    assert cut_neurons(net, r["side"]) == {4: 6}


@pytest.mark.parametrize("seed", range(5))
def test_random_graphs_match_networkx(seed):
    rng = np.random.default_rng(seed)
    n = 60
    dense = (rng.random((n, n)) < 0.08) * rng.integers(1, 20, (n, n))
    np.fill_diagonal(dense, 0)
    adj = sp.csr_matrix(dense)
    src, snk = np.arange(5), np.arange(n - 5, n)

    g = nx.DiGraph()
    for a, b in zip(*adj.nonzero()):
        if not (a in src and b in snk):
            g.add_edge(int(a), int(b), capacity=int(adj[a, b]))
    for s in src:
        g.add_edge("S", int(s))
    for t in snk:
        g.add_edge(int(t), "T")
    expected_edge = nx.maximum_flow_value(g, "S", "T")

    for u, v in g.edges:
        if isinstance(u, int) and isinstance(v, int):
            g[u][v]["capacity"] = float("inf")
    h = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        h.add_edge(("o", u) if isinstance(u, int) else u, ("i", v) if isinstance(v, int) else v, **d)
    for v in range(n):
        h.add_edge(("i", v), ("o", v), **({} if v in src or v in snk else {"capacity": 1}))
    expected_node = nx.maximum_flow_value(h, "S", "T")

    for solver in SOLVERS:
        assert solve(build_network(adj, src, snk, "edge"), solver)["value"] == expected_edge
        assert solve(build_network(adj, src, snk, "node", np.ones(n)), solver)["value"] == expected_node
