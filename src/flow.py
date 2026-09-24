"""Max-flow / min-cut between a sensory modality and the descending neurons.

Two network constructions, both with a super-source S -> sources and sinks -> super-sink T at infinite capacity:
  edge  capacity on each synapse edge = synapse count; the cut is a set of edges, mapped to neurons afterwards
  node  each neuron v is split into v_in -> v_out with capacity node_cap[v]; synapse edges are infinite, so the
        cut is a set of neurons. Sources and sinks get infinite internal capacity (they are the question, not
        the answer).
Direct source -> sink edges cannot be cut by removing intermediates, so they are removed from the network and
reported separately.

Three interchangeable solvers (scipy, ortools, igraph) return the same (value, source_side) result.

    python src/flow.py --dataset male-cns:v1.0   # all pathways, robustness, baselines -> data/processed/<ds>/flow/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import input_share_filter, load_graph, processed_dir  # noqa: E402
from sets import hops_from, load_sets  # noqa: E402

INF = 2**30  # fits scipy's int32 capacities; far above any real cut in this graph


class FlowNetwork:
    def __init__(self, tails, heads, caps, S, T, n_nodes, mode, n, direct_synapses):
        self.tails, self.heads, self.caps = tails, heads, caps
        self.S, self.T, self.n_nodes = S, T, n_nodes
        self.mode, self.n = mode, n
        self.direct_synapses = direct_synapses


def build_network(adj, src, snk, mode="edge", node_cap=None, removed=None):
    """adj: CSR (n x n) synapse counts. src/snk: disjoint idx arrays. node_cap: per-neuron capacity (node mode).
    removed: idx of ablated neurons; all their edges are dropped."""
    n = adj.shape[0]
    coo = adj.tocoo()
    r, c, w = coo.row, coo.col, coo.data.astype(np.int64)
    if removed is not None and len(removed):
        gone = np.zeros(n, bool)
        gone[removed] = True
        alive = ~(gone[r] | gone[c])
        r, c, w = r[alive], c[alive], w[alive]
    is_src = np.zeros(n, bool)
    is_src[src] = True
    is_snk = np.zeros(n, bool)
    is_snk[snk] = True
    direct = is_src[r] & is_snk[c]
    direct_syn = int(w[direct].sum())
    r, c, w = r[~direct], c[~direct], w[~direct]

    if mode == "edge":
        S, T = n, n + 1
        tails = np.r_[r, np.full(len(src), S), snk]
        heads = np.r_[c, src, np.full(len(snk), T)]
        caps = np.r_[np.minimum(w, INF), np.full(len(src) + len(snk), INF)]
        n_nodes = n + 2
    elif mode == "node":
        if node_cap is None:
            raise ValueError("node mode needs node_cap")
        cap = np.minimum(np.asarray(node_cap, dtype=np.int64), INF)
        cap[is_src | is_snk] = INF
        v = np.arange(n)
        S, T = 2 * n, 2 * n + 1
        tails = np.r_[v, r + n, np.full(len(src), S), snk + n]
        heads = np.r_[v + n, c, src, np.full(len(snk), T)]
        caps = np.r_[cap, np.full(len(r) + len(src) + len(snk), INF)]
        n_nodes = 2 * n + 2
    else:
        raise ValueError(mode)
    keep = caps > 0  # zero-capacity arcs carry nothing; dropping them keeps solvers from special-casing them
    return FlowNetwork(tails[keep].astype(np.int64), heads[keep].astype(np.int64), caps[keep].astype(np.int64),
                       S, T, n_nodes, mode, n, direct_syn)


def solve_scipy(net):
    G = sp.csr_matrix((net.caps.astype(np.int32), (net.tails, net.heads)), shape=(net.n_nodes,) * 2)
    res = maximum_flow(G, net.S, net.T, method="dinic")
    R = (G - res.flow).tocsr()  # flow is antisymmetric, so this is the residual graph incl. reverse edges
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    side = np.zeros(net.n_nodes, bool)
    side[breadth_first_order(R, net.S, directed=True, return_predecessors=False)] = True
    return int(res.flow_value), side


def solve_ortools(net):
    from ortools.graph.python import max_flow
    mf = max_flow.SimpleMaxFlow()
    mf.add_arcs_with_capacity(net.tails, net.heads, net.caps)
    status = mf.solve(net.S, net.T)
    if status != mf.OPTIMAL:
        raise RuntimeError(f"OR-Tools max flow status {status}")
    side = np.zeros(net.n_nodes, bool)
    side[np.asarray(mf.get_source_side_min_cut(), dtype=np.int64)] = True
    return int(mf.optimal_flow()), side


def solve_igraph(net):
    import igraph as ig
    g = ig.Graph(n=net.n_nodes, edges=np.c_[net.tails, net.heads].tolist(), directed=True)
    f = g.maxflow(net.S, net.T, capacity=net.caps.astype(float).tolist())
    part = next(p for p in f.partition if net.S in p)
    side = np.zeros(net.n_nodes, bool)
    side[part] = True
    return int(round(f.value)), side


SOLVERS = {"scipy": solve_scipy, "ortools": solve_ortools, "igraph": solve_igraph}


def solve(net, solver="ortools"):
    t0 = time.perf_counter()
    value, side = SOLVERS[solver](net)
    return {"value": value, "side": side, "seconds": time.perf_counter() - t0}


def cut_edges(net, side):
    """Edges from the source side to the sink side, excluding the infinite S/T and split-internal plumbing."""
    m = side[net.tails] & ~side[net.heads]
    return net.tails[m], net.heads[m], net.caps[m]


def cut_neurons(net, side):
    """Neurons in the cut.

    node mode: neurons whose split edge crosses the cut -> {idx: capacity}.
    edge mode: cut synapse edges aggregated per presynaptic neuron -> {idx: synapses cut}.
    """
    t, h, c = cut_edges(net, side)
    if net.mode == "node":
        internal = (t < net.n) & (h == t + net.n) & (c > 0)
        return dict(zip(t[internal].tolist(), c[internal].tolist()))
    real = (t < net.n) & (h < net.n)
    out = {}
    for a, x in zip(t[real].tolist(), c[real].tolist()):
        out[a] = out.get(a, 0) + x
    return out


# ---------------------------------------------------------------------------------------------------------
# Pipeline

# Behaviour-specific pathways. Cutting to *all* descending neurons returns the whole first relay layer (see
# reference_all_descending in the report), so the main analysis targets single identified output cells.
PATHWAYS = {
    "vision_to_giant_fiber": {"source": "vision", "sink_types": ["DNp01"]},
    "taste_to_mn9": {"source": "taste_head", "sink_types": ["MN9"]},
}
MAIN_SHARE = 0.01                     # keep edges carrying >= 1% of the postsynaptic neuron's input
INPUT_SHARES = (0.0, 0.005, 0.01, 0.02)
SYN_THRESHOLDS = (1, 3, 5, 10)        # applied together with MAIN_SHARE for the robustness check
N_RANDOM = 50
BETWEENNESS_SOURCES = 300             # sampled sources for pathway betweenness


def node_cut(adj, src, snk):
    net = build_network(adj, src, snk, "node", np.ones(adj.shape[0]))
    r = solve(net)
    return np.array(sorted(cut_neurons(net, r["side"])), dtype=np.int64), r["value"], net.direct_synapses


def edge_flow(adj, src, snk, removed=None):
    return solve(build_network(adj, src, snk, "edge", removed=removed))["value"]


def jaccard(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 3) if a | b else 1.0


def pagerank(adj, d=0.85, iters=100, tol=1e-10):
    n = adj.shape[0]
    out = np.asarray(adj.sum(1)).ravel().astype(float)
    P = sp.diags(np.divide(1.0, out, out=np.zeros(n), where=out > 0)) @ adj  # row-stochastic
    PT = P.T.tocsr()
    x = np.full(n, 1.0 / n)
    for _ in range(iters):
        dangling = x[out == 0].sum()
        x_new = d * (PT @ x + dangling / n) + (1 - d) / n
        if np.abs(x_new - x).sum() < tol:
            return x_new
        x = x_new
    return x


def pathway_betweenness(adj, src, snk, rng):
    """Betweenness counted only over shortest paths from (sampled) sources to the sinks."""
    import igraph as ig
    coo = adj.tocoo()
    g = ig.Graph(n=adj.shape[0], edges=np.c_[coo.row, coo.col], directed=True)
    s = rng.choice(src, size=min(BETWEENNESS_SOURCES, len(src)), replace=False)
    return np.asarray(g.betweenness(directed=True, sources=s.tolist(), targets=snk.tolist()))


def top_k(score, k, exclude):
    s = score.astype(float).copy()
    s[exclude] = -np.inf
    return np.argsort(s)[::-1][:k]


def on_path(adj, src, snk):
    """Neurons on at least one directed source -> sink path (excluding the sets themselves)."""
    m = np.isfinite(hops_from(adj, src)) & np.isfinite(hops_from(adj.T.tocsr(), snk))
    m[src] = m[snk] = False
    return np.flatnonzero(m)


def describe(neurons, idx):
    sub = neurons.iloc[idx]
    return {"n": len(idx), "types": sub["type"].fillna("?").value_counts().head(12).to_dict(),
            "superclass": sub["superclass"].fillna("?").value_counts().head(6).to_dict()}


def run_pathway(name, spec, full, neurons, sets, global_scores, rng):
    src = sets.query("set == @spec['source']")["idx"].to_numpy()
    snk = neurons.index[neurons["type"].isin(spec["sink_types"])].to_numpy()
    in_str = np.asarray(full.sum(0)).ravel()
    exclude = np.r_[src, snk]
    out = {"source": spec["source"], "n_sources": len(src), "sink_types": spec["sink_types"],
           "sink_bodyIds": neurons["bodyId"].iloc[snk].astype(int).tolist(), "cuts": {}}
    cuts = {}

    for p in INPUT_SHARES:
        adj = input_share_filter(full, p, in_str)
        cut, value, direct = node_cut(adj, src, snk)
        cuts[f"share{p:g}"] = cut
        out["cuts"][f"share{p:g}"] = {"min_share": p, "min_weight": 1, "value": int(value),
                                      "direct_synapses_excluded": direct, **describe(neurons, cut)}
    for t in SYN_THRESHOLDS[1:]:
        adj = input_share_filter(full.multiply(full >= t).tocsr(), MAIN_SHARE, in_str)
        cut, value, direct = node_cut(adj, src, snk)
        cuts[f"share{MAIN_SHARE:g}_w{t}"] = cut
        out["cuts"][f"share{MAIN_SHARE:g}_w{t}"] = {"min_share": MAIN_SHARE, "min_weight": t, "value": int(value),
                                                    "direct_synapses_excluded": direct, **describe(neurons, cut)}

    main_key = f"share{MAIN_SHARE:g}"
    main = cuts[main_key]
    out["main_cut"] = main_key
    out["robustness_jaccard_vs_main"] = {k: jaccard(main, v) for k, v in cuts.items() if k != main_key}

    # Baselines: remove the same number of neurons, measure how much synapse flow survives on the FULL graph.
    # The cut was found on the 1%-share graph, so on the full graph it is not guaranteed to disconnect anything.
    k = len(main)
    main_graph = input_share_filter(full, MAIN_SHARE, in_str)
    base_flow = edge_flow(full, src, snk)
    strategies = {
        "min_cut": main,
        "pathway_betweenness": top_k(pathway_betweenness(main_graph, src, snk, rng), k, exclude),
        "pagerank": top_k(global_scores["pagerank"], k, exclude),
        "out_strength": top_k(global_scores["out_strength"], k, exclude),
    }
    res = {s: edge_flow(full, src, snk, idx) / base_flow for s, idx in strategies.items()}
    pool = on_path(main_graph, src, snk)
    rand = [edge_flow(full, src, snk, rng.choice(pool, size=k, replace=False)) / base_flow for _ in range(N_RANDOM)]
    out["ablation"] = {
        "k": k, "baseline_flow_synapses": int(base_flow),
        "remaining_flow_share": {s: round(v, 4) for s, v in res.items()},
        "random_on_path": {"n": N_RANDOM, "pool": len(pool), "mean": round(float(np.mean(rand)), 4),
                           "sd": round(float(np.std(rand)), 4), "min": round(float(np.min(rand)), 4),
                           "values": [round(v, 4) for v in rand]},
        "overlap_with_min_cut": {s: jaccard(main, idx) for s, idx in strategies.items() if s != "min_cut"},
    }
    rows = [pd.DataFrame({"pathway": name, "variant": key, "idx": idx}) for key, idx in cuts.items()]
    rows += [pd.DataFrame({"pathway": name, "variant": f"baseline_{s}", "idx": idx})
             for s, idx in strategies.items() if s != "min_cut"]
    return out, pd.concat(rows, ignore_index=True)


def reference_all_descending(full, neurons, sets):
    """Why the main analysis does not cut to all descending neurons: the cut is the first relay layer."""
    snk = sets.query("set == 'descending'")["idx"].to_numpy()
    in_str = np.asarray(full.sum(0)).ravel()
    out = {}
    for m in ("taste_head", "olfaction", "johnston_organ", "vision"):
        src = sets.query("set == @m")["idx"].to_numpy()
        out[m] = {}
        for p in (0.0, MAIN_SHARE):
            adj = input_share_filter(full, p, in_str)
            cut, value, _ = node_cut(adj, src, snk)
            first = np.setdiff1d(np.flatnonzero(np.asarray(adj[src].sum(0)).ravel() > 0), np.r_[src, snk])
            out[m][f"share{p:g}"] = {"cut_size": len(cut), "first_layer_size": len(first),
                                     "share_of_cut_in_first_layer": round(float(np.isin(cut, first).mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    full, neurons = load_graph(args.dataset)
    sets = load_sets(args.dataset)
    outdir = processed_dir(args.dataset) / "flow"
    outdir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    global_scores = {"pagerank": pagerank(full), "out_strength": np.asarray(full.sum(1)).ravel()}
    report = {"dataset": args.dataset, "main_share": MAIN_SHARE, "seed": args.seed,
              "reference_all_descending": reference_all_descending(full, neurons, sets), "pathways": {}}
    print(f"reference done ({time.perf_counter() - t0:.0f}s)", flush=True)
    frames = []
    for name, spec in PATHWAYS.items():
        rep, cuts = run_pathway(name, spec, full, neurons, sets, global_scores, rng)
        report["pathways"][name] = rep
        frames.append(cuts)
        a = rep["ablation"]
        print(f"{name}: k={a['k']}, remaining flow {a['remaining_flow_share']}, random {a['random_on_path']['mean']}"
              f" +- {a['random_on_path']['sd']} ({time.perf_counter() - t0:.0f}s)", flush=True)

    cuts = pd.concat(frames, ignore_index=True)
    cuts = cuts.merge(neurons[["idx", "bodyId", "type", "superclass"]], on="idx", how="left")
    cuts.to_parquet(outdir / "cuts.parquet", index=False)
    (outdir / "flow_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ablation"} for k, v in report["pathways"].items()},
                     indent=1, default=str)[:6000])
    print(json.dumps(report["reference_all_descending"], indent=1))


if __name__ == "__main__":
    main()
