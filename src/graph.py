"""Build the processed connectome graph from the raw neuPrint cache and load it at a synapse threshold.

    python src/graph.py --dataset male-cns:v1.0            # build data/processed/<dataset>/ if missing
    python src/graph.py --dataset male-cns:v1.0 --refresh  # rebuild

Output (data/processed/<dataset>/):
    neurons.parquet  one row per neuron that has at least one edge; `idx` is its row/column in the adjacency
    edges.parquet    pre, post (int32 idx), weight, weightHP (int16); self-loops removed

Downstream code should use load_graph() / load_neurons() and never read data/raw directly.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import cache_dir, load_edges  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Sign of a neuron's output synapses, keyed by consensusNt. Assumptions, not measurements:
#   acetylcholine  +1  nicotinic receptors, the main fast excitatory transmitter in the fly CNS
#   gaba           -1  GABA-A (Rdl) chloride channels
#   glutamate      -1  GluCl chloride channels in the CNS; excitatory at the NMJ and at some central synapses
#   histamine      -1  HisCl channels (photoreceptor output)
#   dopamine, serotonin, octopamine  0  metabotropic / modulatory, sign depends on receptor
#   unclear, missing                 0  no usable prediction
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 0, "serotonin": 0, "octopamine": 0, "unclear": 0}
NT_COLUMN = "consensusNt"

# Float columns that are really integer ids with missing values.
ID_LIKE = ["group", "mancBodyid", "mancGroup", "mancSerial", "mcnsSerial",
           "assignedOlHex1", "assignedOlHex2", "celltypeTotalNtPredictions", "totalNtPredictions"]


def processed_dir(dataset):
    return ROOT / "data" / "processed" / dataset.replace(":", "_")


def build(dataset):
    out = processed_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_edges(dataset, columns=["bodyId_pre", "bodyId_post", "weight", "weightHP"])
    pre_id, post_id = raw["bodyId_pre"].to_numpy(), raw["bodyId_post"].to_numpy()
    keep = pre_id != post_id
    n_loops = int((~keep).sum())

    neurons = pd.read_parquet(cache_dir(dataset) / "neurons.parquet")
    for col in ID_LIKE:
        if col in neurons:
            neurons[col] = neurons[col].astype("Int64")

    in_graph = np.union1d(pre_id[keep], post_id[keep])
    neurons = neurons[neurons["bodyId"].isin(in_graph)].sort_values("bodyId").reset_index(drop=True)
    neurons.insert(0, "idx", np.arange(len(neurons), dtype=np.int32))
    neurons["nt"] = neurons[NT_COLUMN].fillna("unclear")
    neurons["nt_sign"] = neurons["nt"].map(NT_SIGN).fillna(0).astype(np.int8)

    ids = neurons["bodyId"].to_numpy()
    if len(ids) != len(in_graph):
        sys.exit(f"{len(in_graph) - len(ids)} edge endpoints are missing from the neuron table; re-run ingest.")
    edges = pa.table({
        "pre": np.searchsorted(ids, pre_id[keep]).astype(np.int32),
        "post": np.searchsorted(ids, post_id[keep]).astype(np.int32),
        "weight": raw["weight"].to_numpy()[keep].astype(np.int16),
        "weightHP": raw["weightHP"].to_numpy()[keep].astype(np.int16),
    })
    neurons.to_parquet(out / "neurons.parquet", index=False)
    pq.write_table(edges, out / "edges.parquet")
    print(f"built {out.relative_to(ROOT)}: {len(neurons):,} neurons, {edges.num_rows:,} edges "
          f"({n_loops} self-loops dropped)")


def load_neurons(dataset, columns=None):
    return pd.read_parquet(processed_dir(dataset) / "neurons.parquet", columns=columns)


def load_edge_table(dataset, min_weight=1, weight_col="weight", excitatory_only=False, neurons=None):
    """Edge list as a pyarrow Table (pre, post, weight_col). Filters are applied while reading."""
    flt = pads.field(weight_col) >= min_weight
    table = pads.dataset(processed_dir(dataset) / "edges.parquet").to_table(
        columns=["pre", "post", weight_col], filter=flt)
    if excitatory_only:
        if neurons is None:
            neurons = load_neurons(dataset, columns=["idx", "nt_sign"])
        exc = neurons["nt_sign"].to_numpy() == 1
        table = table.filter(pa.array(exc[table["pre"].to_numpy()]))
    return table


def load_graph(dataset, min_weight=1, weight_col="weight", excitatory_only=False):
    """Directed weighted adjacency as CSR (row = presynaptic idx), plus the neuron table.

    min_weight is a synapse-count threshold on `weight_col`. excitatory_only keeps edges whose
    presynaptic neuron has nt_sign == +1 (see NT_SIGN). Neurons are never dropped, so idx stays stable
    across thresholds and subgraphs.
    """
    neurons = load_neurons(dataset)
    t = load_edge_table(dataset, min_weight, weight_col, excitatory_only, neurons)
    n = len(neurons)
    adj = sp.csr_matrix((t[weight_col].to_numpy().astype(np.int32),
                         (t["pre"].to_numpy(), t["post"].to_numpy())), shape=(n, n))
    return adj, neurons


def input_share_filter(adj, min_share, in_strength=None):
    """Keep edges u->v that make up at least `min_share` of v's total input synapses.

    in_strength defaults to adj's own column sums; pass the full graph's to keep shares comparable across
    synapse thresholds.
    """
    if min_share <= 0:
        return adj
    if in_strength is None:
        in_strength = np.asarray(adj.sum(0)).ravel()
    coo = adj.tocoo()
    keep = coo.data >= min_share * in_strength[coo.col]
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=adj.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if (processed_dir(args.dataset) / "edges.parquet").exists() and not args.refresh:
        print("processed graph exists (use --refresh to rebuild)")
    else:
        t0 = time.perf_counter()
        build(args.dataset)
        print(f"  {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
