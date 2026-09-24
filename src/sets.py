"""Source (sensory, per modality) and sink (descending) neuron sets, plus a reachability summary.

    python src/sets.py --dataset male-cns:v1.0

Writes data/processed/<dataset>/sets.parquet (modality, role, idx, bodyId, type) and sets_report.json,
which includes Neuroglancer links to a few random members of each set for manual spot checks.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import load_graph, processed_dir  # noqa: E402
from ingest import ROOT  # noqa: E402

# Sources, as pandas queries on the neuron table. All come from `class` / `subclass` / `superclass` annotations.
# Gustatory is split because head taste (labellum, pharynx) enters the brain directly, while leg and wing
# taste enters the VNC and has to ascend.
MODALITIES = {
    "vision":         "`class` == 'visual'",
    "olfaction":      "`class` == 'olfactory'",
    "taste_head":     "`class` == 'gustatory' and superclass == 'cb_sensory'",
    "taste_body":     "`class` == 'gustatory' and superclass in ['vnc_sensory', 'sensory_ascending']",
    "johnston_organ": "`class` == 'mechanosensory' and subclass in ['auditory', 'wind_gravity']",
    "mechano_head":   "`class` == 'mechanosensory'",
    "touch_body":     "`class` == 'mechanosensory_tactile'",
    "proprioception": "`class` == 'mechanosensory_proprioceptive'",
    "hygrosensation": "`class` == 'hygrosensory'",
    "thermosensation": "`class` == 'thermosensory'",
}
# Sinks. `descending` is the main one; `motor` is kept as an alternative readout.
SINKS = {
    "descending": "superclass == 'descending_neuron'",
    "motor":      "superclass in ['vnc_motor', 'cb_motor']",
}

NG_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/male-cns-v1.0.json"
NG_VIEWER = "https://neuroglancer-demo.appspot.com/#!"


def build_sets(neurons):
    rows = []
    for role, defs in (("source", MODALITIES), ("sink", SINKS)):
        for name, expr in defs.items():
            sel = neurons.query(expr)
            rows.append(pd.DataFrame({"set": name, "role": role, "idx": sel["idx"].to_numpy(),
                                      "bodyId": sel["bodyId"].to_numpy(), "type": sel["type"].to_numpy()}))
    sets = pd.concat(rows, ignore_index=True)
    overlap = set(sets.loc[sets.role == "source", "idx"]) & set(sets.loc[sets.role == "sink", "idx"])
    if overlap:
        sys.exit(f"{len(overlap)} neurons are in both a source and a sink set")
    return sets


def hops_from(adj, src):
    """Unweighted hop distance from the nearest node in `src` to every node (multi-source BFS)."""
    n = adj.shape[0]
    coo = adj.tocoo()
    rows = np.r_[coo.row, np.full(len(src), n)]
    cols = np.r_[coo.col, src]
    g = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n + 1, n + 1))
    return shortest_path(g, unweighted=True, indices=n, directed=True)[:n] - 1


def summarize(adj, sets, sink="descending"):
    snk = sets.query("set == @sink")["idx"].to_numpy()
    is_snk = np.zeros(adj.shape[0], bool)
    is_snk[snk] = True
    out = {}
    for m in MODALITIES:
        src = sets.query("set == @m")["idx"].to_numpy()
        sub = adj[src]
        total = int(sub.sum())
        direct = int(sub[:, snk].sum())
        h = hops_from(adj, src)[snk]
        ok = np.isfinite(h)
        out[m] = {
            "neurons": len(src),
            "output_synapses": total,
            "direct_to_sink_synapses": direct,
            "direct_to_sink_share": round(direct / total, 4) if total else None,
            "sinks_contacted_directly": int((np.asarray(sub[:, snk].sum(0)).ravel() > 0).sum()),
            "sinks_reachable": int(ok.sum()),
            "hops_median": float(np.median(h[ok])) if ok.any() else None,
            "hops_share": {str(k): round(float(np.mean(h[ok] <= k)), 3) for k in (1, 2, 3, 4)} if ok.any() else None,
        }
    return out


def ng_base():
    path = ROOT / "data" / "raw" / "neuroglancer" / "male-cns-v1.0.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(NG_BASE, timeout=60).read())
    return json.loads(path.read_text(encoding="utf-8"))


def ng_link(body_ids, base):
    """Neuroglancer state showing only `body_ids` in 3D, with the brain and VNC outlines for context."""
    layers = {l["name"]: l for l in base["layers"]}
    seg = dict(layers["cns-seg"], segments=[str(b) for b in body_ids])
    shells = [dict(layers[n], archived=False, objectAlpha=0.1) for n in ("brain-shell", "vnc-shell")]
    state = {"dimensions": base["dimensions"], "position": base["position"],
             "projectionScale": base.get("projectionScale", 100000),
             "layers": [seg, *shells], "selectedLayer": {"layer": "cns-seg", "visible": True},
             "layout": "3d"}
    return NG_VIEWER + urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")


def spot_checks(sets, k=5, seed=0):
    rng = np.random.default_rng(seed)
    base = ng_base()
    out = {}
    for name, grp in sets.groupby("set", sort=False):
        pick = grp.iloc[rng.choice(len(grp), size=min(k, len(grp)), replace=False)]
        out[name] = {"neurons": pick[["bodyId", "type"]].astype({"bodyId": int}).values.tolist(),
                     "neuroglancer": ng_link(pick["bodyId"], base)}
    return out


def load_sets(dataset):
    return pd.read_parquet(processed_dir(dataset) / "sets.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-weight", type=int, default=1)
    args = ap.parse_args()

    adj, neurons = load_graph(args.dataset, min_weight=args.min_weight)
    sets = build_sets(neurons)
    sets.to_parquet(processed_dir(args.dataset) / "sets.parquet", index=False)

    rep = {"dataset": args.dataset, "min_weight": args.min_weight,
           "definitions": {"sources": MODALITIES, "sinks": SINKS},
           "counts": sets.groupby(["role", "set"], sort=False).size().to_dict(),
           "reachability_to_descending": summarize(adj, sets),
           "spot_checks": spot_checks(sets)}
    rep["counts"] = {f"{r}:{s}": int(v) for (r, s), v in rep["counts"].items()}
    (processed_dir(args.dataset) / "sets_report.json").write_text(json.dumps(rep, indent=2))

    print(f"Sets (min_weight={args.min_weight}):")
    for k, v in rep["counts"].items():
        print(f"  {k:<28}{v:>7,}")
    print("\nReachability to descending neurons:")
    print(pd.DataFrame(rep["reachability_to_descending"]).T.drop(columns="hops_share").to_string())
    print("\nSpot checks (5 random per set; links in sets_report.json):")
    for name, sc in rep["spot_checks"].items():
        print(f"  {name}: {sc['neurons']}")


if __name__ == "__main__":
    main()
