"""EDA of the processed graph: sizes per threshold, degree and weight distributions, NT coverage.

    python src/eda.py --dataset male-cns:v1.0

Writes data/processed/<dataset>/eda.json and prints a readable summary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import NT_SIGN, load_edge_table, load_graph, load_neurons, processed_dir  # noqa: E402

THRESHOLDS = (1, 3, 5, 10)
Q = {"min": 0, "q1": 25, "median": 50, "q3": 75, "p90": 90, "p99": 99, "max": 100}


def dist(x):
    x = np.asarray(x)
    out = {k: float(np.percentile(x, p)) for k, p in Q.items()}
    out["mean"] = float(x.mean())
    out["share_zero"] = round(float((x == 0).mean()), 4)
    return out


def csr_mb(adj):
    return round((adj.data.nbytes + adj.indices.nbytes + adj.indptr.nbytes) / 1e6, 1)


def sizes(dataset):
    rows = []
    for exc in (False, True):
        for t in THRESHOLDS:
            t0 = time.perf_counter()
            adj, _ = load_graph(dataset, min_weight=t, excitatory_only=exc)
            dt = time.perf_counter() - t0
            deg = np.asarray((adj != 0).sum(0)).ravel() + np.diff(adj.indptr)
            rows.append({"subgraph": "excitatory" if exc else "all", "min_weight": t, "edges": int(adj.nnz),
                         "synapses": int(adj.data.sum()), "neurons_with_edges": int((deg > 0).sum()),
                         "csr_mb": csr_mb(adj), "load_s": round(dt, 2)})
    return rows


def degrees(dataset, neurons, min_weight):
    adj, _ = load_graph(dataset, min_weight=min_weight)
    out_deg, in_deg = np.diff(adj.indptr), np.diff(adj.tocsc().indptr)
    out_str, in_str = np.asarray(adj.sum(1)).ravel(), np.asarray(adj.sum(0)).ravel()

    def top(x, k=10):
        i = np.argsort(x)[::-1][:k]
        return [{"bodyId": int(neurons.bodyId.iat[j]), "type": neurons.type.iat[j],
                 "superclass": neurons.superclass.iat[j], "value": int(x[j])} for j in i]

    return {"min_weight": min_weight,
            "out_degree": dist(out_deg), "in_degree": dist(in_deg),
            "out_strength": dist(out_str), "in_strength": dist(in_str),
            "top_out_degree": top(out_deg), "top_in_degree": top(in_deg)}


def weights(dataset):
    t = load_edge_table(dataset, weight_col="weight")
    w = t["weight"].to_numpy().astype(np.int64)
    hp = load_edge_table(dataset, weight_col="weightHP", min_weight=0)["weightHP"].to_numpy()
    return {"distribution": dist(w),
            "share_edges_eq": {str(k): round(float((w == k).mean()), 4) for k in (1, 2)},
            "share_edges_below": {str(k): round(float((w < k).mean()), 4) for k in THRESHOLDS[1:]},
            "share_synapses_below": {str(k): round(float(w[w < k].sum() / w.sum()), 4) for k in THRESHOLDS[1:]},
            "weightHP_zero_edges": int((hp == 0).sum()),
            "weightHP_over_weight": round(float(hp.sum() / w.sum()), 4)}


def nt(dataset, neurons):
    t = load_edge_table(dataset)
    pre, w = t["pre"].to_numpy(), t["weight"].to_numpy().astype(np.int64)
    syn_by_nt = pd.Series(w).groupby(neurons["nt"].to_numpy()[pre]).sum()
    conf = neurons.groupby("nt")["predictedNtConfidence"].describe()[["count", "25%", "50%", "75%"]]
    agree = {c: round(float((neurons["consensusNt"] == neurons[c]).sum() / neurons["consensusNt"].notna().sum()), 4)
             for c in ("predictedNt", "celltypePredictedNt")}
    sens = neurons[neurons.superclass.fillna("").str.contains("sensory")]
    exc_share = float(syn_by_nt.get("acetylcholine", 0) / syn_by_nt.sum())
    return {
        "column_used": "consensusNt",
        "sign_mapping": NT_SIGN,
        "neurons_by_nt": neurons["nt"].value_counts().to_dict(),
        "neurons_missing_nt": int(neurons["consensusNt"].isna().sum()),
        "share_neurons_sign_known": round(float((neurons["nt_sign"] != 0).mean()), 4),
        "synapses_by_presynaptic_nt": {k: int(v) for k, v in syn_by_nt.sort_values(ascending=False).items()},
        "share_synapses_excitatory": round(exc_share, 4),
        "predictedNtConfidence_by_nt": conf.round(3).to_dict(orient="index"),
        "agreement_consensus_vs": agree,
        "sensory_nt_by_superclass": pd.crosstab(sens.superclass, sens.nt).to_dict(orient="index"),
    }


def annotations(dataset, neurons):
    t = load_edge_table(dataset)
    pre, post, w = t["pre"].to_numpy(), t["post"].to_numpy(), t["weight"].to_numpy().astype(np.int64)
    has = neurons["superclass"].notna().to_numpy()
    return {"neurons": len(neurons),
            "with_superclass": int(has.sum()), "with_type": int(neurons["type"].notna().sum()),
            "share_synapses_touching_no_superclass": round(float(w[~(has[pre] & has[post])].sum() / w.sum()), 4),
            "no_superclass_by_status": neurons.loc[~has, "statusLabel"].value_counts().head(8).to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    neurons = load_neurons(args.dataset)
    rep = {"dataset": args.dataset,
           "sizes": sizes(args.dataset),
           "annotations": annotations(args.dataset, neurons),
           "weights": weights(args.dataset),
           "degrees": [degrees(args.dataset, neurons, t) for t in (1, 5)],
           "nt": nt(args.dataset, neurons)}
    (processed_dir(args.dataset) / "eda.json").write_text(json.dumps(rep, indent=2, default=str))

    print("\nSizes (in-memory CSR arrays, not process RSS):")
    print(pd.DataFrame(rep["sizes"]).to_string(index=False))
    print("\nAnnotations:", json.dumps(rep["annotations"], indent=1, default=str))
    print("\nWeights:", json.dumps(rep["weights"], indent=1))
    for d in rep["degrees"]:
        print(f"\nDegrees at min_weight={d['min_weight']}:")
        print(pd.DataFrame({k: d[k] for k in ("out_degree", "in_degree", "out_strength", "in_strength")}).T
              .round(1).to_string())
        print("  top out-degree:", [(x["type"], x["superclass"], x["value"]) for x in d["top_out_degree"][:5]])
        print("  top in-degree: ", [(x["type"], x["superclass"], x["value"]) for x in d["top_in_degree"][:5]])
    print("\nNT:", json.dumps(rep["nt"], indent=1, default=str))


if __name__ == "__main__":
    main()
