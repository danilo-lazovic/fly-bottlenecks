"""Pull the neuron table and the neuron-to-neuron edge list from neuPrint, cached as parquet.

    python src/ingest.py --dataset male-cns:v1.0            # pull whatever is not cached yet, then report
    python src/ingest.py --dataset male-cns:v1.0 --refresh  # drop the cache and pull again
    python src/ingest.py --dataset male-cns:v1.0 --report   # report from cache only, no API calls

Output (data/raw/<dataset>/):
    neurons.parquet        one row per :Neuron, all non-ROI properties except roiInfo
    edges/part-*.parquet   bodyId_pre, bodyId_post, weight, weightHP, weightHR (read the folder as one table)
    manifest.json          batch layout, so an interrupted pull resumes with the same batches
    report.json            counts compared with the published numbers
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from dotenv import load_dotenv
from neuprint import Client
from tqdm import tqdm

SERVER = "https://neuprint.janelia.org"
ROOT = Path(__file__).resolve().parents[1]

PUBLISHED_NEURONS = 166_700
PUBLISHED_SYNAPSES = 125_000_000

NEURON_PAGE = 20_000
EDGE_BATCH = 1_000  # presynaptic neurons per edge query
PAUSE_S = 0.5
RETRIES = 5

# Dropped from the neuron pull: per-ROI JSON is large and ROI membership is recoverable from edges later.
SKIP_NEURON_KEYS = {"roiInfo"}

INT_COLS = {"bodyId": "int64", "pre": "int32", "post": "int32", "upstream": "int32",
            "downstream": "int32", "synweight": "int32", "size": "Int64"}
EDGE_SCHEMA = pa.schema([("bodyId_pre", pa.int64()), ("bodyId_post", pa.int64()),
                         ("weight", pa.int32()), ("weightHP", pa.int32()), ("weightHR", pa.int32())])


def cache_dir(dataset):
    return ROOT / "data" / "raw" / dataset.replace(":", "_")


def get_client(dataset):
    load_dotenv(ROOT / ".env")
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "").strip()
    if not token:
        sys.exit("NEUPRINT_APPLICATION_CREDENTIALS is empty. See .env.example.")
    return Client(SERVER, dataset=dataset, token=token, progress=False)


def query(client, cypher):
    for attempt in range(RETRIES):
        try:
            return client.fetch_custom(cypher)
        except Exception as ex:
            if attempt == RETRIES - 1:
                raise
            wait = 2 ** attempt * 5
            tqdm.write(f"  query failed ({type(ex).__name__}: {str(ex)[:120]}), retry in {wait}s")
            time.sleep(wait)


def neuron_keys(client):
    rois = set(client.all_rois)
    return sorted(k for k in client.fetch_neuron_keys() if k not in rois and k not in SKIP_NEURON_KEYS)


def pull_neurons(client, out):
    keys = neuron_keys(client)
    ret = ", ".join(f"n.`{k}` AS `{k}`" for k in keys)
    total = int(query(client, "MATCH (n:Neuron) RETURN count(n)").iloc[0, 0])

    pages, last = [], -1
    with tqdm(total=total, desc="neurons", unit="row") as bar:
        while True:
            df = query(client, f"MATCH (n:Neuron) WHERE n.bodyId > {last} RETURN {ret} "
                               f"ORDER BY n.bodyId LIMIT {NEURON_PAGE}")
            if df.empty:
                break
            pages.append(df)
            last = int(df["bodyId"].iloc[-1])
            bar.update(len(df))
            time.sleep(PAUSE_S)

    df = pd.concat(pages, ignore_index=True)
    if "somaLocation" in df:
        xyz = df.pop("somaLocation").map(lambda p: p["coordinates"] if isinstance(p, dict) else [None] * 3)
        df[["somaX", "somaY", "somaZ"]] = pd.DataFrame(xyz.tolist(), index=df.index).astype("Int32")
    for col, dtype in INT_COLS.items():
        if col in df:
            df[col] = df[col].astype(dtype)
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    df.to_parquet(out, index=False)
    return df


def pull_edges(client, body_ids, edge_dir, manifest_path):
    edge_dir.mkdir(parents=True, exist_ok=True)
    batches = [body_ids[i:i + EDGE_BATCH] for i in range(0, len(body_ids), EDGE_BATCH)]
    layout = {"edge_batch": EDGE_BATCH, "n_pre": len(body_ids), "n_batches": len(batches),
              "first_id": int(body_ids[0]), "last_id": int(body_ids[-1])}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != layout:
            sys.exit(f"Cached edge batches were made with a different layout {old} vs {layout}. Use --refresh.")
    manifest_path.write_text(json.dumps(layout, indent=2))

    todo = [i for i in range(len(batches)) if not (edge_dir / f"part-{i:05d}.parquet").exists()]
    if len(todo) < len(batches):
        print(f"  {len(batches) - len(todo)}/{len(batches)} edge batches already cached")
    for i in tqdm(todo, desc="edge batches", unit="batch"):
        ids = ",".join(str(b) for b in batches[i])
        df = query(client, f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN [{ids}] "
                           f"RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, "
                           f"e.weight AS weight, e.weightHP AS weightHP, e.weightHR AS weightHR")
        table = pa.Table.from_pandas(df, schema=EDGE_SCHEMA, preserve_index=False) if len(df) \
            else EDGE_SCHEMA.empty_table()
        tmp = edge_dir / f"part-{i:05d}.parquet.tmp"
        pq.write_table(table, tmp)
        tmp.replace(edge_dir / f"part-{i:05d}.parquet")  # atomic, so a crash never leaves a half-written part
        time.sleep(PAUSE_S)


def load_edges(dataset, columns=None, min_weight=1):
    """Edge list from cache. min_weight filters on `weight` while reading, so weak edges never hit RAM."""
    ds = pads.dataset(cache_dir(dataset) / "edges", format="parquet")
    flt = pads.field("weight") >= min_weight if min_weight > 1 else None
    return ds.to_table(columns=columns, filter=flt)


def count_duplicate_pairs(pre, post):
    order = np.lexsort((post, pre))
    p, q = pre[order], post[order]
    return int(((p[1:] == p[:-1]) & (q[1:] == q[:-1])).sum())


def report(dataset):
    d = cache_dir(dataset)
    neurons = pd.read_parquet(d / "neurons.parquet")
    edges = load_edges(dataset)

    pre, post = edges["bodyId_pre"].to_numpy(), edges["bodyId_post"].to_numpy()
    w = edges["weight"].to_numpy().astype(np.int64)
    in_edges = np.union1d(pre, post)
    known = neurons["bodyId"].to_numpy()

    n_super = int(neurons["superclass"].notna().sum()) if "superclass" in neurons else None
    total_w = int(w.sum())
    rep = {
        "dataset": dataset,
        "neurons_rows": len(neurons),
        "neurons_unique_bodyId": int(neurons["bodyId"].nunique()),
        "neurons_with_superclass": n_super,
        "published_neurons": PUBLISHED_NEURONS,
        "diff_superclass_vs_published": None if n_super is None else n_super - PUBLISHED_NEURONS,
        "edges_rows": len(w),
        "edges_duplicate_pairs": count_duplicate_pairs(pre, post),
        "neurons_in_edges": int(len(in_edges)),
        "neurons_without_edges": int(len(np.setdiff1d(known, in_edges))),
        "edge_endpoints_missing_from_neurons": int(len(np.setdiff1d(in_edges, known))),
        "synapses_sum_weight": total_w,
        "synapses_sum_weightHP": int(edges["weightHP"].to_numpy().astype(np.int64).sum()),
        "synapses_sum_weightHR": int(edges["weightHR"].to_numpy().astype(np.int64).sum()),
        "published_synapses": PUBLISHED_SYNAPSES,
        "diff_weight_vs_published_pct": round(100 * (total_w - PUBLISHED_SYNAPSES) / PUBLISHED_SYNAPSES, 2),
        "edges_by_min_weight": {str(t): int((w >= t).sum()) for t in (1, 3, 5, 10)},
        "synapses_by_min_weight": {str(t): int(w[w >= t].sum()) for t in (1, 3, 5, 10)},
        "edges_parquet_mb": round(sum(f.stat().st_size for f in (d / "edges").glob("*.parquet")) / 1e6, 1),
        "neurons_parquet_mb": round((d / "neurons.parquet").stat().st_size / 1e6, 1),
    }
    (d / "report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="neuPrint dataset, e.g. from src/inspect_schema.py")
    ap.add_argument("--refresh", action="store_true", help="delete the cache and pull again")
    ap.add_argument("--report", action="store_true", help="only report from cache")
    args = ap.parse_args()

    d = cache_dir(args.dataset)
    if args.report:
        report(args.dataset)
        return
    if args.refresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

    neurons_path = d / "neurons.parquet"
    edges_done = (d / "manifest.json").exists() and \
        len(list((d / "edges").glob("part-*.parquet"))) == json.loads((d / "manifest.json").read_text())["n_batches"]
    if neurons_path.exists() and edges_done:
        print(f"Cache complete in {d.relative_to(ROOT)}; nothing to pull (use --refresh to force).")
    else:
        client = get_client(args.dataset)
        if neurons_path.exists():
            neurons = pd.read_parquet(neurons_path, columns=["bodyId"])
            print(f"neurons: cached ({len(neurons):,} rows)")
        else:
            neurons = pull_neurons(client, neurons_path)
        body_ids = np.sort(neurons["bodyId"].to_numpy())
        pull_edges(client, body_ids, d / "edges", d / "manifest.json")

    print("\nReport:")
    report(args.dataset)


if __name__ == "__main__":
    main()
