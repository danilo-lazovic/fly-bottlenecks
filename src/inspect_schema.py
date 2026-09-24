"""Read-only probe of neuPrint: list datasets, then describe the Neuron and ConnectsTo schema.

Nothing here pulls the full tables. Output is printed and cached to data/raw/schema/<dataset>.json.

    python src/inspect_schema.py                 # list datasets, auto-pick male CNS if unambiguous
    python src/inspect_schema.py --dataset NAME  # inspect a specific dataset
    python src/inspect_schema.py --refresh       # ignore the cached report
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from neuprint import Client

SERVER = "https://neuprint.janelia.org"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw" / "schema"

# Columns worth a value-count breakdown, matched by name so we don't assume exact spellings.
CATEGORICAL_HINT = re.compile(r"(class|status|nt|neurotransmitter|side|type|group|hemilineage|origin|target|region)", re.I)


def get_token():
    load_dotenv(ROOT / ".env")
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "").strip()
    if not token:
        sys.exit("NEUPRINT_APPLICATION_CREDENTIALS is empty. Copy .env.example to .env and paste your token.")
    return token


def list_datasets(token):
    r = requests.get(f"{SERVER}/api/dbmeta/datasets", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def pick_dataset(datasets, requested):
    names = list(datasets)
    if requested:
        if requested not in names:
            sys.exit(f"Dataset {requested!r} not found. Available: {names}")
        return requested
    candidates = [n for n in names if re.search(r"male", n, re.I) and not re.search(r"female", n, re.I)
                  and re.search(r"cns", n, re.I)]
    if len(candidates) == 1:
        return candidates[0]
    sys.exit(f"Could not pick a unique male CNS dataset (candidates: {candidates}). Re-run with --dataset NAME.")


def q(client, cypher):
    return client.fetch_custom(cypher)


def describe_neurons(client, roi_names):
    keys = sorted(client.fetch_neuron_keys())
    roi_set = set(roi_names)
    prop_keys = [k for k in keys if k not in roi_set]
    roi_keys = [k for k in keys if k in roi_set]

    counts = q(client, "MATCH (n:Neuron) RETURN count(n) AS n_neuron").iloc[0, 0]
    n_segment = None
    try:
        n_segment = int(q(client, "MATCH (n:Segment) RETURN count(n) AS n").iloc[0, 0])
    except Exception as ex:  # can be slow/forbidden on big datasets; not essential
        print(f"  (Segment count skipped: {ex})")

    # count() ignores nulls, so one scan gives completeness for every property.
    ret = ", ".join(f"count(n.`{k}`) AS `{k}`" for k in prop_keys)
    filled = q(client, f"MATCH (n:Neuron) RETURN {ret}").iloc[0].to_dict()

    sample = q(client, "MATCH (n:Neuron) WHERE n.type IS NOT NULL RETURN n LIMIT 3")
    sample_rows = [{k: v for k, v in row.items() if k not in roi_set} for row in sample["n"]]

    columns = []
    for k in prop_keys:
        example = next((r[k] for r in sample_rows if r.get(k) is not None), None)
        columns.append({
            "name": k,
            "non_null": int(filled[k]),
            "pct_filled": round(100 * filled[k] / counts, 2) if counts else None,
            "example": example if not isinstance(example, str) or len(example) < 120 else example[:117] + "...",
        })

    value_counts = {}
    for c in columns:
        if not CATEGORICAL_HINT.search(c["name"]) or c["non_null"] == 0:
            continue
        k = c["name"]
        try:
            df = q(client, f"MATCH (n:Neuron) WHERE n.`{k}` IS NOT NULL "
                           f"RETURN n.`{k}` AS value, count(*) AS n ORDER BY n DESC")
        except Exception as ex:
            value_counts[k] = {"error": str(ex)}
            continue
        value_counts[k] = {"n_distinct": len(df), "top": df.head(15).astype({"value": str}).values.tolist()}

    return {
        "n_neuron": int(counts),
        "n_segment": n_segment,
        "columns": columns,
        "roi_flag_columns": len(roi_keys),
        "value_counts": value_counts,
    }


def describe_connections(client):
    sample = q(client, "MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) "
                       "RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, properties(e) AS props LIMIT 5")
    keys = sorted({k for p in sample["props"] for k in p})
    rows = []
    for _, r in sample.iterrows():
        props = {k: (v if not isinstance(v, str) or len(v) < 160 else v[:157] + "...") for k, v in r["props"].items()}
        rows.append({"bodyId_pre": int(r["bodyId_pre"]), "bodyId_post": int(r["bodyId_post"]), **props})
    return {"relationship": "ConnectsTo", "property_keys": keys, "sample": rows}


def describe_meta(client):
    meta = client.meta
    keep = {}
    for k, v in meta.items():
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 200):
            keep[k] = v
        else:
            keep[k] = f"<{type(v).__name__}, len={len(v) if hasattr(v, '__len__') else '?'}>"
    return keep


def print_report(rep):
    print(f"\n=== Dataset: {rep['dataset']} ===")
    print("\n-- :Meta (scalar fields) --")
    for k, v in rep["meta"].items():
        print(f"  {k}: {v}")

    n = rep["neurons"]
    seg = "n/a" if n["n_segment"] is None else f"{n['n_segment']:,}"
    print(f"\n-- :Neuron  ({n['n_neuron']:,} nodes; :Segment total {seg}) --")
    print(f"  {n['roi_flag_columns']} additional boolean ROI-membership columns not listed.")
    print(f"  {'column':<32}{'non-null':>12}{'% filled':>10}  example")
    for c in n["columns"]:
        print(f"  {c['name']:<32}{c['non_null']:>12,}{c['pct_filled']:>9.1f}%  {c['example']!r}")

    print("\n-- Value counts (categorical-looking columns, top 15) --")
    for k, vc in n["value_counts"].items():
        if "error" in vc:
            print(f"  {k}: ERROR {vc['error']}")
            continue
        top = ", ".join(f"{v}={c:,}" for v, c in vc["top"])
        print(f"  {k} ({vc['n_distinct']:,} distinct): {top}")

    e = rep["connections"]
    print(f"\n-- :ConnectsTo (Neuron -> Neuron) --")
    print(f"  relationship properties: {e['property_keys']}")
    for row in e["sample"]:
        print(f"  {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    token = get_token()
    datasets = list_datasets(token)
    print("Available datasets:")
    for name, info in datasets.items():
        extra = {k: info.get(k) for k in ("last-mod", "uuid", "description") if isinstance(info, dict) and info.get(k)}
        print(f"  {name}  {extra}")

    dataset = pick_dataset(datasets, args.dataset)
    print(f"\nSelected: {dataset}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{dataset.replace(':', '_').replace('/', '_')}.json"
    if cache.exists() and not args.refresh:
        print(f"(using cached report {cache.relative_to(ROOT)}; --refresh to re-query)")
        rep = json.loads(cache.read_text(encoding="utf-8"))
    else:
        client = Client(SERVER, dataset=dataset, token=token, progress=False)
        rep = {
            "dataset": dataset,
            "server_version": client.fetch_version(),
            "meta": describe_meta(client),
            "neurons": describe_neurons(client, client.all_rois),
            "connections": describe_connections(client),
        }
        cache.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print_report(rep)


if __name__ == "__main__":
    main()
