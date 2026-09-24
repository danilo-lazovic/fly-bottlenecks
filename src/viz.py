"""Interactive 3D view of each pathway: sensory sources, the min-cut neurons and the target cell, inside the CNS.

    python src/viz.py --dataset male-cns:v1.0   # -> reports/bottleneck_3d.html

Skeletons and ROI meshes come from neuPrint and are cached under data/raw/<dataset>/ (skeletons/, meshes/).
Everything else is read from data/processed. The page loads plotly.js from cdnjs; the data is inlined.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import load_neurons, processed_dir  # noqa: E402
from ingest import ROOT, cache_dir, get_client  # noqa: E402

VOXEL_UM = 0.008
MESH_ROIS = ["CentralBrain", "VNC", "Optic(L)", "Optic(R)"]
MESH_GRID_UM = 4.5          # vertex-clustering cell size for the context meshes
SKELETON_STRIDE = 6         # keep every n-th node along unbranched stretches
N_SOURCE_SAMPLE = 80
TEMPLATE = Path(__file__).resolve().parent / "templates" / "viewer.html"
OUT = ROOT / "reports" / "bottleneck_3d.html"

# Extra layer shown per pathway: the identified cell types that dominate the full-graph (0% share) cut.
KNOWN_TYPES = {"vision_to_giant_fiber": ["LPLC2", "LC4"]}
N_KNOWN_PER_TYPE = 15


class Cache:
    def __init__(self, dataset):
        self.dataset = dataset
        self.root = cache_dir(dataset)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_client(self.dataset)
        return self._client

    def skeleton(self, body_id):
        path = self.root / "skeletons" / f"{body_id}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.client.fetch_skeleton(int(body_id), format="pandas").to_parquet(path, index=False)
        return pd.read_parquet(path)

    def mesh(self, roi):
        path = self.root / "meshes" / f"{roi}.obj"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.client.fetch_roi_mesh(roi))
        return path.read_text()


def simplify_skeleton(sk):
    """Polyline coords (um) with NaN breaks; keeps roots, leaves, branch points and every n-th node."""
    ids = sk["rowId"].to_numpy()
    parent = sk["link"].to_numpy()
    pos = {r: i for i, r in enumerate(ids)}
    n_children = pd.Series(parent).value_counts()
    keep = np.array([(p == -1) or (n_children.get(r, 0) != 1) or (i % SKELETON_STRIDE == 0)
                     for i, (r, p) in enumerate(zip(ids, parent))])
    xyz = sk[["x", "y", "z"]].to_numpy() * VOXEL_UM

    anchor = np.empty(len(ids), dtype=np.int64)  # nearest kept ancestor-or-self, by row
    done = np.zeros(len(ids), bool)
    for _ in range(len(ids)):  # SWC rows are normally parent-first, so this finishes in one pass
        progressed = False
        for i in np.flatnonzero(~done):
            p = parent[i]
            if p == -1:
                anchor[i] = i
            elif done[pos[p]]:
                anchor[i] = i if keep[i] else anchor[pos[p]]
            else:
                continue
            done[i] = progressed = True
        if done.all() or not progressed:
            break

    segs = []
    for i in np.flatnonzero(keep):
        if parent[i] == -1:
            continue
        j = anchor[pos[parent[i]]]
        segs.append((xyz[i], xyz[j]))
    if not segs:
        return [], [], []
    a = np.array(segs)  # (m, 2, 3)
    pts = np.concatenate([a, np.full((len(a), 1, 3), np.nan)], axis=1).reshape(-1, 3)
    pts = np.round(pts)
    return [None if np.isnan(v) else float(v) for v in pts[:, 0]], \
           [None if np.isnan(v) else float(v) for v in pts[:, 1]], \
           [None if np.isnan(v) else float(v) for v in pts[:, 2]]


def decimate_obj(text, grid_um):
    verts, faces = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    v = np.asarray(verts) * VOXEL_UM
    f = np.asarray(faces)
    cell = np.floor(v / grid_um).astype(np.int64)
    _, cluster, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    cluster = cluster.ravel()
    centers = np.zeros((len(counts), 3))
    np.add.at(centers, cluster, v)
    centers /= counts[:, None]
    f2 = cluster[f]
    ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = np.unique(np.sort(f2[ok], axis=1), axis=0)
    return {"x": np.round(centers[:, 0]).tolist(), "y": np.round(centers[:, 1]).tolist(),
            "z": np.round(centers[:, 2]).tolist(), "i": f2[:, 0].tolist(), "j": f2[:, 1].tolist(),
            "k": f2[:, 2].tolist()}


def neuron_group(cache, neurons, idx, label):
    out = []
    for i in tqdm(idx, desc=label, leave=False):
        row = neurons.iloc[int(i)]
        x, y, z = simplify_skeleton(cache.skeleton(int(row["bodyId"])))
        out.append({"bodyId": int(row["bodyId"]), "type": row["type"] if isinstance(row["type"], str) else "untyped",
                    "superclass": row["superclass"] if isinstance(row["superclass"], str) else "",
                    "x": x, "y": y, "z": z})
    return out


def build_data(dataset, seed=0):
    rng = np.random.default_rng(seed)
    cache = Cache(dataset)
    neurons = load_neurons(dataset)
    report = json.loads((processed_dir(dataset) / "flow" / "flow_report.json").read_text())
    cuts = pd.read_parquet(processed_dir(dataset) / "flow" / "cuts.parquet")
    sets = pd.read_parquet(processed_dir(dataset) / "sets.parquet")

    meshes = {roi: decimate_obj(cache.mesh(roi), MESH_GRID_UM) for roi in MESH_ROIS}
    pathways = {}
    for name, rep in report["pathways"].items():
        main = cuts.query("pathway == @name and variant == @rep['main_cut']")["idx"].to_numpy()
        src_all = sets.query("set == @rep['source']")["idx"].to_numpy()
        src = rng.choice(src_all, size=min(N_SOURCE_SAMPLE, len(src_all)), replace=False)
        snk = neurons.index[neurons["bodyId"].isin(rep["sink_bodyIds"])].to_numpy()
        groups = {"sources": neuron_group(cache, neurons, src, f"{name} sources"),
                  "cut": neuron_group(cache, neurons, main, f"{name} cut"),
                  "sink": neuron_group(cache, neurons, snk, f"{name} sink")}
        if name in KNOWN_TYPES:
            full_cut = cuts.query("pathway == @name and variant == 'share0'")
            pick = []
            for t in KNOWN_TYPES[name]:
                ids = full_cut.loc[full_cut["type"] == t, "idx"].to_numpy()
                pick += rng.choice(ids, size=min(N_KNOWN_PER_TYPE, len(ids)), replace=False).tolist()
            groups["known"] = neuron_group(cache, neurons, np.array(pick), f"{name} known")
        a = rep["ablation"]
        pathways[name] = {
            "source": rep["source"], "n_sources": rep["n_sources"], "sink_types": rep["sink_types"],
            "cut_sizes": {k: v["n"] for k, v in rep["cuts"].items()},
            "main_cut": rep["main_cut"], "robustness": rep["robustness_jaccard_vs_main"],
            "ablation": {"k": a["k"], "remaining": a["remaining_flow_share"],
                         "random_mean": a["random_on_path"]["mean"], "random_sd": a["random_on_path"]["sd"],
                         "random_values": a["random_on_path"]["values"]},
            "known_types": KNOWN_TYPES.get(name, []),
            "known_counts": {t: int((cuts.query("pathway == @name and variant == 'share0'")["type"] == t).sum())
                             for t in KNOWN_TYPES.get(name, [])},
            "groups": groups,
        }
    return {"dataset": dataset, "meshes": meshes, "pathways": pathways,
            "reference": report["reference_all_descending"], "main_share": report["main_share"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    data = build_data(args.dataset)
    OUT.parent.mkdir(exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
