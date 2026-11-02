"""Download Elliptic (PyG mirror) and build data/processed/elliptic_local.pt + stats.json."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

URL = "https://data.pyg.org/datasets/elliptic"
FILES = ["elliptic_txs_features.csv.zip", "elliptic_txs_edgelist.csv.zip",
         "elliptic_txs_classes.csv.zip"]
N_LOCAL = 93


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_zip_csv(path, **kw):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        assert len(names) == 1, (path, z.namelist())
        with z.open(names[0]) as f:
            return pd.read_csv(f, **kw)


def col_stats(x):
    mu, sd = x.astype(np.float64).mean(0), x.astype(np.float64).std(0)
    return {"max_abs_col_mean": float(np.abs(mu).max()), "col_std_min": float(sd.min()),
            "col_std_max": float(sd.max()), "col_std_median": float(np.median(sd))}


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = raw / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {name} exists ({dst.stat().st_size} B)")
            continue
        tmp = dst.with_suffix(".part")
        print(f"[get] {URL}/{name}")
        urllib.request.urlretrieve(f"{URL}/{name}", tmp)
        tmp.rename(dst)
    return {name: sha256(raw / name) for name in FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "data"))
    args = ap.parse_args()
    root = Path(args.root)
    raw, out = root / "raw", root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    hashes = download(raw)

    feat = read_zip_csv(raw / FILES[0], header=None)
    edges = read_zip_csv(raw / FILES[1])
    cls = read_zip_csv(raw / FILES[2])
    assert feat.shape[1] == 2 + 93 + 72, feat.shape  # txId, time_step, 93 local, 72 aggregated
    print("features", feat.shape, "edges", edges.shape, "classes", cls.shape)

    txid = feat[0].to_numpy(np.int64)
    ts = feat[1].to_numpy(np.int64)
    x = feat.iloc[:, 2:2 + N_LOCAL].to_numpy(np.float32)
    label = cls["class"].astype(str).map({"1": 1, "2": 0, "unknown": -1})
    assert label.notna().all(), cls["class"].unique()
    cls_by_tx = dict(zip(cls["txId"].to_numpy(np.int64), label.to_numpy(np.int64)))
    assert len(cls_by_tx) == len(txid)
    y = np.array([cls_by_tx[t] for t in txid], dtype=np.int64)

    gid = {t: i for i, t in enumerate(txid)}
    src = edges["txId1"].map(gid).to_numpy(np.int64)
    dst = edges["txId2"].map(gid).to_numpy(np.int64)
    cross = int((ts[src] != ts[dst]).sum())

    graphs, per_t = [], []
    for t in range(1, int(ts.max()) + 1):
        idx = np.where(ts == t)[0]  # raw file order
        local = np.full(len(txid), -1, dtype=np.int64)
        local[idx] = np.arange(len(idx))
        keep = (ts[src] == t) & (ts[dst] == t)
        ei = np.stack([local[src[keep]], local[dst[keep]]])
        yt = y[idx]
        graphs.append({"t": t, "x": torch.from_numpy(x[idx]), "y": torch.from_numpy(yt),
                       "edge_index": torch.from_numpy(ei), "txid": torch.from_numpy(txid[idx])})
        per_t.append({"t": t, "nodes": int(len(idx)), "edges": int(keep.sum()),
                      "illicit": int((yt == 1).sum()), "licit": int((yt == 0).sum()),
                      "unknown": int((yt == -1).sum()),
                      "sorted_by_txid": bool(np.all(np.diff(txid[idx]) > 0))})

    def labelled(lo, hi):
        return sum(r["illicit"] + r["licit"] for r in per_t if lo <= r["t"] <= hi)

    stats = {
        "raw_sha256": hashes,
        "n_nodes": int(len(txid)), "n_edges": int(len(src)), "cross_timestep_edges": cross,
        "n_illicit": int((y == 1).sum()), "n_licit": int((y == 0).sum()),
        "n_unknown": int((y == -1).sum()),
        "time_step_col_is_sorted": bool(np.all(np.diff(ts) >= 0)),
        # check whether the released features are already standardised
        "local_col_stats_all": col_stats(x),
        "local_col_stats_train_t34": col_stats(x[ts <= 34]),
        "labelled_split_t34": [labelled(1, 34), labelled(35, 49)],
        "labelled_split_t35": [labelled(1, 35), labelled(36, 49)],
        "paper_labelled_train_test": [29894, 16670],
        "per_timestep": per_t,
    }
    torch.save({"meta": {k: v for k, v in stats.items() if k != "per_timestep"},
                "graphs": graphs}, out / "elliptic_local.pt")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_timestep"}, indent=1))


if __name__ == "__main__":
    main()
