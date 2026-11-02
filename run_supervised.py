"""Supervised temporal-GCN on Elliptic (Table 1, Fig. 3, Tables 4-5).

Defaults follow the paper as literally as possible; the ambiguous bits are flags.
"""
from tgcn.determinism import reexec_with_hashseed

reexec_with_hashseed()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tgcn.data import labelled_mask, load_graphs, split  # noqa: E402
from tgcn.determinism import (enable_determinism, env_report, fingerprint_array,  # noqa: E402
                              fingerprint_model)
from tgcn.model import ABLATION  # noqa: E402
from tgcn.train import build_model, evaluate, train_model  # noqa: E402

ROOT = Path(__file__).resolve().parent


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/processed/elliptic_local.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default="model0", choices=sorted(ABLATION))
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm_mode", default="seq_nodes", choices=["seq_nodes", "per_node"])
    ap.add_argument("--output", default="log_softmax", choices=["log_softmax", "softmax_nll"])
    ap.add_argument("--edges", default="directed", choices=["directed", "undirected", "reversed"])
    ap.add_argument("--train_last_t", type=int, default=34)
    ap.add_argument("--standardize", default="none", choices=["none", "all", "train"])
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--class_weights", default="none", help="licit,illicit e.g. 0.3,0.7")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    a = get_args()
    out = Path(a.out)
    if (out / "metrics.json").exists() and not a.force:
        print(f"[skip] {out}/metrics.json exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    enable_determinism(a.seed, a.threads)
    cfg = {**vars(a), "layers": ABLATION[a.model]}

    graphs, meta = load_graphs(a.data, a.edges, a.standardize, a.train_last_t)
    train, test = split(graphs, a.train_last_t)
    masks = [labelled_mask(g) for g in train]
    cfg["n_train_labelled"] = int(sum(int(m.sum()) for m in masks))
    cfg["n_test_labelled"] = int(sum(int(labelled_mask(g).sum()) for g in test))
    cfg["env"] = env_report()
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1), flush=True)

    model = build_model(cfg)
    fp = {"init": fingerprint_model(model)}
    print(f"[fingerprint] init {fp['init']}", flush=True)

    hist_f = open(out / "history.jsonl", "w")

    def log(row):
        print(json.dumps(row), flush=True)
        hist_f.write(json.dumps(row) + "\n")
        hist_f.flush()

    t0 = time.time()
    train_model(model, train, masks, cfg, log=log, eval_fn=lambda m: evaluate(m, test))
    train_s = time.time() - t0
    res = evaluate(model, test)
    fp["final"] = fingerprint_model(model)
    fp["preds"] = fingerprint_array(res["y_pred"], res["p_illicit"])
    print(f"[fingerprint] final {fp['final']} preds {fp['preds']}", flush=True)

    torch.save(model.state_dict(), out / "model.pt")
    np.savez(out / "preds.npz", y_true=res["y_true"], y_pred=res["y_pred"],
             p_illicit=res["p_illicit"])
    (out / "fingerprints.json").write_text(json.dumps(fp, indent=1))
    (out / "metrics.json").write_text(json.dumps(
        {"overall": res["overall"], "per_timestep": res["per_timestep"], "train_seconds": train_s},
        indent=1))
    print("[result]", json.dumps(res["overall"]), f"train {train_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
