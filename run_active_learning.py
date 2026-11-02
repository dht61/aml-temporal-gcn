"""Pool-based active learning (Sec. 6.2, Fig. 4, Tables 2-3).

Start with 2000 random labels, then each round: retrain from scratch, score the remaining pool,
label the top 2000. Repeats until the pool is empty. All arms share the first batch and the
per-round init seed, so they only differ in what they query.
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
                              fingerprint_model, seed_all)
from tgcn.model import ABLATION  # noqa: E402
from tgcn.train import build_model, evaluate, train_model  # noqa: E402
from tgcn.uncertainty import mc_aa_samples, mc_dropout_samples, scores, top_b  # noqa: E402

ROOT = Path(__file__).resolve().parent


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True, choices=["mc_dropout", "mc_aa", "random"])
    ap.add_argument("--acquisition", required=True,
                    choices=["bald", "entropy", "variation_ratio", "mean_std", "random"])
    ap.add_argument("--data", default=str(ROOT / "data/processed/elliptic_local.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--max_iters", type=int, default=0, help="0 = until the pool is empty")
    ap.add_argument("--T_dropout", type=int, default=50)
    ap.add_argument("--T_aa", type=int, default=10)
    ap.add_argument("--eps_max", type=float, default=0.1)
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
    a = ap.parse_args()
    if (a.method == "random") != (a.acquisition == "random"):
        ap.error("random sampling is --method random --acquisition random")
    return a


def main():
    a = get_args()
    out = Path(a.out)
    if (out / "done.json").exists():
        print(f"[skip] {out}/done.json exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    enable_determinism(a.seed, a.threads)
    cfg = {**vars(a), "layers": ABLATION[a.model]}

    graphs, _ = load_graphs(a.data, a.edges, a.standardize, a.train_last_t)
    train, test = split(graphs, a.train_last_t)
    offsets = np.cumsum([0] + [g.x.shape[0] for g in train])
    pool_all = np.concatenate([offsets[i] + np.where(labelled_mask(g).numpy())[0]
                               for i, g in enumerate(train)])
    y_all = np.concatenate([g.y.numpy() for g in train])
    cfg["pool_size"] = int(len(pool_all))
    cfg["env"] = env_report()
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1), flush=True)

    def to_masks(ids):
        flat = torch.zeros(int(offsets[-1]), dtype=torch.bool)
        flat[torch.from_numpy(ids)] = True
        return [flat[offsets[i]:offsets[i + 1]] for i in range(len(train))]

    def rng(it):
        return np.random.default_rng([a.seed, it])

    # resume from the newest post-acquisition state
    done_its = sorted(int(p.stem.split("it")[1]) for p in out.glob("state_it*.npz"))
    rows_path = out / "iterations.jsonl"
    if done_its:
        k = done_its[-1]
        labelled = np.load(out / f"state_it{k}.npz")["labelled"]
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        rows = [r for r in rows if r["iteration"] <= k]
        it = k + 1
        print(f"[resume] from state_it{k}.npz, {len(labelled)} labelled", flush=True)
    else:
        labelled = np.sort(rng(0).choice(pool_all, size=a.batch, replace=False))
        rows, it = [], 0
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    while True:
        t0 = time.time()
        seed_all(a.seed * 1000 + it)  # same init for every arm at a given iteration
        model = build_model(cfg)
        fp_init = fingerprint_model(model)
        train_model(model, train, to_masks(labelled), cfg)
        t_train = time.time() - t0
        res = evaluate(model, test)
        row = {"iteration": it, "n_labelled": int(len(labelled)),
               "frac_pool": len(labelled) / len(pool_all),
               "n_illicit_labelled": int((y_all[labelled] == 1).sum()),
               **res["overall"], "train_seconds": t_train,
               "fp_init": fp_init, "fp_final": fingerprint_model(model),
               "fp_preds": fingerprint_array(res["y_pred"], res["p_illicit"])}

        remaining = np.setdiff1d(pool_all, labelled)
        if len(remaining) == 0 or (a.max_iters and it + 1 >= a.max_iters):
            torch.save(model.state_dict(), out / f"model_it{it}.pt")
            rows.append(row)
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("[iter]", json.dumps(row), flush=True)
            break

        t1 = time.time()
        b = min(a.batch, len(remaining))
        if a.method == "random":
            new = rng(it + 1).choice(remaining, size=b, replace=False)
        else:
            if a.method == "mc_dropout":
                samples = mc_dropout_samples(model, train, a.T_dropout)
            else:
                samples = mc_aa_samples(model, train, a.T_aa, a.eps_max, to_masks(remaining))
            s = scores(torch.cat(samples, dim=1)[:, torch.from_numpy(remaining)], a.acquisition)
            sel = top_b(s, b)
            new = remaining[sel]
            row["score_selected_mean"] = float(s[torch.from_numpy(sel)].mean())
            row["score_pool_mean"] = float(s.mean())
        row["acquire_seconds"] = time.time() - t1
        row["n_illicit_queried"] = int((y_all[new] == 1).sum())
        labelled = np.sort(np.concatenate([labelled, new]))

        np.savez(out / f"state_it{it}.npz", labelled=labelled, new=np.sort(new))
        torch.save(model.state_dict(), out / f"model_it{it}.pt")
        rows.append(row)
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("[iter]", json.dumps(row), flush=True)
        it += 1

    (out / "done.json").write_text(json.dumps({"iterations": len(rows)}))


if __name__ == "__main__":
    main()
