"""Training / evaluation used by both the supervised runs and the AL loop.

One optimiser step per timestep graph (in temporal order), Adam lr 1e-3, 50 epochs, no early stopping.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .model import TemporalGCN, out_log_probs, probs


def build_model(cfg):
    return TemporalGCN(layers=cfg["layers"], in_dim=93, hidden=cfg["hidden"], n_classes=2,
                       K=cfg["K"], dropout=cfg["dropout"], lstm_mode=cfg["lstm_mode"])


def train_model(model, graphs, masks, cfg, log=None, eval_fn=None):
    # masks[i] marks which nodes of graphs[i] have a usable label
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w = cfg["class_weights"]
    w = None if w in (None, "none") else torch.tensor([float(v) for v in w.split(",")])
    history = []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for g, m in zip(graphs, masks):
            k = int(m.sum())
            if k == 0:
                continue
            logits = model(g.x, g.edge_index)
            loss = F.nll_loss(out_log_probs(logits[m], cfg["output"]), g.y[m], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * k
            n += k
        row = {"epoch": ep, "train_loss": tot / max(n, 1)}
        if eval_fn is not None and (ep % cfg.get("eval_every", 10) == 0 or ep == cfg["epochs"]):
            row.update({f"test_{k}": v for k, v in eval_fn(model)["overall"].items()})
        history.append(row)
        if log is not None:
            log(row)
    return history


@torch.no_grad()
def predict(model, graphs):
    model.eval()
    out = []
    for g in graphs:
        out.append(probs(model(g.x, g.edge_index)))
    return out


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary",
                                                 zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n": int(len(y_true))}


def evaluate(model, graphs):
    # illicit is the positive class; labelled nodes only
    P = predict(model, graphs)
    ys, yh, pi, per_t = [], [], [], []
    for g, p in zip(graphs, P):
        m = g.y >= 0
        y = g.y[m].numpy()
        pred = p[m].argmax(1).numpy()
        ys.append(y)
        yh.append(pred)
        pi.append(p[m, 1].numpy())
        per_t.append({"t": g.t, "illicit": int((y == 1).sum()),
                      "illicit_correct": int(((y == 1) & (pred == 1)).sum()),
                      "licit": int((y == 0).sum()), **metrics(y, pred)})
    y_true, y_pred = np.concatenate(ys), np.concatenate(yh)
    return {"overall": metrics(y_true, y_pred), "per_timestep": per_t,
            "y_true": y_true, "y_pred": y_pred, "p_illicit": np.concatenate(pi)}
