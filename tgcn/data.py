"""Elliptic as 49 per-timestep graphs.

Uses the 93 local features only (no time step, no aggregated features).
Labels: 1 = illicit, 0 = licit, -1 = unknown (never used in loss or metrics).
Node order within a timestep is the raw file order - this is the sequence the LSTM sees.
"""
from dataclasses import dataclass

import torch

N_LOCAL = 93
ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass
class Graph:
    t: int
    x: torch.Tensor           # (n, 93)
    y: torch.Tensor           # (n,)
    edge_index: torch.Tensor  # (2, e), local ids, txId1 -> txId2
    txid: torch.Tensor        # (n,)


def load_graphs(processed_path, edges="directed", standardize="none", train_last_t=34):
    """standardize: 'none', 'all' (z-score over all nodes) or 'train' (train timesteps only)."""
    blob = torch.load(processed_path)
    if standardize != "none":
        src = [g["x"] for g in blob["graphs"] if standardize == "all" or g["t"] <= train_last_t]
        if standardize not in ("all", "train"):
            raise ValueError(standardize)
        allx = torch.cat(src).double()
        mu, sd = allx.mean(0), allx.std(0, unbiased=False)
        sd[sd == 0] = 1.0
        for g in blob["graphs"]:
            g["x"] = ((g["x"].double() - mu) / sd).float()
    graphs = []
    for g in blob["graphs"]:
        ei = g["edge_index"]
        if edges == "reversed":
            ei = ei.flip(0)
        elif edges == "undirected":
            ei = torch.cat([ei, ei.flip(0)], dim=1)
        elif edges != "directed":
            raise ValueError(edges)
        graphs.append(Graph(g["t"], g["x"], g["y"], ei.contiguous(), g["txid"]))
    return graphs, blob["meta"]


def split(graphs, train_last_t=34):
    # temporal split from Weber et al.: train on t <= 34, test on the rest
    train = [g for g in graphs if g.t <= train_last_t]
    test = [g for g in graphs if g.t > train_last_t]
    return train, test


def labelled_mask(g):
    return g.y >= 0
