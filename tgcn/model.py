"""Temporal-GCN (Eq. 16) and the Table 4 ablation variants.

    H1 = ReLU(LSTM(X)); H2 = ReLU(TAGCN(H1, E)); H3 = Softmax(TAGCN(H2, E))

hidden 50, K=3, dropout 0.7 after each hidden layer. Uses PyG 2.0.4's TAGConv (one bias per hop).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv

# Table 4: (layer 1, layer 2, layer 3); model6/7 are 2-layer
ABLATION = {
    "model0": ("lstm", "tag", "tag"),
    "model1": ("linear", "tag", "tag"),
    "model2": ("lstm", "linear", "tag"),
    "model3": ("lstm", "tag", "linear"),
    "model4": ("lstm", "linear", "linear"),
    "model5": ("linear", "linear", "linear"),
    "model6": ("lstm", "tag"),
    "model7": ("tag", "tag"),
}


class LSTMLayer(nn.Module):
    """seq_nodes: the whole node matrix as one sequence (what nn.LSTM does with 2-D input on torch>=1.11).
    per_node: every node is its own length-1 sequence."""

    def __init__(self, in_dim, out_dim, mode):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, out_dim)
        self.mode = mode

    def forward(self, x):
        if self.mode == "seq_nodes":
            out, _ = self.lstm(x)
        elif self.mode == "per_node":
            out, _ = self.lstm(x.unsqueeze(0))
            out = out.squeeze(0)
        else:
            raise ValueError(self.mode)
        return out


class TemporalGCN(nn.Module):
    def __init__(self, layers=ABLATION["model0"], in_dim=93, hidden=50, n_classes=2, K=3,
                 dropout=0.7, lstm_mode="seq_nodes"):
        super().__init__()
        dims = [in_dim] + [hidden] * (len(layers) - 1) + [n_classes]
        self.kinds = tuple(layers)
        mods = []
        for kind, d_in, d_out in zip(self.kinds, dims[:-1], dims[1:]):
            if kind == "lstm":
                mods.append(LSTMLayer(d_in, d_out, lstm_mode))
            elif kind == "tag":
                mods.append(TAGConv(d_in, d_out, K=K))
            elif kind == "linear":
                mods.append(nn.Linear(d_in, d_out))
            else:
                raise ValueError(kind)
        self.layers = nn.ModuleList(mods)
        self.dropout = dropout
        self.mc_dropout = False  # set True to keep dropout on at eval time

    def forward(self, x, edge_index):
        h = x
        last = len(self.layers) - 1
        for i, (kind, layer) in enumerate(zip(self.kinds, self.layers)):
            h = layer(h, edge_index) if kind == "tag" else layer(h)
            if i < last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training or self.mc_dropout)
        return h


def out_log_probs(logits, output):
    # softmax_nll = probabilities fed straight into NLLLoss (literal reading of the paper), kept as a variant
    if output == "log_softmax":
        return F.log_softmax(logits, dim=-1)
    if output == "softmax_nll":
        return F.softmax(logits, dim=-1)
    raise ValueError(output)


def probs(logits):
    return F.softmax(logits, dim=-1)
