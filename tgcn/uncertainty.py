"""MC-dropout and MC-AA sampling + acquisition functions (paper Sec. 3-4).

Samplers return one (T, n, C) tensor of class probabilities per graph.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import probs


@torch.no_grad()
def mc_dropout_samples(model, graphs, T):
    model.eval()
    model.mc_dropout = True
    try:
        return [torch.stack([probs(model(g.x, g.edge_index)) for _ in range(T)]) for g in graphs]
    finally:
        model.mc_dropout = False


def mc_aa_epsilons(T, eps_max):
    # Eq. 8: T steps of 2*eps_max/T from -eps_max (T=10, eps_max=0.1 -> -0.10 ... 0.08)
    beta = 2.0 * eps_max / T
    return beta * (torch.arange(T, dtype=torch.float64) - T / 2)


def mc_aa_samples(model, graphs, T, eps_max, masks):
    """MC-AA: treat every pool node as licit, take one FGSM gradient wrt the features,
    then T forward passes at x + eps * sign(grad). masks[i] = pool nodes of graphs[i]."""
    model.eval()
    eps = mc_aa_epsilons(T, eps_max).float()
    out = []
    for g, m in zip(graphs, masks):
        k = int(m.sum())
        if k == 0:  # nothing to score here
            s = torch.zeros_like(g.x)
        else:
            x = g.x.clone().requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                logits = model(x, g.edge_index)
                target = torch.zeros(k, dtype=torch.long)
                loss = F.nll_loss(F.log_softmax(logits[m], dim=-1), target, reduction="sum")
                (grad,) = torch.autograd.grad(loss, x)
            s = grad.sign()
        with torch.no_grad():
            out.append(torch.stack([probs(model(g.x + e * s, g.edge_index)) for e in eps]))
    return out


def _xlogx(p):
    return torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))


def scores(samples, acquisition):
    """samples: (T, N, C). Higher = queried first."""
    mean = samples.mean(0)
    if acquisition == "entropy":
        return -_xlogx(mean).sum(1)
    if acquisition == "bald":
        return -_xlogx(mean).sum(1) + _xlogx(samples).sum(2).mean(0)
    if acquisition == "variation_ratio":
        return 1.0 - mean.max(1).values
    if acquisition == "mean_std":
        var = (samples ** 2).mean(0) - mean ** 2
        return var.clamp_min(0).sqrt().mean(1)
    raise ValueError(acquisition)


def top_b(score, b):
    order = np.argsort(-score.numpy(), kind="stable")  # stable so ties are deterministic
    return order[:b]
