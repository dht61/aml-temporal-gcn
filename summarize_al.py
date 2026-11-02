"""Active learning tables (2, 3), Fig. 4 and Fig. 3.

Table 3 = two-sided Wilcoxon on the paired per-iteration accuracies vs random.

    python summarize_al.py --al al_pernode --sup sweep2/pn --sup_seq table1/model0
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parent
ACQ = ["bald", "entropy", "variation_ratio", "mean_std"]
ACQ_NAME = {"bald": "BALD", "entropy": "Entropy", "variation_ratio": "Variation ratio",
            "mean_std": "Mean STD"}
PAPER_T2 = {"bald": (28.07, 27.1), "entropy": (28.9, 27.3), "variation_ratio": (28.9, 28.3),
            "mean_std": (28.68, 27.09)}  # minutes, (MC-AA, MC-dropout), GPU
PAPER_T3 = {"bald": (0.0009, 0.0217), "entropy": (0.2552, 0.309), "variation_ratio": (0.266, 0.0071),
            "mean_std": (0.5507, 0.0112)}  # p, (MC-AA vs random, MC-dropout vs random)
C = {"mc_aa": "#2a78d6", "mc_dropout": "#eb6834", "random": "#1baf7a"}
LABEL = {"mc_aa": "MC-AA", "mc_dropout": "MC-dropout", "random": "Random"}
MARK = {"mc_aa": "o", "mc_dropout": "s", "random": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_arm(d):
    f = d / "iterations.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    return rows if (d / "done.json").exists() else None


def sup_runs(prefix):
    """metrics.json of every <prefix>_seed* run, e.g. results/sweep2/pn_seed3."""
    group, stem = prefix.rsplit("/", 1)
    return [json.loads(p.read_text()) for p in sorted((ROOT / "results" / group).glob(f"{stem}_seed*/metrics.json"))]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig4(arms, sup_acc, metric, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, q in zip(axes.flat, ACQ):
        style(ax)
        for m in ("mc_aa", "mc_dropout", "random"):
            rows = arms.get((m, "random" if m == "random" else q))
            if not rows:
                continue
            x = [100 * r["frac_pool"] for r in rows]
            y = [100 * r[metric] for r in rows]
            ax.plot(x, y, color=C[m], linewidth=2, marker=MARK[m], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[m])
        if sup_acc is not None:
            ax.axhline(sup_acc, color=INK2, linestyle="--", linewidth=1)
            ax.annotate("fully supervised", (2, sup_acc), xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=INK2)
        ax.set_title(ACQ_NAME[q], fontsize=11, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("% of training pool labelled", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"test {'accuracy' if metric == 'accuracy' else 'illicit F1'} (%)", fontsize=9,
                      color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig3(runs_by_name, out):
    """Illicit nodes per test timestep: actual vs correctly detected (mean over seeds)."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    style(ax)
    first = next(iter(runs_by_name.values()))[0]["per_timestep"]
    t = [r["t"] for r in first]
    series = [("Actual illicit", [r["illicit"] for r in first], C["mc_aa"])]
    for (name, runs), color in zip(runs_by_name.items(), (C["mc_dropout"], C["random"])):
        det = np.mean([[r["illicit_correct"] for r in m["per_timestep"]] for m in runs], axis=0)
        series.append((f"Detected: {name} ({len(runs)} seeds)", det, color))
    for name, y, color in series:
        ax.plot(t, y, color=color, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE,
                markeredgewidth=1, label=name)
    ax.axvline(43, color=INK2, linestyle=":", linewidth=1)
    ax.annotate("dark-market shutdown", (43, 0.55 * ax.get_ylim()[1]), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xticks(t)
    ax.set_xlabel("time step (test set)", fontsize=9, color=INK2)
    ax.set_ylabel("illicit transactions", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.36, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--al", default="al_pernode", help="results subdir with seed0/<method>_<acq>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sup", default="sweep2/pn", help="fully supervised runs for the Fig. 4 reference")
    ap.add_argument("--sup_seq", default="table1/model0", help="second Fig. 3 series (literal reading)")
    a = ap.parse_args()
    al_dir = ROOT / "results" / a.al / f"seed{a.seed}"
    figs = ROOT / "results" / a.al / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    arms = {}
    for m, q in [(m, q) for m in ("mc_aa", "mc_dropout") for q in ACQ] + [("random", "random")]:
        rows = load_arm(al_dir / f"{m}_{q}")
        if rows:
            arms[(m, q)] = rows
    sup = sup_runs(a.sup)
    sup_acc = 100 * float(np.mean([r["overall"]["accuracy"] for r in sup])) if sup else None
    sup_f1 = 100 * float(np.mean([r["overall"]["f1"] for r in sup])) if sup else None

    lines = [f"**Active learning** ({a.al}, seed {a.seed}; {len(arms)}/9 arms finished). "
             f"Fully supervised reference ({a.sup}, {len(sup)} seeds): acc {sup_acc:.2f}, F1 {sup_f1:.2f}"
             if sup else f"**Active learning** ({a.al}, seed {a.seed}; {len(arms)}/9 arms finished)", ""]

    # Table 2: end-to-end runtime per arm (sum of training + acquisition), minutes
    lines += ["**Table 2** runtime (min, ours on 8 CPU threads / paper on GPU)", "",
              "| Acquisition | MC-AA ours | MC-AA paper | MC-dropout ours | MC-dropout paper |", "|---|---|---|---|---|"]

    def minutes(rows):
        return sum(r["train_seconds"] + r.get("acquire_seconds", 0.0) for r in rows) / 60

    for q in ACQ:
        cells = [f"{minutes(arms[(m, q)]):.1f}" if (m, q) in arms else "pending" for m in ("mc_aa", "mc_dropout")]
        lines.append(f"| {ACQ_NAME[q]} | {cells[0]} | {PAPER_T2[q][0]} | {cells[1]} | {PAPER_T2[q][1]} |")
    if ("random", "random") in arms:
        lines.append(f"| Random | {minutes(arms[('random', 'random')]):.1f} | – | | |")

    # Table 3: Wilcoxon p-values on paired per-iteration curves vs random
    rnd = arms.get(("random", "random"))
    lines += ["", "**Table 3** Wilcoxon p (two-sided, paired per-iteration accuracy vs random; F1 in brackets)", "",
              "| Acquisition | MC-AA ours | MC-AA paper | MC-dropout ours | MC-dropout paper |", "|---|---|---|---|---|"]
    for q in ACQ:
        cells = []
        for m in ("mc_aa", "mc_dropout"):
            rows = arms.get((m, q))
            if not rows or not rnd:
                cells.append("pending")
                continue
            ps = []
            for k in ("accuracy", "f1"):
                x, y = np.array([r[k] for r in rows]), np.array([r[k] for r in rnd])
                ps.append(wilcoxon(x, y).pvalue if np.any(x != y) else float("nan"))
            cells.append(f"{ps[0]:.4f} ({ps[1]:.4f})")
        lines.append(f"| {ACQ_NAME[q]} | {cells[0]} | {PAPER_T3[q][0]} | {cells[1]} | {PAPER_T3[q][1]} |")

    # Per-arm curves and the "matches fully supervised after X%" claim (paper: BALD, 20% = iteration 2)
    lines += ["", "**Curves** (accuracy % / F1 % per iteration; first % of pool where accuracy ≥ fully supervised)", ""]
    for (m, q), rows in arms.items():
        hit = next((100 * r["frac_pool"] for r in rows if sup_acc and 100 * r["accuracy"] >= sup_acc), None)
        curve = " ".join(f"{100 * r['accuracy']:.2f}/{100 * r['f1']:.1f}" for r in rows)
        lines.append(f"- {LABEL[m]} {q}: reaches supervised at "
                     f"{f'{hit:.0f}%' if hit is not None else 'never'}; {curve}")

    if arms:
        fig4(arms, sup_acc, "accuracy", figs / "fig4_accuracy.png")
        fig4(arms, sup_f1, "f1", figs / "fig4_f1.png")
    fig3_runs = {k: v for k, v in (("per_node", sup), ("literal seq_nodes", sup_runs(a.sup_seq))) if v}
    if fig3_runs:
        fig3(fig3_runs, figs / "fig3_illicit_per_timestep.png")
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / a.al / "summary.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
