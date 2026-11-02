"""Tables 1, 4, 5 and the variant sweep from results/*/<run>_seed<i>/metrics.json.

    python summarize_supervised.py --sweep sweep2 --ablation ablation_pernode --md results/summary.md
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ["accuracy", "precision", "recall", "f1", "macro_f1"]
PAPER_T1 = {"accuracy": 97.7, "precision": 92.7, "recall": 71.3, "f1": 80.6}
PAPER_T4 = {"model0": 97.77, "model1": 97.66, "model2": 97.76, "model3": 97.57, "model4": 97.44,
            "model5": 97.51, "model6": 97.63, "model7": 96.65}
PAPER_T5 = {"K3": 97.77, "K2": 97.71, "K1": 97.75}
LAYERS = {"model0": "LSTM/TAG/TAG", "model1": "Lin/TAG/TAG", "model2": "LSTM/Lin/TAG",
          "model3": "LSTM/TAG/Lin", "model4": "LSTM/Lin/Lin", "model5": "Lin/Lin/Lin",
          "model6": "LSTM/TAG", "model7": "TAG/TAG"}


def load(group_dir):
    """{variant: {seed: overall metrics + train_seconds}} for every finished run."""
    runs = defaultdict(dict)
    for f in sorted(Path(group_dir).glob("*_seed*/metrics.json")):
        m = re.fullmatch(r"(.+)_seed(\d+)", f.parent.name)
        d = json.loads(f.read_text())
        runs[m.group(1)][int(m.group(2))] = {**d["overall"], "train_seconds": d["train_seconds"]}
    return runs


def fmt(vals):
    vals = [100 * v for v in vals]
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.2f} ± {sd:.2f} ({min(vals):.2f}–{max(vals):.2f})"


def table1(runs):
    seeds = runs.get("model0", {})
    if not seeds:
        return ["Table 1: no finished model0 runs in results/table1/."]
    rows = list(seeds.values())
    out = [f"**Table 1** (model0, K=3, {len(rows)} seeds: {sorted(seeds)}), % mean ± std (min–max)", "",
           "| Metric | Ours | Paper |", "|---|---|---|"]
    for k in KEYS:
        out.append(f"| {k} | {fmt([r[k] for r in rows])} | {PAPER_T1.get(k, '–')} |")
    collapsed = [s for s, r in seeds.items() if r["f1"] == 0]
    out += ["", f"Seeds predicting all-licit (F1 = 0): {collapsed or 'none'}. "
            f"Mean train time {st.mean(r['train_seconds'] for r in rows) / 60:.1f} min.",
            "Per seed (acc / prec / rec / F1):"]
    for s in sorted(seeds):
        r = seeds[s]
        out.append(f"- seed {s}: " + " / ".join(f"{100 * r[k]:.2f}" for k in KEYS[:4]))
    return out


def ablation(t1, ab):
    """Model-0 / K=3 rows use the ablation's own model0 runs if it has them, else table1 seeds 0-4."""
    base = ab.get("model0") or {s: r for s, r in t1.get("model0", {}).items() if s < 5}
    out = ["**Table 4** (accuracy %, seeds 0–4)", "", "| Model | Layers | Ours | Paper |",
           "|---|---|---|---|"]
    for name in PAPER_T4:
        seeds = base if name == "model0" else ab.get(name, {})
        ours = fmt([r["accuracy"] for r in seeds.values()]) if seeds else "pending"
        out.append(f"| {name} | {LAYERS[name]} | {ours} | {PAPER_T4[name]} |")
    out += ["", "**Table 5** (model0 accuracy % vs K, seeds 0–4)", "", "| K | Ours | Paper |",
            "|---|---|---|"]
    for k in PAPER_T5:
        seeds = base if k == "K3" else ab.get(f"model0_{k}", {})
        ours = fmt([r["accuracy"] for r in seeds.values()]) if seeds else "pending"
        out.append(f"| {k[1:]} | {ours} | {PAPER_T5[k]} |")
    return out


def sweep(t1, sw):
    """Each variant vs the faithful model0 on the same seeds; dF1 is the paired mean difference."""
    out = ["**Variant sweep** (% mean ± std (min–max); ΔF1 vs model0 on the same seeds)", "",
           "| Variant | Seeds | Acc | Prec | Rec | F1 | ΔF1 |", "|---|---|---|---|---|---|---|"]
    base = t1.get("model0", {})
    for name in ["model0"] + sorted(sw):
        seeds = {s: r for s, r in base.items() if s < 3} if name == "model0" else sw[name]
        if not seeds:
            continue
        paired = [100 * (r["f1"] - base[s]["f1"]) for s, r in seeds.items() if s in base]
        d = f"{st.mean(paired):+.2f}" if paired and name != "model0" else "–"
        cells = [fmt([r[k] for r in seeds.values()]) for k in KEYS[:4]]
        out.append(f"| {name} | {sorted(seeds)} | " + " | ".join(cells) + f" | {d} |")
    out.append(f"| paper | 1 run | {PAPER_T1['accuracy']} | {PAPER_T1['precision']} | "
               f"{PAPER_T1['recall']} | {PAPER_T1['f1']} | |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--md", help="also write the tables to this markdown file")
    ap.add_argument("--sweep", default="sweep", help="results subdir holding the variant sweep")
    ap.add_argument("--ablation", default="ablation", help="results subdir holding Tables 4-5 runs")
    a = ap.parse_args()
    t1 = load(Path(a.results) / "table1")
    lines = table1(t1)
    ab = load(Path(a.results) / a.ablation)
    if ab:
        lines += [""] + ablation(t1, ab)
    sw = load(Path(a.results) / a.sweep)
    if sw:
        lines += [""] + sweep(t1, sw)
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
