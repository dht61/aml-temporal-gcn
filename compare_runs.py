"""Check two runs of the same config are bitwise identical.  python compare_runs.py DIR_A DIR_B"""
import json
import sys
from pathlib import Path


def prints(d):
    d = Path(d)
    if (d / "iterations.jsonl").exists():
        rows = [json.loads(line) for line in (d / "iterations.jsonl").read_text().splitlines()]
        return {f"it{r['iteration']}_{k}": r[k] for r in rows for k in ("fp_init", "fp_final", "fp_preds")}
    return json.loads((d / "fingerprints.json").read_text())


a, b = prints(sys.argv[1]), prints(sys.argv[2])
bad = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
for k in sorted(set(a) | set(b)):
    print(f"{'OK  ' if k not in bad else 'DIFF'} {k} {a.get(k, '-')[:16]} {b.get(k, '-')[:16]}")
print(f"[det] {sys.argv[1]} vs {sys.argv[2]}: {'IDENTICAL' if not bad else f'{len(bad)} MISMATCH'}")
sys.exit(1 if bad else 0)
