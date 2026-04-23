"""
analyze_results.py — break down Lean agreement by ground-truth polarity.

A key finding from our benchmark study: LEMO's ground-truth labels use
closed-world reasoning (an answer is F if the query cannot be derived),
while Lean 4 uses classical logic (F requires an explicit proof of ¬P).
We therefore report Lean agreement separately for T and F cases.
"""

import csv
from pathlib import Path
from collections import defaultdict

path = Path(__file__).parent / "benchmark_lean_results.csv"
rows = list(csv.DictReader(open(path)))

# Group by (split, gt_answer)
buckets = defaultdict(list)
for r in rows:
    buckets[(r["split"], r["gt_answer"])].append(r)

splits = sorted({r["split"] for r in rows})
print(f"{'Split':<26} {'#T':>4} {'T-✓':>5} {'T-rate':>8}   {'#F':>4} {'F-✓':>5} {'F-rate':>8}")
print("-" * 72)
for s in splits:
    t_rs = buckets[(s, "T")]
    f_rs = buckets[(s, "F")]
    t_ok = sum(1 for r in t_rs if r["lean_agrees_with_gt"] == "True")
    f_ok = sum(1 for r in f_rs if r["lean_agrees_with_gt"] == "True")
    t_n, f_n = len(t_rs), len(f_rs)
    t_rate = 100 * t_ok / t_n if t_n else 0
    f_rate = 100 * f_ok / f_n if f_n else 0
    print(f"{s:<26} {t_n:>4d} {t_ok:>5d} {t_rate:>7.1f}%   "
          f"{f_n:>4d} {f_ok:>5d} {f_rate:>7.1f}%")

# Overall
t_total = [r for r in rows if r["gt_answer"] == "T"]
f_total = [r for r in rows if r["gt_answer"] == "F"]
t_ok = sum(1 for r in t_total if r["lean_agrees_with_gt"] == "True")
f_ok = sum(1 for r in f_total if r["lean_agrees_with_gt"] == "True")
print("-" * 72)
print(f"{'Overall':<26} {len(t_total):>4d} {t_ok:>5d} {100*t_ok/len(t_total):>7.1f}%   "
      f"{len(f_total):>4d} {f_ok:>5d} {100*f_ok/len(f_total):>7.1f}%")

print("\nInterpretation:")
print("  T-rate = Lean soundly verifies positive derivations via forward chaining.")
print("  F-rate is lower because LEMO uses closed-world semantics while Lean is")
print("  classical; a claim 'F' in the benchmark means 'not derivable', which is")
print("  NOT the same as 'provably false' in Lean. A production RLVF-Lean pipeline")
print("  would add an explicit closed-world axiom (CWA) to close this gap.")
