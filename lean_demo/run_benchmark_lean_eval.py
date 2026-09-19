"""
run_benchmark_lean_eval.py
==========================
Dispatch real LEMO benchmark rows to the Lean 4 kernel and measure
how often Lean formally verifies the ground-truth answer.

This is the multi-step extension of `run_pilot_study.py`: instead of
8 hand-written Socrates variants, we auto-translate a stratified sample
of rows from `data_v2/test_*.csv` using `lemo_to_lean.emit_theorem`,
and record Lean's per-question verdict.

Output: `benchmark_lean_results.csv` with one row per question, plus
a printed summary table.
"""

from __future__ import annotations

import csv
import random
import sys
import time
import os
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
# Keep the original 187-question run on data_v2 intact; name the output
# after the dataset it was measured on, so the two cannot be confused again.
OUT_NAME = "benchmark_lean_results_%s.csv" % os.environ.get("LEMO_DATA_DIR", "data")
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from lemo_to_lean import parse_row, emit_theorem, verify_with_lean  # noqa: E402


# The Lean study originally read data_v2/ -- the five-domain expansion -- while
# every other table in the paper is measured on data/. Table 5 was therefore
# reporting a different dataset than Tables 1-4 without saying so. Default to
# data/ so the formal-verification numbers sit on the same benchmark as the
# rest; pass --data_dir data_v2 to reproduce the original figures.
DATA_DIR = _HERE.parent / os.environ.get("LEMO_DATA_DIR", "data")

# (split_name, csv_filename, n_rows_to_sample)
# Scaled up from 20 rows/split. At ~0.37 s per trace the original 187-question
# sample cost under two minutes, so the small n was not a compute limit -- and
# 187 questions is thin for a per-split T/F breakdown (variant2's F cell had
# n=20).
#
# variant3_mixed is added because it is the one split whose labels we need an
# *independent* check on: every other label in this benchmark comes from our own
# forward_chain oracle, which is also the oracle RLVF is rewarded against.
# Reviewer 2hi4 objected that training reward and evaluation semantics are
# therefore coupled. A Lean 4 kernel derivation is not our oracle, so agreement
# here is an external audit of the control's labels rather than a restatement
# of them.
SPLITS = [
    ("base",                   "test_base.csv",           150),
    ("variant1_redundant",     "test_variant1.csv",       150),
    ("variant2_essential",     "test_variant2.csv",       150),
    ("variant3_contradiction", "test_variant3.csv",       150),
    # variant3_mixed is held out for now: the runner burned six hours of CPU on
    # this split with no output and produced nothing. Translation of these rows
    # is fast when tested in isolation, so the hang is in the verification loop
    # and is not yet diagnosed. The four splits above complete in ~26 minutes.
]


def load_sample(csv_path: Path, n: int, seed: int = 0) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    all_results = []
    t_total_start = time.time()

    for split_name, fname, n in SPLITS:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[skip] {path} missing")
            continue

        print(f"\n── Split: {split_name} ({fname})  |  sampling {n} rows")
        sample = load_sample(path, n)

        n_rows = 0
        n_translated = 0
        n_questions = 0
        n_lean_accept = 0
        n_lean_reject = 0
        total_time = 0.0

        for row in sample:
            parsed = parse_row(row)
            n_rows += 1
            if parsed is None:
                continue
            n_translated += 1

            for q_idx in range(len(parsed.questions)):
                script, _, gt_T = emit_theorem(parsed, q_idx)
                t0 = time.time()
                r, err = verify_with_lean(script, timeout=30.0)
                dt = time.time() - t0
                total_time += dt
                n_questions += 1
                if r == 1:
                    n_lean_accept += 1
                else:
                    n_lean_reject += 1

                all_results.append({
                    "split": split_name,
                    "group_id": row.get("group_id", ""),
                    "type": row.get("type", ""),
                    "q_idx": q_idx,
                    "gt_answer": "T" if gt_T else "F",
                    "lean_verdict": "accept" if r == 1 else "reject",
                    "lean_agrees_with_gt": r == 1,
                    "time_seconds": round(dt, 2),
                })

        rate = n_lean_accept / n_questions if n_questions else 0.0
        print(f"   rows parsed successfully : {n_translated}/{n_rows}")
        print(f"   questions tested         : {n_questions}")
        print(f"   Lean accepted (= agrees) : {n_lean_accept} ({rate*100:.1f}%)")
        print(f"   Lean rejected            : {n_lean_reject}")
        print(f"   avg verification time    : "
              f"{total_time / max(1, n_questions):.2f} s/question")

    out = _HERE / OUT_NAME
    if all_results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)

    total_dt = time.time() - t_total_start

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Split':<26} {'Q':>5} {'Accept':>7} {'Rate':>7}")
    print("-" * 72)
    by_split: Dict[str, List[dict]] = {}
    for r in all_results:
        by_split.setdefault(r["split"], []).append(r)
    for split, rs in by_split.items():
        n = len(rs)
        k = sum(1 for r in rs if r["lean_agrees_with_gt"])
        print(f"{split:<26} {n:>5d} {k:>7d} {100*k/n:>6.1f}%")
    n_total = len(all_results)
    n_accept = sum(1 for r in all_results if r["lean_agrees_with_gt"])
    print("-" * 72)
    print(f"{'Overall':<26} {n_total:>5d} {n_accept:>7d} {100*n_accept/max(1,n_total):>6.1f}%")
    print(f"Total wall time: {total_dt:.1f} s   ({total_dt / max(1, n_total):.2f} s/question)")
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
