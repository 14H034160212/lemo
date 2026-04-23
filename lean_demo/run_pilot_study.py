"""
run_pilot_study.py
==================
Pilot study for RLVF-Lean: dispatch a small battery of candidate
reasoning traces to Lean 4 and record the kernel's step-level reward.

This is the experiment reported in Section 6.5 / Appendix of the paper.
Results are written to `pilot_results.csv`.
"""

from __future__ import annotations

import csv
import time
from typing import List, Tuple

from lean_verifier_bridge import lean_reward


# --------------------------------------------------------------------- #
# Test battery                                                          #
# --------------------------------------------------------------------- #
# Each row: (trace_id, category, expected_reward, tactic_list)
TRACES: List[Tuple[str, str, int, List[str]]] = [
    # ----- Correct halt-on-contradiction traces -----
    ("T1-halt-minimal", "correct_halt", +1, [
        "have h_mortal : Mortal Socrates := men_are_mortal Socrates h_man",
        "exact absurd h_mortal h_not_mortal",
    ]),
    ("T2-halt-direct", "correct_halt", +1, [
        "exact absurd (men_are_mortal Socrates h_man) h_not_mortal",
    ]),
    ("T3-halt-contradiction-tactic", "correct_halt", +1, [
        "apply h_not_mortal",
        "exact men_are_mortal Socrates h_man",
    ]),

    # ----- Naive "continue deducing" traces (expected reward = -1) -----
    ("T4-naive-stops-early", "naive_continue", -1, [
        "have h_mortal : Mortal Socrates := men_are_mortal Socrates h_man",
        "-- model stops here, never derives False",
    ]),
    ("T5-naive-wrong-goal", "naive_continue", -1, [
        "exact h_man",
    ]),
    ("T6-naive-empty", "naive_continue", -1, [
        "sorry",
    ]),

    # ----- Malformed tactics (syntax errors, expected -1) -----
    ("T7-syntax-error", "malformed", -1, [
        "this_tactic_does_not_exist blah blah",
    ]),
    ("T8-misspelled-lemma", "malformed", -1, [
        "exact absurd (mens_are_mortals Socrates h_man) h_not_mortal",
    ]),
]


# --------------------------------------------------------------------- #
# Runner                                                                #
# --------------------------------------------------------------------- #

def main():
    results = []
    print(f"{'ID':<28} {'category':<16} {'expected':>9} {'actual':>7} {'time_s':>8} {'ok':>4}")
    print("-" * 80)

    t_start = time.time()
    for tid, cat, expected, trace in TRACES:
        t0 = time.time()
        reward, stderr = lean_reward(trace, timeout=30.0)
        dt = time.time() - t0
        ok = "✓" if reward == expected else "✗"
        print(f"{tid:<28} {cat:<16} {expected:>+9d} {reward:>+7d} {dt:>8.2f} {ok:>4}")
        results.append({
            "trace_id": tid,
            "category": cat,
            "expected_reward": expected,
            "actual_reward": reward,
            "match": reward == expected,
            "time_seconds": round(dt, 2),
            "n_tactics": len(trace),
        })
    t_total = time.time() - t_start

    # Save CSV
    out_path = "pilot_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Summary
    n = len(results)
    n_match = sum(r["match"] for r in results)
    n_good = sum(1 for r in results if r["category"] == "correct_halt")
    n_good_ok = sum(1 for r in results
                    if r["category"] == "correct_halt" and r["match"])
    n_bad = n - n_good
    n_bad_ok = n_match - n_good_ok

    print("-" * 80)
    print(f"Total traces           : {n}")
    print(f"Agreement with oracle  : {n_match}/{n} = {100*n_match/n:.1f}%")
    print(f"  correct halt traces  : {n_good_ok}/{n_good}")
    print(f"  naive/malformed      : {n_bad_ok}/{n_bad}")
    print(f"Mean verification time : {sum(r['time_seconds'] for r in results)/n:.2f} s / trace")
    print(f"Total wall time        : {t_total:.1f} s")
    print(f"Results written to     : {out_path}")


if __name__ == "__main__":
    main()
