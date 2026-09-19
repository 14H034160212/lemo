"""
Training-side counterexamples for contradiction detection.

The problem this fixes
----------------------
In data/train_fusion.csv there are exactly two verification outcomes:

    base / Variant 2   "Step 1: Verify facts. Facts are consistent."
    Variant 3          "Step 1: Verify facts. Conflict detected! ... Step 2: Stop."

and every Variant-3 instance carries an extra negated fact. So across the whole
training corpus "an extra negated fact is present" and "the premises are
contradictory" are perfectly correlated. A model can satisfy the objective by
keying on the surface cue and never checking whether the negation actually
conflicts with anything derivable.

That is what the two-class control measured. On test_variant3_mixed.csv,
Fusion-LRA scores 0.861 on contradictory instances and 0.794 on plainly
consistent ones, but only 0.550 -- near chance -- on `consistent_mixed`, where
the extra negated fact is about something the chain cannot derive and is
therefore harmless. Often it is the *same sentence* that makes a base instance
contradictory.

This script generates the missing class: instances that carry a negated fact
and are nonetheless consistent, with a trace that verifies, finds no conflict,
and proceeds. Answers come from the shared oracle, never from a template.

Leakage control: the template space is small (15 names x 30 colour pairs x a
few structures), so random generation would reproduce held-out test instances
verbatim. Every candidate whose (facts, rules) pair occurs in
data/test_variant3_mixed.csv is dropped, and the count is reported.
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import forward_chain, check_answer, detect_contradiction
from scripts.data_generation.make_variant3_mixed import (
    base_chain, questions_for, oracle_answers, ATTRS, COLORS, NAMES,
)

TEST_FILE = "data/test_variant3_mixed.csv"


def load_test_keys(path):
    """(facts, rules) of every held-out instance, so training cannot reuse them."""
    keys = set()
    if not os.path.exists(path):
        print(f"!! {path} missing -- cannot exclude test instances; aborting")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            keys.add((r["facts"], r["rules"]))
    return keys


def consistent_trace(name, facts, rules, question, answer):
    """A verification trace that finds the negation harmless and proceeds.

    Deliberately mirrors the Variant-3 template's first step, so the two
    classes differ in what the check *concludes*, not in whether it happens.
    """
    return (
        f"Reasoning: Step 1: Verify facts. A negative fact is present, so check "
        f"whether it conflicts with anything the rules derive. It does not: "
        f"nothing in the chain entails the opposite, so the facts are "
        f"consistent. Step 2: Apply rules to the consistent premise set. "
        f"Answer: {answer}"
    )


def build(n_groups, seed):
    rng = random.Random(seed)
    random.seed(seed)
    test_keys = load_test_keys(TEST_FILE)
    rows, dropped_leak, dropped_oracle = [], 0, 0

    for _ in range(n_groups):
        name = rng.choice(NAMES)
        for mode in ("disj", "mixed"):
            if mode == "disj":
                facts, rules, (c1, c2) = base_chain(name)
                extra = f"{name} is not {c1}"       # other disjunct still fires
            else:
                facts, rules, _ = base_chain(name, drop=rng.choice(["young", "entry"]))
                base_ans, _ = oracle_answers(facts, rules, name)
                underivable = [a for a, v in zip(ATTRS, base_ans) if v == "F"]
                if not underivable:
                    dropped_oracle += 1
                    continue
                extra = f"{name} is not {rng.choice(underivable)}"

            f = facts + [extra]
            ans, contra = oracle_answers(f, rules, name)
            if contra:                              # must NOT be contradictory
                dropped_oracle += 1
                continue

            fs, rs = " | ".join(f), " | ".join(rules)
            if (fs, rs) in test_keys:               # held out -- never train on it
                dropped_leak += 1
                continue

            for q, a in zip(questions_for(name), ans):
                a_text = "True" if a == "T" else "False"
                rows.append({
                    "input_text": f"Facts: {fs}\nRules: {rs}\nQuestion: {q}\n"
                                  f"Think step by step.",
                    "target_text": consistent_trace(name, fs, rs, q, a_text),
                })
    return rows, dropped_leak, dropped_oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=6000,
                    help="candidate groups to draw; the reachable unique space "
                         "saturates near 1650 instances after test exclusion")
    ap.add_argument("--target_rows", type=int, default=16000,
                    help="oversample the unique instances up to this many rows. "
                         "16000 matches the contradiction class exactly, so "
                         "'extra negated fact' becomes 50/50 contradictory vs "
                         "consistent instead of 100%% contradictory.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base", default="data/train_fusion.csv")
    ap.add_argument("--out", default="data/train_fusion_v3aug.csv")
    args = ap.parse_args()

    aug, leak, bad = build(args.groups, args.seed)

    # Deduplicate, then oversample to the target. The base corpus is itself
    # built by repeating its source rows five times, so repetition here is the
    # same treatment, not a new liberty -- but it must be counted, because the
    # reachable unique space is small and the effective sample size is the
    # number of distinct instances, not the row count.
    seen, uniq = set(), []
    for r in aug:
        k = (r["input_text"], r["target_text"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    n_uniq = len(uniq)
    if args.target_rows and n_uniq:
        reps = -(-args.target_rows // n_uniq)
        aug = (uniq * reps)[:args.target_rows]
    else:
        aug = uniq
    with open(args.base, encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))

    combined = base_rows + aug
    random.Random(args.seed).shuffle(combined)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
        w.writeheader()
        w.writerows(combined)

    n_t = sum(1 for r in aug if r["target_text"].rstrip().endswith("True"))
    print(f"Wrote {args.out}")
    print(f"  original          : {len(base_rows)}")
    print(f"  counterexamples   : {len(aug)}  ({len(aug)/len(combined):.1%} of the corpus)")
    print(f"    unique instances: {n_uniq}  (each repeated ~{len(aug)/max(n_uniq,1):.1f}x)")
    print(f"    their answers   : True={n_t}  False={len(aug)-n_t}")
    print(f"  dropped (in test) : {leak}")
    print(f"  dropped (oracle)  : {bad}")
    print(f"  total             : {len(combined)}")


if __name__ == "__main__":
    main()
