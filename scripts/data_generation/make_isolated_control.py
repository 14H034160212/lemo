"""
The same three inference forms, stripped of the chain.

Sec. `sec:chain` shows that inside a four-step chain the contrapositive arm
collapses while the matched negated-premise arm is perfect. The obvious
objection is that this only restates a known result -- modus tollens is harder
than modus ponens -- measured on our own data. That objection is testable: if
the deficit belongs to the RULE, the same rule should fail when it is the only
inference in the instance; if it belongs to the CHAIN, it should not.

So each arm here asks for exactly the inference the chain arm asks for at Q3,
with the two upstream steps replaced by a fact and the downstream step removed:

    mp      Anne is rough.      If someone is rough then they are young.
    mt      Anne is rough.      If someone is not young then they are not rough.
    mp_neg  Anne is not sleepy. If someone is not sleepy then they are young.

all three concluding `Anne is young`. Same predicates, same wording, same
oracle, one step instead of three.

An all-True split is saturated by a constant responder, so every instance also
carries a second query on a predicate nothing derives. Labels are 50/50 and the
majority-class baseline is 0.500; the target inference is read at Q1.
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import forward_chain

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen",
         "Ivan", "Julia", "Kevin", "Linda", "Mike", "Nancy", "Oscar"]

# The arms of the chain control, verbatim, so the only difference between the
# two experiments is the presence of the chain.
ARMS = {
    "mp":     ("{n} is rough",      "If someone is rough then they are young."),
    "mt":     ("{n} is rough",      "If someone is not young then they are not rough."),
    "mp_neg": ("{n} is not sleepy", "If someone is not sleepy then they are young."),
}
# Distractors: present in the chain instances too, and derive nothing about
# `young` or about the unanswerable query.
DISTRACTORS = [
    "If someone is tall then they are warm.",
    "If someone is quiet then they are careful.",
]
UNANSWERABLE = "nice"   # no rule concludes it, so closed-world makes it False


def build(name, rng):
    out = {}
    for arm, (fact_t, rule) in ARMS.items():
        facts = fact_t.format(n=name)
        rules = [rule] + DISTRACTORS[:]
        rng.shuffle(rules)
        rs = " | ".join(rules)
        closure = forward_chain(facts, rs)
        ans = ["T" if closure.get("young") is True else "F",
               "T" if closure.get(UNANSWERABLE) is True else "F"]
        qs = [f"Q1: {name} is young.", f"Q2: {name} is {UNANSWERABLE}."]
        out[arm] = (facts, rs, qs, ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/test_isolated_control.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, dropped = [], 0
    for i in range(args.pairs):
        name = rng.choice(NAMES)
        triple = build(name, rng)
        # every arm must be oracle-verified T,F -- otherwise the arms are not
        # matched and the instance is discarded rather than silently reported
        if any(a != ["T", "F"] for _, _, _, a in triple.values()):
            dropped += 1
            continue
        for arm, (fs, rs, qs, ans) in triple.items():
            rows.append({
                "group_id": f"iso_{i}", "type": f"iso_{arm}", "arm": arm,
                "facts": fs, "rules": rs,
                "questions": " | ".join(qs), "answers": " | ".join(ans),
                "equiv_laws_used": "",
            })
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["group_id", "type", "arm", "facts", "rules", "questions",
            "answers", "equiv_laws_used"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    from collections import Counter
    allans = [a for r in rows for a in r["answers"].split(" | ")]
    print(f"Wrote {args.out}: {len(rows)} instances")
    print(f"  arms     : {dict(Counter(r['arm'] for r in rows))}")
    print(f"  dropped  : {dropped} (oracle disagreed across arms)")
    print(f"  labels   : T={allans.count('T')} F={allans.count('F')} "
          f"-> majority-class baseline {max(allans.count('T'), allans.count('F'))/len(allans):.4f}")


if __name__ == "__main__":
    main()
