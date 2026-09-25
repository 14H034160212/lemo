"""
The contrapositive at increasing chain depth.

Two results in this paper bracket the effect: the same rule is applied at or
near ceiling with no chain in front of it, and at 0.000--0.107 with two derived
steps in front of it. Depth is then the obvious variable and it has not been
measured, so this split holds the rule, its wording, the query and the oracle
fixed and varies only how many steps must be derived before the rule can fire.

    depth 0   the fact states the rule's antecedent outright (the isolated case)
    depth d   the fact starts a chain of d implications ending in the antecedent

All three arms extend. mp and mt need the chain to deliver `rough`; mp_neg
needs `not sleepy`, which a chain of positive implications cannot produce, so
its final link is itself negated ("If someone is <p> then they are not
sleepy"). Depth is therefore matched across arms.

Every arm carries a foil whose fact does not enter the chain, leaving the
target underivable and the gold answer False, so a policy keying on "the
queried predicate occurs in some rule" scores 0.500 rather than 1.000.
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import forward_chain

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen",
         "Ivan", "Julia", "Kevin", "Linda", "Mike", "Nancy", "Oscar"]
LINK = ["fuzzy", "smooth", "bright", "heavy", "clean", "sharp", "soft", "loud",
        "calm", "wide", "dry", "neat", "mild", "firm", "plain", "keen"]
TARGET, ANTE, NEG_ANTE = "young", "rough", "sleepy"
UNANSWERABLE = "nice"
DISTRACTORS = ["If someone is tall then they are warm.",
               "If someone is quiet then they are careful."]
# The manipulated rule. mp and mt are logically equivalent; mp_neg carries a
# negated antecedent but needs no contraposition.
THIRD = {
    "mp":     f"If someone is {ANTE} then they are {TARGET}.",
    "mt":     f"If someone is not {TARGET} then they are not {ANTE}.",
    "mp_neg": f"If someone is not {NEG_ANTE} then they are {TARGET}.",
}


def lead_in(chain, arm):
    """Rules carrying the fact down to the manipulated rule's antecedent."""
    if not chain:
        return []
    rules = []
    for i, p in enumerate(chain[:-1]):
        rules.append(f"If someone is {p} then they are {chain[i + 1]}.")
    last = chain[-1]
    if arm == "mp_neg":
        rules.append(f"If someone is {last} then they are not {NEG_ANTE}.")
    else:
        rules.append(f"If someone is {last} then they are {ANTE}.")
    return rules


def build(name, depth, rng):
    """{arm: (facts, rules, questions, answers)} with `depth` derived steps."""
    out = {}
    for arm in ("mp", "mp_neg", "mt"):
        # a fresh chain per arm, so no predicate is shared across arms by luck
        chain = rng.sample(LINK, depth) if depth else []
        rules_base = lead_in(chain, arm) + [THIRD[arm]] + DISTRACTORS
        if depth:
            trigger = f"{name} is {chain[0]}"
        else:
            trigger = (f"{name} is not {NEG_ANTE}" if arm == "mp_neg"
                       else f"{name} is {ANTE}")
        for kind, facts in (("", trigger), ("_foil", f"{name} is tall")):
            rules = rules_base[:]
            rng.shuffle(rules)
            rs = " | ".join(rules)
            closure = forward_chain(facts, rs)
            out[arm + kind] = (
                facts, rs,
                f"Q1: {name} is {TARGET}. | Q2: {name} is {UNANSWERABLE}.",
                " | ".join("T" if closure.get(p) is True else "F"
                           for p in (TARGET, UNANSWERABLE)),
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_depth", type=int, default=60)
    ap.add_argument("--depths", default="0,1,2,4,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/test_mt_depth.csv")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    assert max(depths) <= len(LINK), "not enough link predicates for that depth"
    rng = random.Random(args.seed)
    rows, dropped = [], 0
    for d in depths:
        made, guard = 0, 0
        while made < args.per_depth and guard < args.per_depth * 20:
            guard += 1
            name = rng.choice(NAMES)
            g = build(name, d, rng)
            # trigger arms must be T,F and foil arms F,F; anything else means
            # the arms are not matched, and is discarded rather than reported
            if not all(a == ("F | F" if k.endswith("_foil") else "T | F")
                       for k, (_, _, _, a) in g.items()):
                dropped += 1
                continue
            for arm, (fs, rs, qs, ans) in g.items():
                rows.append({"group_id": f"d{d}_{made}", "type": f"d{d}_{arm}",
                             "arm": f"d{d}_{arm}", "depth": d, "facts": fs,
                             "rules": rs, "questions": qs, "answers": ans,
                             "equiv_laws_used": ""})
            made += 1
        if made < args.per_depth:
            print(f"  WARNING: depth {d} produced only {made}/{args.per_depth}")

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["group_id", "type", "arm", "depth", "facts", "rules", "questions",
            "answers", "equiv_laws_used"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    q1 = [r["answers"].split(" | ")[0] for r in rows]
    print(f"Wrote {args.out}: {len(rows)} instances, dropped {dropped}")
    print(f"  by depth : {dict(sorted(Counter(r['depth'] for r in rows).items()))}")
    print(f"  arms     : {len(set(r['arm'] for r in rows))} strata")
    print(f"  Q1 gold  : T={q1.count('T')} F={q1.count('F')} "
          f"(balanced, so predicate presence alone scores 0.500)")


if __name__ == "__main__":
    main()
