"""
Matched modus-ponens / modus-tollens pairs, to measure what one weak rule costs
inside a chain.

The frontier models in our evaluation answer the first two steps of a
four-step chain correctly and the last two incorrectly. The natural reading is
that the third step fails because it requires contraposition, and the fourth
fails only as a consequence -- but that is inference, not measurement. This
benchmark makes it a measurement.

Each instance is generated twice, identical in every respect except the form of
the third rule:

    mp  "If someone is rough then they are young."          modus ponens
    mt  "If someone is not young then they are not rough."  modus tollens

The two are logically equivalent, and the shared forward-chaining oracle
confirms the closures are identical, so both carry the same gold answers. Any
difference in model accuracy on the pair is attributable to the inference form
alone.

Queries are laid out so the chain position of each is known:

    Q1 cold   upstream of the manipulated rule   (disjunction elimination + MP)
    Q2 rough  upstream                            (MP)
    Q3 young  THE MANIPULATED STEP                (MP in one arm, MT in the other)
    Q4 nice   downstream of it                    (MP from young)

Q3 measures the primary failure. Q4 measures propagation: an error there is
usually not an independent reasoning failure but a consequence of Q3, and an
aggregate score cannot tell the two apart. That separation is the point.
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
COLORS = ["green", "blue", "red", "yellow", "purple", "orange"]
CHAIN = ["cold", "rough", "young", "nice"]   # Q1..Q4, in chain order


def build_pair(name, rng):
    c1, c2 = rng.sample(COLORS, 2)
    facts = [f"{name} is {c1} or {c2}", f"{name} is not {c1}"]
    common = [
        f"If someone is {c1} then they are cold.",
        f"If someone is {c2} then they are cold.",
        "If someone is cold then they are rough.",
        "If someone is young then they are nice.",
        "If someone is tall then they are warm.",       # distractor
    ]
    # Three ways to make `young` derivable. mp and mt differ only in inference
    # form. mp_neg carries a negation but needs no contraposition, which is
    # what separates "cannot contrapose" from "cannot handle negation" -- the
    # mt rule confounds the two, since it is both a contrapositive and a
    # double negative.
    arms = {
        "mp":     "If someone is rough then they are young.",
        "mt":     "If someone is not young then they are not rough.",
        "mp_neg": "If someone is not sleepy then they are young.",
    }
    qs = [f"Q{i+1}: {name} is {a}." for i, a in enumerate(CHAIN)]
    out = {}
    for arm, third in arms.items():
        arm_facts = facts + ([f"{name} is not sleepy"] if arm == "mp_neg" else [])
        rules = common[:3] + [third] + common[3:]
        rng.shuffle(rules)
        fs, rs = " | ".join(arm_facts), " | ".join(rules)
        closure = forward_chain(fs, rs)
        ans = ["T" if closure.get(a) is True else "F" for a in CHAIN]
        out[arm] = (fs, rs, qs, ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/test_mt_control.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, dropped = [], 0
    for i in range(args.pairs):
        name = rng.choice(NAMES)
        pair = build_pair(name, rng)
        # both arms must be all-True; if the oracle disagrees the pair is not
        # matched and is discarded rather than silently reported
        if any(a != ["T"] * 4 for _, _, _, a in pair.values()):
            dropped += 1
            continue
        for arm, (fs, rs, qs, ans) in pair.items():
            rows.append({
                "group_id": f"mtc_{i}", "type": f"chain_{arm}", "arm": arm,
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
    print(f"Wrote {args.out}: {len(rows)} instances ({len(rows)//2} matched pairs)")
    print(f"  arms          : {dict(Counter(r['arm'] for r in rows))}")
    print(f"  dropped       : {dropped} (oracle disagreed between arms)")
    allans = [a for r in rows for a in r["answers"].split(" | ")]
    print(f"  labels        : all-True by construction "
          f"(T={allans.count('T')}, F={allans.count('F')})")
    print("  NOTE: an all-True split is saturated by a constant-True responder;")
    print("  it is interpretable only per question position, which is its purpose.")


if __name__ == "__main__":
    main()
