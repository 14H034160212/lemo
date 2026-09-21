"""
Depth-scaled benchmark: does checking the premises first actually save work?

The practical argument for a verify-before-deduce prior is that it should let a
model stop early. That argument has never been measured, and the obvious way to
set it up does not work: burying the contradiction deep in the chain makes
*detecting* it cost exactly as much as answering, because under the conservative
semantics you have to derive the closure either way.

The structure where checking first genuinely pays is the opposite one -- a
contradiction that is visible in the premises, and queries that sit far down the
chain:

    facts:  X is a1.  X is warm.  X is not warm.        <- 1 step to spot
    rules:  a1 -> a2 -> a3 -> ... -> a(n)               <- n hops to answer
    query:  is X a(n)?

A model that validates the premise set answers all four queries immediately and
its cost is flat in n. A model that chains forward pays for n hops per query, so
its cost grows with n. Plotting tokens against n separates the two behaviours,
and the gap is the quantity the deployment argument is actually about.

Each depth also gets a consistent control with the same chain length, so a model
cannot score by answering False whenever the chain is long, and so the cost of
the honest derivation is measured too.

Labels come from the shared oracle, never from the template.
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import forward_chain, check_answer, detect_contradiction

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen",
         "Ivan", "Julia", "Kevin", "Linda", "Mike", "Nancy", "Oscar"]
# Chain attributes, distinct from the contradiction pair so the two are
# independent: the contradiction is never on the path to the query.
CHAIN_ATTRS = ["quick", "bright", "calm", "eager", "firm", "gentle", "humble",
               "keen", "lively", "mild", "noble", "patient", "quiet", "ready",
               "steady", "tidy", "upright", "vivid", "warmhearted", "zealous"]
CONTRA_ATTRS = ["warm", "tall", "loud", "sharp", "heavy"]


def rule(p, q):
    return f"If someone is {p} then they are {q}."


def build(name, depth, contradictory, rng):
    """A depth-hop chain, optionally with a premise-level contradiction.

    The contradicting pair is on an attribute that appears nowhere in the chain,
    so the contradiction adds no derivation work -- only the obligation to look.
    """
    attrs = rng.sample(CHAIN_ATTRS, depth + 1)
    facts = [f"{name} is {attrs[0]}"]
    rules = [rule(attrs[i], attrs[i + 1]) for i in range(depth)]
    rng.shuffle(rules)                     # the chain is not given in order
    if contradictory:
        # detect_contradiction() only finds conflicts produced by applying a
        # rule -- parse_facts() lets a later fact silently overwrite an earlier
        # one, so a contradictory pair stated directly in the facts is invisible
        # to it. The conflict is therefore raised by one rule application: still
        # a single hop, and still independent of the query chain.
        c, d = rng.sample(CONTRA_ATTRS, 2)
        facts += [f"{name} is {c}", f"{name} is not {d}"]
        rules += [rule(c, d)]
    # queries: the deepest attribute plus three others along the chain
    idx = [depth] + sorted(rng.sample(range(1, depth), min(3, depth - 1)))
    qs = [f"Q{k+1}: {name} is {attrs[i]}." for k, i in enumerate(idx)]

    fs, rs = " | ".join(facts), " | ".join(rules)
    if detect_contradiction(fs, rs):
        ans = ["F"] * len(qs)
        is_contra = True
    else:
        closure = forward_chain(fs, rs)
        ans = ["T" if check_answer(q, closure) is True else "F" for q in qs]
        is_contra = False
    return fs, rs, qs, ans, is_contra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--per_depth", type=int, default=60,
                    help="instances per (depth, class); API cost scales with this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/test_depth.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, rejected = [], 0
    for depth in args.depths:
        for contradictory in (True, False):
            made = 0
            while made < args.per_depth:
                name = rng.choice(NAMES)
                fs, rs, qs, ans, is_contra = build(name, depth, contradictory, rng)
                if is_contra != contradictory:      # oracle must agree with intent
                    rejected += 1
                    continue
                rows.append({
                    "group_id": f"d{depth}_{'c' if contradictory else 'k'}_{made}",
                    "type": f"depth{depth}_{'contra' if contradictory else 'consistent'}",
                    "depth": depth,
                    "facts": fs, "rules": rs,
                    "questions": " | ".join(qs), "answers": " | ".join(ans),
                    "equiv_laws_used": "",
                })
                made += 1
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["group_id", "type", "depth", "facts", "rules", "questions",
            "answers", "equiv_laws_used"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    allans = [a for r in rows for a in r["answers"].split(" | ")]
    t = allans.count("T")
    print(f"Wrote {args.out}: {len(rows)} instances / {len(allans)} questions")
    print(f"  depths        : {args.depths}")
    print(f"  labels        : T={t} ({t/len(allans):.3f})  F={len(allans)-t}")
    print(f"  majority-class baseline = {max(t, len(allans)-t)/len(allans):.4f}")
    print(f"  rejected by oracle      = {rejected}")
    import collections
    c = collections.Counter(r["type"] for r in rows)
    for k in sorted(c, key=lambda s: (int(s.split("_")[0][5:]), s)):
        print(f"    {k:26s} {c[k]}")


if __name__ == "__main__":
    main()
