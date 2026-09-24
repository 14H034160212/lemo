"""
The isolated control, with a foil that rules out the obvious shortcut.

The first isolated split (make_isolated_control.py) puts one substantive rule
in front of the model and asks for its conclusion, gold True, plus a query on a
predicate no rule mentions, gold False. GPT-4.1 scores 1.000 on both. That is
consistent with doing the inference, and equally consistent with a surface rule
this paper is otherwise at pains to control for:

    the queried predicate appears in some rule  =>  True
    it appears nowhere                          =>  False

which separates the two queries perfectly without any inference at all.

So each arm gets a foil: the same rule, a fact that does not trigger it, and
the same target query, whose gold answer is now False. The queried predicate
still appears in the rule, so the shortcut answers True and is wrong. A model
doing the inference gets both; a model keying on predicate presence gets
exactly one. We report the paired rate -- both halves of an arm correct -- in
the spirit of RULEBREAKERS' matched scoring.
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

RULES = {
    "mp":     "If someone is rough then they are young.",
    "mt":     "If someone is not young then they are not rough.",
    "mp_neg": "If someone is not sleepy then they are young.",
}
TRIGGER = {"mp": "{n} is rough", "mt": "{n} is rough", "mp_neg": "{n} is not sleepy"}
# Satisfies no antecedent of any rule present, so `young` stays underivable.
FOIL_FACT = "{n} is tall"
DISTRACTORS = ["If someone is tall then they are warm.",
               "If someone is quiet then they are careful."]


def build(name, rng):
    out = {}
    for arm, rule in RULES.items():
        for kind, fact_t in (("", TRIGGER[arm]), ("_foil", FOIL_FACT)):
            facts = fact_t.format(n=name)
            rules = [rule] + DISTRACTORS[:]
            rng.shuffle(rules)
            rs = " | ".join(rules)
            closure = forward_chain(facts, rs)
            # Q1 is the target inference; Q2 is the same unanswerable control
            # as the first split, kept so the two files stay comparable.
            ans = ["T" if closure.get("young") is True else "F",
                   "T" if closure.get("nice") is True else "F"]
            qs = [f"Q1: {name} is young.", f"Q2: {name} is nice."]
            out[arm + kind] = (facts, rs, qs, ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/test_isolated_control2.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, dropped = [], 0
    for i in range(args.pairs):
        name = rng.choice(NAMES)
        group = build(name, rng)
        # every trigger arm must be T,F and every foil arm F,F -- otherwise the
        # group is not matched and is discarded rather than silently reported
        ok = all(a == (["F", "F"] if k.endswith("_foil") else ["T", "F"])
                 for k, (_, _, _, a) in group.items())
        if not ok:
            dropped += 1
            continue
        for arm, (fs, rs, qs, ans) in group.items():
            rows.append({
                "group_id": f"iso2_{i}", "type": f"iso_{arm}", "arm": arm,
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
    q1 = [r["answers"].split(" | ")[0] for r in rows]
    print(f"Wrote {args.out}: {len(rows)} instances, dropped {dropped}")
    print(f"  arms   : {dict(Counter(r['arm'] for r in rows))}")
    print(f"  Q1 gold: T={q1.count('T')} F={q1.count('F')}  "
          f"(the target inference is balanced, so predicate-presence scores 0.500)")
    print(f"  all    : T={allans.count('T')} F={allans.count('F')} "
          f"-> majority baseline {max(allans.count('T'),allans.count('F'))/len(allans):.4f}")


if __name__ == "__main__":
    main()
