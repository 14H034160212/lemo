"""
Build data/test_variant3_mixed.csv -- a two-class control for Variant 3.

Why this file has to exist
--------------------------
test_variant3.csv is single-class: under the conservative contradiction
semantics every question on a contradictory instance is False, so the split is
100% F and its majority-class baseline is 1.0000. A model that simply answers
False whenever it sees an extra negated fact scores a perfect 1.0000 there
without ever detecting a contradiction. Measured on the September 2026
checkpoints:

    Fusion-LRA               Variant 3 = 0.9820
    generative RLVF (fixed)  Variant 3 = 1.0000, predicting F on 4000/4000

Neither number is evidence of contradiction detection, and the trace text does
not rescue them: "conflict"/"contradiction" appears in 95.0% of Fusion-LRA's
*base*-split traces, where there is no contradiction at all. The word is part
of the verification preamble, not a detection signal -- which is exactly the
format-compliance objection Reviewer 2hi4 raised.

This split interleaves three instance types that share a surface form -- a base
chain plus one extra negated fact -- so that answering False on sight scores at
the baseline instead of perfect:

  contra            the extra fact contradicts the chain          -> all F
  consistent_disj   the extra fact negates one disjunct; the
                    other still fires, so nothing breaks          -> oracle
  consistent_mixed  a chain with a rule removed, plus a negated
                    fact about something the chain no longer
                    derives                                       -> oracle

consistent_mixed is the sharp one: the extra fact can be the *same sentence*
("Anne is not young") that makes a base instance contradictory, and here it is
consistent, because the rule that would have derived `young` is gone. Surface
matching cannot separate the two; only following the rules can.

Every row is checked with the shared oracle: detect_contradiction() must agree
with the intended class, and the answers for consistent rows are computed by
forward_chain(), never assumed.
"""

import csv
import os
import random
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import (
    forward_chain, check_answer, detect_contradiction,
)

NAMES = ["Anne", "Bob", "Claire", "David", "Emma",
         "Frank", "Grace", "Helen", "Ivan", "Julia",
         "Kevin", "Linda", "Mike", "Nancy", "Oscar"]
COLORS = ["green", "blue", "red", "yellow", "purple", "orange"]
ATTRS = ["cold", "rough", "young", "nice"]


def rule(p, q):
    return f"If someone is {p} then they are {q}."


def questions_for(name):
    return [f"Q{i+1}: {name} is {a}." for i, a in enumerate(ATTRS)]


def base_chain(name, drop=None):
    """The canonical chain, optionally with one link removed.

    drop=None     everything derivable: cold, rough, young, nice
    drop="young"  no 'not young -> not rough', so young and nice are unreachable
    drop="entry"  no colour -> cold rules, so nothing downstream fires at all

    Both drops remove derivations without making anything inconsistent, which
    is what lets the same negated sentence be contradictory in one instance and
    consistent in another. Two different drops are needed so that every one of
    the four attributes can appear negated in *both* classes -- with only the
    "young" drop, `cold`/`rough` would occur exclusively in contradictory rows
    and a model could separate the classes on the predicate alone.
    """
    c1 = random.choice(COLORS)
    c2 = random.choice([c for c in COLORS if c != c1])
    facts = [f"{name} is {c1} or {c2}"]
    rules = []
    if drop != "entry":
        rules += [rule(c1, "cold"), rule(c2, "cold")]
    rules.append(rule("cold", "rough"))
    if drop != "young":
        rules.append(rule("not young", "not rough"))
    rules += [rule("young", "cold"), rule("young", "nice"),
              "If someone is tall then they are warm."]
    return facts, rules, (c1, c2)


def oracle_answers(facts, rules, name):
    """Labels from the shared oracle, under the conservative semantics."""
    f, r = " | ".join(facts), " | ".join(rules)
    if detect_contradiction(f, r):
        return ["F"] * len(ATTRS), True
    closure = forward_chain(f, r)
    out = []
    for q in questions_for(name):
        val = check_answer(q, closure)
        out.append("T" if val is True else "F")
    return out, False


def build(num_groups, seed=0):
    random.seed(seed)
    rows = []
    stats = {"contra": 0, "consistent_disj": 0, "consistent_mixed": 0, "rejected": 0}

    for _ in range(num_groups):
        name = random.choice(NAMES)
        gid = str(uuid.uuid4())

        # ---- 1. contradiction: negate something the chain derives ----
        facts, rules, (c1, c2) = base_chain(name)
        extra = f"{name} is not {random.choice(ATTRS)}"
        f = facts + [extra]
        ans, contra = oracle_answers(f, rules, name)
        if not contra:
            stats["rejected"] += 1            # oracle disagreed -> drop it
        else:
            rows.append((gid, "contra", f, rules, name, ans))
            stats["contra"] += 1

        # ---- 2. consistent: negate one disjunct, the other still fires ----
        facts, rules, (c1, c2) = base_chain(name)
        f = facts + [f"{name} is not {c1}"]
        ans, contra = oracle_answers(f, rules, name)
        if contra:
            stats["rejected"] += 1
        else:
            rows.append((gid, "consistent_disj", f, rules, name, ans))
            stats["consistent_disj"] += 1

        # ---- 3. consistent: negate something this chain cannot derive ----
        # Same sentence shape as (1) -- often the very same sentence -- but the
        # rule that would derive it has been removed, so nothing is broken.
        facts, rules, (c1, c2) = base_chain(
            name, drop=random.choice(["young", "entry"]))
        base_ans, _ = oracle_answers(facts, rules, name)
        underivable = [a for a, v in zip(ATTRS, base_ans) if v == "F"]
        if not underivable:
            stats["rejected"] += 1
        else:
            f = facts + [f"{name} is not {random.choice(underivable)}"]
            ans, contra = oracle_answers(f, rules, name)
            if contra:
                stats["rejected"] += 1
            else:
                rows.append((gid, "consistent_mixed", f, rules, name, ans))
                stats["consistent_mixed"] += 1

    return rows, stats


def main():
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows, stats = build(num)
    random.shuffle(rows)

    out = "data/test_variant3_mixed.csv"
    header = ["group_id", "type", "facts", "rules", "questions", "answers",
              "equiv_laws_used"]
    os.makedirs("data", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for gid, typ, facts, rules, name, ans in rows:
            w.writerow({
                "group_id": gid, "type": typ,
                "facts": " | ".join(facts), "rules": " | ".join(rules),
                "questions": " | ".join(questions_for(name)),
                "answers": " | ".join(ans), "equiv_laws_used": "",
            })

    allans = [a for _, _, _, _, _, ans in rows for a in ans]
    t = allans.count("T")
    print(f"Wrote {out}: {len(rows)} rows / {len(allans)} questions")
    print(f"  composition : {stats}")
    print(f"  labels      : T={t} ({t/len(allans):.3f})  F={len(allans)-t}")
    print(f"  majority-class baseline = {max(t, len(allans)-t)/len(allans):.4f}")


if __name__ == "__main__":
    main()
