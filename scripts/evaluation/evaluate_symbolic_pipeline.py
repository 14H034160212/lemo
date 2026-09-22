"""
The neuro-symbolic baseline, run end to end and reported where it wins and
where it stops.

Reviewer request: a systematic comparison against symbolic and verifier-based
methods, which the draft answered with a single external guard. The honest
comparison is unfavourable and worth making explicitly, because it is what
bounds any claim a trained model can make here.

On this benchmark the labels are computed by forward chaining, so a solver
handed a correct formalisation scores 1.000 by construction and there is nothing
to compare. The measurable question is the pipeline: parse the natural-language
premises into logic, then solve. That has two failure modes and they fall in
different places.

  coverage  the fraction of instances the parser can render at all
  accuracy  among those, how often the solver's answer matches the label

Run against the synthetic splits, where the text is generated from the same
templates the parser expects, and against LogicNLI / MNLI / FOLIO, where it is
not. The gap between the two is the regime in which a trained model has any role
at all: the solver is unbeatable inside its grammar and useless outside it.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.utils.forward_chain import (
    forward_chain, check_answer, detect_contradiction, parse_facts, parse_rules,
)

_Q = re.compile(r"^Q?\d*:?\s*(.+?)\s+is\s+(not\s+)?(\w+)\.?\s*$", re.I)


def solver_answer(facts, rules, question):
    """Conservative semantics, matching how the benchmark is labelled."""
    if detect_contradiction(facts, rules):
        return "F"
    closure = forward_chain(facts, rules)
    val = check_answer(question, closure)
    return "T" if val is True else "F"


def covered(facts, rules, question):
    """Whether the pipeline can render this instance at all.

    A rule string that parses to nothing, or a query naming an attribute the
    parser never saw, means the formalisation step failed -- which is the
    failure the synthetic splits never exercise.
    """
    if not parse_rules(rules):
        return False
    if not parse_facts(facts):
        return False
    return bool(_Q.match(question.strip()))


def run_split(path, facts_f="facts", rules_f="rules", q_f="questions",
              a_f="answers", max_rows=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if max_rows:
        rows = rows[:max_rows]
    n = cov = correct = 0
    for r in rows:
        if facts_f not in r:
            return None
        qs = str(r[q_f]).split(" | ")
        gold = [a.strip() for a in str(r[a_f]).split(" | ")]
        for q, g in zip(qs, gold):
            n += 1
            if not covered(r[facts_f], r[rules_f], q):
                continue
            cov += 1
            correct += (solver_answer(r[facts_f], r[rules_f], q) == g)
    return {"questions": n, "coverage": round(cov / n, 4) if n else 0.0,
            "accuracy_on_covered": round(correct / cov, 4) if cov else None,
            "accuracy_overall": round(correct / n, 4) if n else None}


def run_nl(path):
    """LogicNLI / MNLI, whose rows store one prompt string rather than fields."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = cov = correct = 0
    for r in rows:
        t = str(r["input_text"])
        gold = "T" if str(r["target_text"]).strip() == "True" else "F"
        m = re.search(r"Facts:\s*(.*?)\nQuestion:\s*(.*?)(?:\n|$)", t, re.S)
        n += 1
        if not m:
            continue
        body, q = m.group(1), m.group(2)
        # the prompt puts facts and rules in one block; split on sentences that
        # look like rules
        sents = [x.strip() for x in body.replace("\n", " ").split(".") if x.strip()]
        rules = " | ".join(x + "." for x in sents if x.lower().startswith("if"))
        facts = " | ".join(x for x in sents if not x.lower().startswith("if"))
        if not covered(facts, rules, q):
            continue
        cov += 1
        correct += (solver_answer(facts, rules, q) == gold)
    return {"questions": n, "coverage": round(cov / n, 4) if n else 0.0,
            "accuracy_on_covered": round(correct / cov, 4) if cov else None,
            "accuracy_overall": round(correct / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/symbolic_pipeline.json")
    ap.add_argument("--max_rows", type=int, default=1000)
    args = ap.parse_args()

    out = {}
    print("synthetic splits (text generated from the parser's own templates)")
    print(f'{"split":38s}{"questions":>11s}{"coverage":>10s}{"acc|covered":>13s}')
    for f in sorted(glob.glob("data/test_*.csv")):
        r = run_split(f, max_rows=args.max_rows)
        if not r:
            continue
        out[os.path.basename(f)] = r
        acc = r["accuracy_on_covered"]
        print(f'{os.path.basename(f):38s}{r["questions"]:11d}{r["coverage"]:10.4f}'
              f'{(acc if acc is not None else float("nan")):13.4f}')

    print("\nnatural language (text the parser was not built for)")
    for f in ["data/real_world/logicnli_eval_fixed.csv", "data/real_world/mnli_eval.csv"]:
        if not os.path.exists(f):
            continue
        r = run_nl(f)
        out[os.path.basename(f)] = r
        acc = r["accuracy_on_covered"]
        print(f'{os.path.basename(f):38s}{r["questions"]:11d}{r["coverage"]:10.4f}'
              f'{(acc if acc is not None else float("nan")):13.4f}')

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
