"""
generate_invariance_pairs.py
============================
For every base_positive row in train.csv, generate all 6 variant4
logically-equivalent reformulations.

Output: data/train_lire_pairs.csv
Columns: group_id, law, base_facts, base_rules, equiv_facts, equiv_rules,
         questions, answers

Each row is a (base, equiv) pair with identical answers.
The LIRE trainer uses these to enforce prediction consistency.
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Reuse equivalence generators from data_gen.py
from data_gen import (
    rule, contraposition, double_negation, implication_law,
    identity_law, commutativity_or, demorgan_law,
)

INPUT_FILE  = "data/train.csv"
OUTPUT_FILE = "data/train_lire_pairs.csv"


def extract_color_pair(rules_str: str):
    """
    From rules like "If someone is blue then they are cold. | If someone is orange then cold..."
    extract (c1, c2) = the first two color→cold rules.
    """
    colors_found = []
    for r in rules_str.split(" | "):
        m = re.match(r"If someone is (\w+) then they are cold", r.strip(), re.I)
        if m:
            colors_found.append(m.group(1))
        if len(colors_found) == 2:
            break
    if len(colors_found) < 2:
        return None
    return colors_found[0], colors_found[1]


def build_equiv_rules(rules_list: list, c1: str, c2: str, law: str) -> list:
    """
    Replace the first rule (c1→cold) with a logically equivalent form.
    Returns the new rules list.
    """
    rest = rules_list[1:]   # drop original rule(c1, cold); keep c2→cold onward

    if law == "contrapositive":
        new_first = contraposition(c1, "cold")
        return [new_first] + rest

    elif law == "double_negation":
        new_first = double_negation(c1, "cold")
        return [new_first] + rest

    elif law == "implication":
        new_first = implication_law(c1, "cold")
        return [new_first] + rest

    elif law == "identity":
        new_first = identity_law(c1, "cold")
        return [new_first] + rest

    elif law == "commutativity":
        # "If someone is c2 or c1 then they are cold." replaces both c1→cold and c2→cold
        comm_rule = commutativity_or(c1, c2)
        rest_no_c2 = [r for r in rest if not re.match(
            rf"If someone is {c2} then they are cold", r.strip(), re.I)]
        return [comm_rule] + rest_no_c2

    elif law == "demorgan":
        # Adds "if not c1 and not c2 then not cold" on top of existing rules
        dm_rule = demorgan_law(c1, c2)
        return [dm_rule] + rules_list   # keeps all original rules

    return rules_list


LAWS = ["contrapositive", "double_negation", "implication",
        "identity", "commutativity", "demorgan"]


def generate_pairs():
    print(f"Reading {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    base_pos = [r for r in rows if r["type"] == "base_positive"]
    print(f"  Found {len(base_pos)} base_positive rows → generating {len(base_pos)*6} pairs")

    out_rows = []
    skipped = 0

    for row in base_pos:
        facts_str   = row["facts"]
        rules_str   = row["rules"]
        questions   = row["questions"]
        answers     = row["answers"]
        gid         = row["group_id"]

        cp = extract_color_pair(rules_str)
        if cp is None:
            skipped += 1
            continue
        c1, c2 = cp

        rules_list = [r.strip() for r in rules_str.split(" | ")]

        for law in LAWS:
            equiv_rules_list = build_equiv_rules(rules_list, c1, c2, law)
            equiv_rules_str  = " | ".join(equiv_rules_list)

            out_rows.append({
                "group_id":   gid,
                "law":        law,
                "base_facts": facts_str,
                "base_rules": rules_str,
                "equiv_facts": facts_str,          # facts unchanged
                "equiv_rules": equiv_rules_str,
                "questions":  questions,
                "answers":    answers,             # same answers (logical equiv)
            })

    print(f"  Generated {len(out_rows)} pairs  (skipped {skipped})")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fieldnames = ["group_id","law","base_facts","base_rules",
                  "equiv_facts","equiv_rules","questions","answers"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  Saved → {OUTPUT_FILE}")

    # Quick sanity: verify closures match for a sample
    from scripts.utils.forward_chain import forward_chain
    sample = out_rows[0]
    cl_base  = forward_chain(sample["base_facts"],  sample["base_rules"])
    cl_equiv = forward_chain(sample["equiv_facts"], sample["equiv_rules"])
    print(f"\nSanity (first pair, law={sample['law']}):")
    print(f"  base closure:  {cl_base}")
    print(f"  equiv closure: {cl_equiv}")
    print(f"  Match: {cl_base == cl_equiv}")


if __name__ == "__main__":
    generate_pairs()
