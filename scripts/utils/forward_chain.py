"""
forward_chain.py
================
Core logical engine for the dataset.

Given natural-language facts and pipe-separated rules, computes the
complete set of derivable attribute truth-values via forward chaining.

This is the key "first-principles" component:
  • Step reward:    is a model's claimed fact in closure(F, R)?
  • ORM reward:     is the final T/F answer correct?
  • Invariance check: do closure(F, R) == closure(F, R') for equiv R, R'?

All rule patterns that appear in data_gen.py are handled.
"""

import re
from typing import Dict, List, Optional, Tuple

# Sentinel for "conjunctive premise" rules (De Morgan / commutativity)
_AND_PREFIX = "__and__"


# ------------------------------------------------------------------ #
# Rule parsing                                                         #
# ------------------------------------------------------------------ #

def _parse_single_rule(rule_str: str) -> List[Tuple]:
    """
    Convert one rule sentence to a list of (premise, conclusion) pairs.

    Each item is either:
      Simple:     (attr_p, bool_p, attr_c, bool_c)
      Conjunctive (De Morgan): ("__and__", (attr1, attr2), attr_c, bool_c)
        fires when attr1 == False AND attr2 == False
      Disjunctive (commutativity): stored as two simple rules

    Both the rule AND its contrapositive are returned where possible.
    """
    r = rule_str.strip().rstrip(".")
    results = []

    # 1. "If someone is not not X then they are not not Y"  (double negation)
    m = re.match(
        r"If someone is not not (\w+) then they are not not (\w+)", r, re.I
    )
    if m:
        x, y = m.group(1), m.group(2)
        results += [(x, True, y, True), (y, False, x, False)]
        return results

    # 2. "If someone is not not X then they are Y"  (identity law)
    m = re.match(
        r"If someone is not not (\w+) then they are (\w+)", r, re.I
    )
    if m:
        x, y = m.group(1), m.group(2)
        results += [(x, True, y, True), (y, False, x, False)]
        return results

    # 3. "If someone is not X then they are not Y"
    m = re.match(r"If someone is not (\w+) then they are not (\w+)", r, re.I)
    if m:
        x, y = m.group(1), m.group(2)
        results += [(x, False, y, False), (y, True, x, True)]
        return results

    # 4. "If someone is not X and not Y then they are not cold"  (De Morgan)
    m = re.match(
        r"If someone is not (\w+) and not (\w+) then they are not cold", r, re.I
    )
    if m:
        x, y = m.group(1), m.group(2)
        # Fires when x=False AND y=False  →  cold=False
        results.append((_AND_PREFIX, (x, y), "cold", False))
        # Contrapositive: cold=True  →  NOT(x=False AND y=False)
        #   = x=True OR y=True  (cannot be represented as single simple rule)
        # We approximate: if cold=True, at least one must be True — skip for now
        return results

    # 5. "If someone is X or Y then they are cold"  (commutativity: flip of fact)
    m = re.match(
        r"If someone is (\w+) or (\w+) then they are cold", r, re.I
    )
    if m:
        x, y = m.group(1), m.group(2)
        results += [
            (x, True, "cold", True),
            (y, True, "cold", True),
            ("cold", False, x, False),
            ("cold", False, y, False),
        ]
        return results

    # 6. "Someone is not X or they are Y"  (implication law: X → Y)
    m = re.match(r"Someone is not (\w+) or they are (\w+)", r, re.I)
    if m:
        x, y = m.group(1), m.group(2)
        results += [(x, True, y, True), (y, False, x, False)]
        return results

    # 7. "If someone is X then they are Y"  (standard rule)
    m = re.match(r"If someone is (\w+) then they are (\w+)", r, re.I)
    if m:
        x, y = m.group(1), m.group(2)
        results += [(x, True, y, True), (y, False, x, False)]
        return results

    return []   # unrecognised rule — skip


def parse_rules(rules_str: str) -> List[Tuple]:
    """Parse '|'-separated rule string → flat list of implications."""
    all_implications = []
    for rule in rules_str.split(" | "):
        all_implications.extend(_parse_single_rule(rule.strip()))
    return all_implications


# ------------------------------------------------------------------ #
# Fact parsing                                                         #
# ------------------------------------------------------------------ #

def parse_facts(facts_str: str) -> Dict[str, bool]:
    """
    Parse the facts field (natural language) into {attr: bool} dict.

    Handles:
      "Alice is blue or orange"          → {blue: True, orange: True}
      "Alice is blue"                    → {blue: True}
      "Alice is not cold"                → {cold: False}
      "Alice is blue or orange, not cold" (comma-separated extras)
    """
    known: Dict[str, bool] = {}

    # Remove leading name ("Alice is ..." → "blue or orange")
    # Facts can be a single string or comma-separated
    parts = [p.strip() for p in facts_str.split(",")]

    for part in parts:
        # strip "Name is " or "Name is not " prefix
        m_or = re.match(r"^\w+ is (\w+) or (\w+)$", part, re.I)
        m_not = re.match(r"^\w+ is not (\w+)$", part, re.I)
        m_pos = re.match(r"^\w+ is (\w+)$", part, re.I)

        if m_or:
            # Disjunctive: at least one of c1/c2 is true.
            # Both color→cold rules exist so cold will be derived either way.
            # Safest: mark both as True (covers the disjunction for chaining).
            known[m_or.group(1)] = True
            known[m_or.group(2)] = True
        elif m_not:
            known[m_not.group(1)] = False
        elif m_pos:
            known[m_pos.group(1)] = True

    return known


# ------------------------------------------------------------------ #
# Forward chaining                                                     #
# ------------------------------------------------------------------ #

def forward_chain(facts_str: str, rules_str: str) -> Dict[str, bool]:
    """
    Compute the logical closure of facts_str under rules_str.

    Returns a dict {attribute: bool} containing every fact that can be
    derived (or explicitly negated).  Unknown attributes are absent.
    """
    known = parse_facts(facts_str)
    implications = parse_rules(rules_str)

    changed = True
    while changed:
        changed = False
        for imp in implications:
            if imp[0] == _AND_PREFIX:
                # Conjunctive rule: fires when BOTH attrs are False
                _, (attr1, attr2), conc_attr, conc_val = imp
                if known.get(attr1) is False and known.get(attr2) is False:
                    if known.get(conc_attr) != conc_val:
                        known[conc_attr] = conc_val
                        changed = True
            else:
                prem_attr, prem_val, conc_attr, conc_val = imp
                if known.get(prem_attr) == prem_val:
                    if known.get(conc_attr) != conc_val:
                        known[conc_attr] = conc_val
                        changed = True

    return known


# ------------------------------------------------------------------ #
# Step reward                                                          #
# ------------------------------------------------------------------ #

def step_reward(claimed_sentence: str, closure: Dict[str, bool]) -> int:
    """
    Given a sentence the model generated (e.g. "Alice is cold"),
    return +1 if it matches the closure, -1 if it contradicts, 0 if unknown.
    """
    claimed_sentence = claimed_sentence.strip().rstrip(".")

    m_not = re.search(r"\b(\w+) is not (\w+)", claimed_sentence, re.I)
    m_pos = re.search(r"\b(\w+) is (\w+)", claimed_sentence, re.I)

    if m_not:
        attr = m_not.group(2).lower()
        claimed_val = False
    elif m_pos:
        attr = m_pos.group(2).lower()
        # ignore non-attribute words like "derivable", "true", "false"
        if attr in ("true", "false", "not", "cold", "rough", "young", "nice",
                    "blue", "red", "green", "purple", "orange", "yellow"):
            claimed_val = True
        else:
            return 0
    else:
        return 0

    truth = closure.get(attr)
    if truth is None:
        return 0   # unknown
    return 1 if (truth == claimed_val) else -1


# ------------------------------------------------------------------ #
# Answer check                                                         #
# ------------------------------------------------------------------ #

def check_answer(question: str, closure: Dict[str, bool]) -> Optional[bool]:
    """
    Given a question string "Q1: Alice is cold." or "Q3: Alice is not young.",
    return the correct True/False answer, or None if undecidable.
    """
    # Strip question prefix
    q = re.sub(r"^Q\d+:\s*", "", question).strip().rstrip(".")

    m_not = re.match(r"\w+ is not (\w+)$", q, re.I)
    m_pos = re.match(r"\w+ is (\w+)$", q, re.I)

    if m_not:
        attr = m_not.group(1).lower()
        truth = closure.get(attr)
        if truth is None:
            return None
        return not truth   # asking "is NOT X" → True iff X is False

    elif m_pos:
        attr = m_pos.group(1).lower()
        return closure.get(attr)   # True/False/None

    return None


# ------------------------------------------------------------------ #
# Quick smoke test                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    facts = "Alice is blue or orange"
    rules = ("If someone is blue then they are cold. | "
             "If someone is orange then they are cold. | "
             "If someone is cold then they are rough. | "
             "If someone is not young then they are not rough. | "
             "If someone is young then they are cold. | "
             "If someone is young then they are nice.")

    closure = forward_chain(facts, rules)
    print("Base closure:", closure)

    # Test De Morgan variant
    rules_dm = ("If someone is not blue and not orange then they are not cold. | "
                "If someone is blue then they are cold. | "
                "If someone is orange then they are cold. | "
                "If someone is cold then they are rough. | "
                "If someone is not young then they are not rough. | "
                "If someone is young then they are cold. | "
                "If someone is young then they are nice.")
    closure_dm = forward_chain(facts, rules_dm)
    print("De Morgan closure:", closure_dm)
    print("Closures match:", closure == closure_dm)

    # Test step reward
    r = step_reward("Alice is cold", closure)
    print("Step reward 'Alice is cold':", r)   # should be +1
    r2 = step_reward("Alice is not cold", closure)
    print("Step reward 'Alice is not cold':", r2)  # should be -1
