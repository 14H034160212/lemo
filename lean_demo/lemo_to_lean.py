"""
lemo_to_lean.py
===============
Translate LEMO benchmark rows (CSV) into Lean 4 theorems, and
dispatch them to the Lean kernel for step-level verification.

Each CSV row contains:
  - facts:   natural-language statements ("X is blue or orange, not cold")
  - rules:   pipe-separated if-then rules
  - questions: pipe-separated queries
  - answers: pipe-separated T/F

The translator:
  1. Extracts all predicates appearing in facts/rules/questions.
  2. Declares each predicate as a Lean `Prop`.
  3. Emits hypotheses for facts and rules.
  4. For each (question, ground-truth) pair, generates a Lean theorem
     that encodes the expected answer, plus a proof script produced by
     forward chaining.
  5. Runs `lean` and records the kernel's accept/reject verdict.

A Lean acceptance of the generated proof means:
  "The ground truth is formally derivable from the premises in the
   Lean 4 kernel, confirming soundness of the multi-step reasoning."
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Reuse the project's forward-chaining oracle to generate proof skeletons
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from scripts.utils.forward_chain import forward_chain, check_answer   # noqa: E402


# --------------------------------------------------------------------- #
# Parsing                                                                #
# --------------------------------------------------------------------- #

# Entity can be a single word or "the X" style. We capture everything
# up to " is ".
_FACT_RE_OR  = re.compile(r"^\s*(.+?)\s+is\s+(\w+)\s+or\s+(\w+)\s*$", re.I)
_FACT_RE_NOT = re.compile(r"^\s*(.+?)\s+is\s+not\s+(\w+)\s*$", re.I)
_FACT_RE_POS = re.compile(r"^\s*(.+?)\s+is\s+(\w+)\s*$", re.I)

_RULE_POS = re.compile(r"^If\s+(?:something|it)\s+is\s+(\w+)\s+then\s+(?:something|it)\s+is\s+(\w+)\.?$", re.I)
_RULE_NEG = re.compile(r"^If\s+(?:something|it)\s+is\s+not\s+(\w+)\s+then\s+(?:something|it)\s+is\s+not\s+(\w+)\.?$", re.I)

_Q_POS = re.compile(r"^Q\d+:\s*(.+?)\s+is\s+(\w+)\.?\s*$", re.I)
_Q_NEG = re.compile(r"^Q\d+:\s*(.+?)\s+is\s+not\s+(\w+)\.?\s*$", re.I)


@dataclass
class ParsedRow:
    entity: str                       # "Eastport"
    pos_facts: List[str]              # attributes asserted True
    neg_facts: List[str]              # attributes asserted False
    disj_facts: List[Tuple[str, str]] # (A, B) where fact is "X is A or B"
    rules: List[Tuple[bool, str, bool, str]]
    # (prem_pos, prem_attr, conc_pos, conc_attr)
    questions: List[Tuple[bool, str, bool]]
    # (question_is_positive, attr, expected_answer_T?)


def parse_row(row: Dict[str, str]) -> Optional[ParsedRow]:
    """Parse a CSV row into a ParsedRow. Returns None if anything fails."""
    facts_str = str(row.get("facts", "")).strip()
    rules_str = str(row.get("rules", "")).strip()
    qs_str = str(row.get("questions", "")).strip()
    ans_str = str(row.get("answers", "")).strip()

    entity = None
    pos_facts, neg_facts, disj = [], [], []

    # Facts may be comma-or-pipe separated
    for piece in re.split(r"[|,]", facts_str):
        p = piece.strip()
        if not p:
            continue
        m = _FACT_RE_OR.match(p)
        if m:
            entity = entity or m.group(1)
            disj.append((m.group(2).lower(), m.group(3).lower()))
            continue
        m = _FACT_RE_NOT.match(p)
        if m:
            entity = entity or m.group(1)
            neg_facts.append(m.group(2).lower())
            continue
        m = _FACT_RE_POS.match(p)
        if m:
            entity = entity or m.group(1)
            pos_facts.append(m.group(2).lower())
            continue
        # Unrecognized fact form — skip
        return None

    if entity is None:
        return None

    # Rules
    rules: List[Tuple[bool, str, bool, str]] = []
    for rule in rules_str.split(" | "):
        r = rule.strip()
        m = _RULE_NEG.match(r)
        if m:
            rules.append((False, m.group(1).lower(), False, m.group(2).lower()))
            continue
        m = _RULE_POS.match(r)
        if m:
            rules.append((True, m.group(1).lower(), True, m.group(2).lower()))
            continue
        # Skip complex De Morgan / disjunctive-premise rules for this demo
        return None

    # Questions + answers
    qs = [q.strip() for q in qs_str.split(" | ") if q.strip()]
    ans = [a.strip().upper() for a in ans_str.split("|")]
    ans = [a.strip() for a in ans]
    if len(qs) != len(ans):
        return None

    questions: List[Tuple[bool, str, bool]] = []
    for q, a in zip(qs, ans):
        m = _Q_NEG.match(q)
        if m:
            questions.append((False, m.group(2).lower(), a == "T"))
            continue
        m = _Q_POS.match(q)
        if m:
            questions.append((True, m.group(2).lower(), a == "T"))
            continue
        return None

    return ParsedRow(entity, pos_facts, neg_facts, disj, rules, questions)


# --------------------------------------------------------------------- #
# Lean emission                                                          #
# --------------------------------------------------------------------- #

_LEAN_PROLOGUE = r"""
-- Auto-generated from LEMO benchmark row
namespace LEMO.Auto
"""

_LEAN_EPILOGUE = r"""
end LEMO.Auto
"""


def _collect_attrs(p: ParsedRow) -> List[str]:
    attrs: Set[str] = set(p.pos_facts) | set(p.neg_facts)
    for a, b in p.disj_facts:
        attrs.update([a, b])
    for _, a, _, b in p.rules:
        attrs.update([a, b])
    for _, a, _ in p.questions:
        attrs.add(a)
    return sorted(attrs)


def _rule_name(i: int) -> str:
    return f"rule{i}"


def emit_theorem(p: ParsedRow, q_idx: int) -> Tuple[str, bool, bool]:
    """
    Emit a complete Lean 4 script that encodes premises + a theorem
    stating the ground truth for question q_idx. Returns:
       (lean_source, expected_provable, gt_answer_T)
    where `expected_provable` is True iff we believe the resulting theorem
    should type-check in Lean (i.e. the ground truth is sound).
    """
    attrs = _collect_attrs(p)
    q_pos, q_attr, gt_T = p.questions[q_idx]

    lines = [_LEAN_PROLOGUE.rstrip()]

    # Declare predicates
    for a in attrs:
        lines.append(f"axiom {a} : Prop")
    lines.append("open Classical")
    lines.append("")

    # Build the statement: premises -> goal
    # Premises:
    hyps: List[str] = []
    hname = 0
    for a in p.pos_facts:
        hyps.append(f"(h{hname} : {a})"); hname += 1
    for a in p.neg_facts:
        hyps.append(f"(h{hname} : ¬ {a})"); hname += 1
    for a, b in p.disj_facts:
        hyps.append(f"(h{hname} : {a} ∨ {b})"); hname += 1
    for i, (pp, pa, cp, ca) in enumerate(p.rules):
        lhs = pa if pp else f"¬ {pa}"
        rhs = ca if cp else f"¬ {ca}"
        hyps.append(f"(h{hname} : {lhs} → {rhs})"); hname += 1

    # Compute forward_chain closure to decide proof style
    # Convert ParsedRow back to facts_str/rules_str for forward_chain
    facts_strs: List[str] = []
    if p.pos_facts:
        facts_strs.append(f"{p.entity} is " + " or ".join(p.pos_facts)
                          if len(p.pos_facts) == 1
                          else ", ".join(f"{p.entity} is {a}" for a in p.pos_facts))
    for a in p.neg_facts:
        facts_strs.append(f"{p.entity} is not {a}")
    for a, b in p.disj_facts:
        facts_strs.append(f"{p.entity} is {a} or {b}")
    facts_str = ", ".join(facts_strs)
    rules_str = " | ".join(
        f"If someone is {'not ' if not pp else ''}{pa} then they are "
        f"{'not ' if not cp else ''}{ca}."
        for pp, pa, cp, ca in p.rules
    )
    try:
        closure = forward_chain(facts_str, rules_str)
    except Exception:
        closure = {}

    # Determine goal polarity according to ground truth
    # If gt_T:  goal = (query polarity as stated)
    # If !gt_T: goal = ¬(query polarity as stated), which means we want to
    #          prove the negation.
    query_prop = q_attr if q_pos else f"¬ {q_attr}"
    goal = query_prop if gt_T else f"¬ ({query_prop})"

    # Short-circuit: if forward_chain says the closure is contradictory
    # (some attr has both True and False settings by rule firing), then
    # goal = False is classically provable, so we emit `∀ Q, Q`-style proof.
    # But for simplicity in this demo we just emit the goal as computed
    # and rely on `tauto` or manual proof to close it.
    lines.append(f"theorem _goal_q{q_idx + 1}")
    for h in hyps:
        lines.append(f"    {h}")
    lines.append(f"    : {goal} := by")

    # ----- Emit a manual forward-chaining proof -----
    # For each disjunctive fact we case-split; within each branch we
    # forward-chain via `have` statements until the query (or its
    # negation) is derivable, then close with `exact`.
    disj_count = len(p.disj_facts)
    disj_start = len(p.pos_facts) + len(p.neg_facts)

    def _proof_branch(initial_attrs: Dict[str, bool], indent: str) -> List[str]:
        """
        Emit a proof body assuming `initial_attrs` are known.
        Chases rules forward until it can close the goal.
        Returns the list of indented lines.
        """
        known: Dict[str, bool] = dict(initial_attrs)
        # map attr -> Lean-term that proves it
        terms: Dict[Tuple[str, bool], str] = {}
        # seed from positive/negative facts
        hid = 0
        for a in p.pos_facts:
            terms[(a, True)] = f"h{hid}"; hid += 1
        for a in p.neg_facts:
            terms[(a, False)] = f"h{hid}"; hid += 1
        # disj hypotheses don't produce direct terms, but within each
        # branch we will add the local binding
        hid += disj_count  # skip disj hyp slots
        # rule hypotheses start at this hid
        rule_hid0 = hid

        # Merge in the branch-local binding (passed via initial_attrs)
        for attr, val in initial_attrs.items():
            terms.setdefault((attr, val), f"h_{attr}_{'T' if val else 'F'}")

        out: List[str] = []
        fresh_count = 0
        changed = True
        while changed:
            changed = False
            for i, (pp, pa, cp, ca) in enumerate(p.rules):
                rh = f"h{rule_hid0 + i}"
                if (pa, pp) in terms and (ca, cp) not in terms:
                    # apply rule `rh` to proof of `pa`
                    arg = terms[(pa, pp)]
                    nm = f"f{fresh_count}"; fresh_count += 1
                    sign = "" if cp else "¬ "
                    out.append(f"{indent}have {nm} : {sign}{ca} := {rh} {arg}")
                    terms[(ca, cp)] = nm
                    changed = True

        # Now try to close the goal
        goal_attr = q_attr
        goal_pos = q_pos if gt_T else (not q_pos)
        # If gt_T=False, goal was `¬ (query)`; we need to emit a proof of that.
        if gt_T:
            key = (goal_attr, goal_pos)
            if key in terms:
                out.append(f"{indent}exact {terms[key]}")
            else:
                out.append(f"{indent}sorry -- could not derive {'¬ ' if not goal_pos else ''}{goal_attr}")
        else:
            # Want to prove ¬ P where P = (goal_attr if q_pos else ¬ goal_attr)
            # Strategy: assume hyp, derive False.
            neg_key = (goal_attr, not q_pos)  # opposite polarity
            # If we already know the opposite, take intro-and-absurd
            if neg_key in terms:
                if q_pos:
                    # goal: ¬ goal_attr; we have terms[(goal_attr, False)]
                    out.append(f"{indent}intro h_pos")
                    out.append(f"{indent}exact absurd h_pos {terms[neg_key]}")
                else:
                    # goal: ¬ (¬ goal_attr); we have terms[(goal_attr, True)]
                    out.append(f"{indent}intro h_not; exact absurd {terms[neg_key]} h_not")
            else:
                out.append(f"{indent}sorry -- could not derive negation")
        return out

    proof_body: List[str] = []
    if disj_count == 0:
        proof_body.extend(_proof_branch({}, "  "))
    elif disj_count == 1:
        a, b = p.disj_facts[0]
        hi = disj_start
        proof_body.append(f"  rcases h{hi} with ha | hb")
        proof_body.append(f"  ·")
        # In branch `ha`, the attr `a` is True and bound to the case name `ha`
        # But `rcases h0 with ha | hb` in Lean 4 binds first arm to `ha`
        # where ha : <attr_a>. We rename proof term accordingly.
        proof_body.append(f"    have h_{a}_T : {a} := ha")
        proof_body.extend(_proof_branch({a: True}, "    "))
        proof_body.append(f"  ·")
        proof_body.append(f"    have h_{b}_T : {b} := hb")
        proof_body.extend(_proof_branch({b: True}, "    "))
    else:
        # More than one disjunctive fact — complex branching; skip for demo
        proof_body.append("  sorry -- multi-disjunction case not implemented")
    lines.extend(proof_body)

    lines.append("")
    lines.append(_LEAN_EPILOGUE.rstrip())

    # We mark `expected_provable = True` iff forward_chain's closure
    # agrees with the row's ground truth.
    fc_verdict = check_answer(f"Q?: {p.entity} is {'not ' if not q_pos else ''}{q_attr}.", closure)
    # Reinterpret: answer is T iff query polarity matches closure
    expected = (fc_verdict is True and gt_T) or (fc_verdict is False and not gt_T) or (fc_verdict is None)
    return "\n".join(lines), expected, gt_T


# --------------------------------------------------------------------- #
# Dispatch                                                               #
# --------------------------------------------------------------------- #

def verify_with_lean(
    script: str,
    lean_bin: str = "lean",
    timeout: float = 30.0,
) -> Tuple[int, str]:
    """Write `script` to a temp file and run Lean. Returns (reward, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = Path(f.name)

    try:
        res = subprocess.run(
            [lean_bin, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0, "timeout"
    except FileNotFoundError:
        return 0, "lean not found"
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    combined = (res.stderr or "") + "\n" + (res.stdout or "")
    if "sorry" in combined.lower() or "admit" in combined.lower():
        return -1, combined
    if res.returncode == 0:
        return 1, combined
    return -1, combined


if __name__ == "__main__":
    # Smoke test: translate one row and run it
    import csv
    with open(_ROOT / "data_v2" / "test_base.csv") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    p = parse_row(row)
    print("Parsed:", p)
    script, expected, gt = emit_theorem(p, 0)
    print("--- Lean script ---")
    print(script)
    print("--- end ---")
    r, err = verify_with_lean(script)
    print(f"\nVerdict: r = {r:+d}; expected_provable = {expected}")
    if r != 1 and err:
        print("Lean output:\n" + "\n".join(err.splitlines()[:20]))
