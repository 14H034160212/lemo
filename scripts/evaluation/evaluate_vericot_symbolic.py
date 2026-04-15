"""
VeriCoT-style Symbolic Baseline for LEMO benchmark.

Implements a forward-chaining propositional theorem prover that:
  1. Parses facts and rules from the LEMO format
  2. Normalises all V4 equivalent rule forms (double-negation, contrapositive,
     De Morgan, implication) back to positive-antecedent form
  3. Runs forward chaining to closure
  4. Answers each question symbolically

This is the "VeriCoT" style baseline: explicit step-by-step premise verification
before deduction, rather than end-to-end neural prediction.

Reference:
  Sun et al. (2023) "Verify-and-Edit: A Knowledge-Enhanced CoT Framework"
  (conceptual analogue for formal logic / symbolic verification setting)

Usage:
  python scripts/evaluation/evaluate_vericot_symbolic.py \
      --test_files data/test_base.csv data/test_variant3.csv ... \
      --output_dir results/baselines/vericot_symbolic
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Rule parsing helpers
# ---------------------------------------------------------------------------

def count_neg(s):
    """Strip leading 'not ' tokens and return (stripped_prop, is_positive)."""
    s = s.strip()
    count = 0
    while s.lower().startswith('not '):
        count += 1
        s = s[4:].strip()
    # even negations = positive; odd = negative
    return s, (count % 2 == 0)


def parse_rules(rules_text):
    """
    Parse rules into normalised (ant_prop, ant_pos, con_prop, con_pos) tuples.
    Returns list of rule tuples.  Also adds the contrapositive of every rule.
    Special 'compound' De Morgan rules are encoded as 5-tuples:
      ('AND', prop1, prop2, polarity1, polarity2, con_prop, con_pos)
    """
    result = []
    if not rules_text:
        return result

    for rule_raw in rules_text.split('|'):
        rule = rule_raw.strip().rstrip('.')
        if not rule:
            continue

        # ── Pattern 1: "If some(thing|one) is ANT then it/they is/are CON"
        m = re.match(
            r'If some(?:thing|one) is (.+?) then (?:it|they) (?:is|are) (.+)',
            rule, re.I)
        if m:
            ant_str, con_str = m.group(1).strip(), m.group(2).strip()

            # De Morgan variant inside pattern 1:
            # "If something is not X and not Y then it is not Z"
            dm = re.match(r'not (.+?) and not (.+)', ant_str, re.I)
            if dm:
                p1 = dm.group(1).strip()
                p2 = dm.group(2).strip()
                con_prop, con_pos = count_neg(con_str)
                result.append(('AND', p1, False, p2, False, con_prop, con_pos))
                continue

            # Commutativity (disjunctive antecedent):
            # "If something is A or B then it is C" → two rules: A→C and B→C
            or_m = re.match(r'(.+?)\s+or\s+(.+)', ant_str, re.I)
            if or_m:
                con_prop, con_pos = count_neg(con_str)
                for part in [or_m.group(1).strip(), or_m.group(2).strip()]:
                    ant_prop, ant_pos = count_neg(part)
                    result.append((ant_prop, ant_pos, con_prop, con_pos))
                    result.append((con_prop, not con_pos, ant_prop, not ant_pos))
                continue

            ant_prop, ant_pos = count_neg(ant_str)
            con_prop, con_pos = count_neg(con_str)
            result.append((ant_prop, ant_pos, con_prop, con_pos))
            # Add contrapositive: ¬con → ¬ant
            result.append((con_prop, not con_pos, ant_prop, not ant_pos))
            continue

        # ── Pattern 2: "Some(thing|one) is not X or it/they is/are Y"
        #    ¬X ∨ Y  ≡  X → Y
        m = re.match(
            r'Some(?:thing|one) is not (.+?) or (?:it|they) (?:is|are) (.+)',
            rule, re.I)
        if m:
            prop_x = m.group(1).strip()
            con_str = m.group(2).strip()
            con_prop, con_pos = count_neg(con_str)
            # X → Y
            result.append((prop_x, True, con_prop, con_pos))
            # ¬Y → ¬X  (contrapositive)
            result.append((con_prop, not con_pos, prop_x, False))
            continue

    return result


def parse_facts(facts_text):
    """
    Return dict {prop: bool} from facts string.
    Handles: "entity is A or B", "entity is not A", "entity is A | entity is B"
    """
    known = {}
    if not facts_text:
        return known

    for fact in facts_text.split('|'):
        fact = fact.strip().rstrip('.')
        if not fact:
            continue
        # Match "X is/are PROP_EXPR"
        m = re.match(r'.+?\s+(?:is|are)\s+(.+)$', fact, re.I)
        if not m:
            continue
        prop_expr = m.group(1).strip()

        # Disjunction: "A or B" → both True
        if re.search(r'\bor\b', prop_expr, re.I):
            parts = re.split(r'\bor\b', prop_expr, flags=re.I)
            for p in parts:
                prop, pos = count_neg(p.strip())
                if prop:
                    known[prop] = pos
        else:
            prop, pos = count_neg(prop_expr)
            if prop:
                known[prop] = pos
    return known


def forward_chain(known, rules):
    """Apply rules to known facts until fixpoint. Modifies known in-place."""
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if len(rule) == 4:
                ant_prop, ant_pos, con_prop, con_pos = rule
                if known.get(ant_prop) == ant_pos:
                    if known.get(con_prop) != con_pos:
                        known[con_prop] = con_pos
                        changed = True
            elif len(rule) == 7 and rule[0] == 'AND':
                # ('AND', p1, pol1, p2, pol2, con_prop, con_pos)
                _, p1, pol1, p2, pol2, con_prop, con_pos = rule
                if known.get(p1) == pol1 and known.get(p2) == pol2:
                    if known.get(con_prop) != con_pos:
                        known[con_prop] = con_pos
                        changed = True


def symbolic_predict(facts_text, rules_text, question_text):
    """Return 'T' or 'F'."""
    known = parse_facts(facts_text)
    rules = parse_rules(rules_text)
    forward_chain(known, rules)

    # Parse question: "Qn: entity is [not] PROP."
    m = re.match(r'Q\d+:\s+.+?\s+(?:is|are)\s+(.+?)\.?$', question_text, re.I)
    if not m:
        return 'F'

    q_prop, q_pos = count_neg(m.group(1).strip())
    derived = known.get(q_prop)

    if derived is not None:
        return 'T' if derived == q_pos else 'F'
    # Unknown → default False (closed-world assumption)
    return 'F'


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_file(csv_path, output_dir, split_name):
    rows_in = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    correct = 0
    rows_out = []
    for row in rows_in:
        facts    = row.get('facts', '')
        rules    = row.get('rules', '')
        question = row.get('question', '')
        gt       = row.get('ground_truth', '').strip()

        pred = symbolic_predict(facts, rules, question)
        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        rows_out.append({**row, 'symbolic_pred': pred, 'correct': int(is_correct)})

    acc = correct / len(rows_in) if rows_in else 0.0
    print(f'  [{split_name}]  accuracy = {acc:.4f}  ({correct}/{len(rows_in)})')

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'vericot_{split_name}_predictions.csv')
    if rows_out:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    return split_name, acc, correct, len(rows_in)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_files', nargs='+', required=True)
    parser.add_argument('--output_dir', default='results/baselines/vericot_symbolic')
    args = parser.parse_args()

    # Use prediction files (per-question format), not raw data files
    # The prediction CSVs have columns: facts, rules, question, ground_truth
    results = []
    print('\n=== VeriCoT Symbolic Baseline ===\n')
    for path in args.test_files:
        split = os.path.splitext(os.path.basename(path))[0]
        # Strip leading model prefix if any (e.g. qwen3_base → base)
        for pfx in ['qwen3_rlvf_', 'qwen3_lire_', 'qwen3_', 'qwen_rlvf_', 'qwen_lire_', 'qwen_', 'llama_', 'bert_']:
            if split.startswith(pfx):
                split = split[len(pfx):]
                break
        # Strip trailing _predictions
        split = split.replace('_predictions', '')
        name, acc, correct, total = evaluate_file(path, args.output_dir, split)
        results.append((name, acc, correct, total))

    print('\n  === Summary ===')
    print(f'  {"Split":<40} {"Accuracy":>10} {"Correct":>10} {"Total":>8}')
    print('  ' + '-' * 72)
    base_acc = None
    for name, acc, correct, total in results:
        delta = f'  {acc - base_acc:+.4f}' if base_acc is not None else '  (base)'
        if base_acc is None:
            base_acc = acc
        print(f'  {name:<40} {acc:>10.4f} {correct:>10} {total:>8}{delta}')

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, 'vericot_accuracy_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'accuracy', 'correct', 'total'])
        for name, acc, correct, total in results:
            writer.writerow([name, f'{acc:.6f}', correct, total])
    print(f'\n  Saved: {summary_path}')
    print('  ✅ VeriCoT evaluation FINISHED.\n')


if __name__ == '__main__':
    main()
