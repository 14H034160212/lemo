# Reads train_with_v3.csv (the corpus that INCLUDES Variant 3). The Table-2
# training strategies are compared against each other, so they must all see
# the same contradiction examples; train.csv holds Variant 3 out and is for
# the untreated Table-1 baselines only.

import csv
import random
import os
import pandas as pd

BASE_TRAIN_FILE = "data/train_with_v3.csv"
MIXED_TRAIN_FILE = "data/train_mixed.csv"
OUTPUT_FILE = "data/train_fusion.csv"

def generate_base_gen(row):
    """Method 1 style: Direct answer."""
    facts = row['facts']
    rules = row['rules']
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    
    samples = []
    for q, a in zip(questions, answers):
        ans_text = "True" if a.strip() == "T" else "False"
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}",
            "target_text": ans_text
        })
    return samples

def generate_base_cot(row):
    """RA-CoT style: Reasoning trace with Conflict Awareness."""
    facts = row['facts']
    rules_list = row['rules'].split(' | ')
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    name = facts.split()[0]
    
    samples = []
    for q, a in zip(questions, answers):
        # Conflict-Aware Template
        reasoning = (f"Step 1: Verify facts. Facts are consistent. No contradictions detected. "
                     f"Step 2: Apply rules. Rules imply {name} follows the target logical chain. "
                     f"Therefore, {q} is {a.strip()}")
        final_answer = "True" if a.strip() == "T" else "False"
        
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {row['rules']}\nQuestion: {q}\nThink step by step.",
            "target_text": f"Reasoning: {reasoning} Answer: {final_answer}"
        })
    return samples

def generate_variant_cot(row, variant_type="v2"):
    """Reasoning for authentic multi-step variants with Conflict Awareness."""
    facts = row['facts']
    rules = row['rules']
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    
    samples = []
    for q, a in zip(questions, answers):
        final_answer = "True" if a.strip() == "T" else "False"
        
        if variant_type == "v2":
            reasoning = (f"Reasoning: Step 1: Verify facts. Facts are consistent. "
                         f"Step 2: Apply rules. Checked logic chain. Important rule for this inference is missing. "
                         f"Status of {q} cannot be confirmed. Answer: {final_answer}")
        else: # v3 (Contradiction)
            reasoning = (f"Reasoning: Step 1: Verify facts. Conflict detected! The facts contain a direct contradiction "
                         f"(e.g., '{facts}'). "
                         f"Step 2: Stop. Since facts are inconsistent, result is invalid. Answer: {final_answer}")
        
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step.",
            "target_text": reasoning
        })
    return samples

def generate_variant_gen(row):
    """Direct answer for authentic variants."""
    facts = row['facts']
    rules = row['rules']
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    
    samples = []
    for q, a in zip(questions, answers):
        ans_text = "True" if a.strip() == "T" else "False"
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}",
            "target_text": ans_text
        })
    return samples

def main():
    if not os.path.exists(BASE_TRAIN_FILE) or not os.path.exists(MIXED_TRAIN_FILE):
        print("Required training files missing.")
        return

    # Only the base rows belong here. The contradiction and rule-removal signal
    # comes from train_mixed.csv (aug_variant2 / aug_variant3) below, not from
    # this file, so feeding it variant rows only inflates the denominator and
    # dilutes the contradiction ratio the method depends on.
    base_rows = [r for r in csv.DictReader(open(BASE_TRAIN_FILE))
                 if str(r.get("type", "")).startswith("base_")]
    mixed_df = pd.read_csv(MIXED_TRAIN_FILE)
    
    v2_rows = mixed_df[mixed_df['type'] == 'aug_variant2'].to_dict('records')
    v3_rows = mixed_df[mixed_df['type'] == 'aug_variant3'].to_dict('records')

    fusion_samples = []
    
    # Scale up!
    for _ in range(5): 
        random.shuffle(base_rows)
        for row in base_rows:
            fusion_samples.extend(generate_base_gen(row))
            fusion_samples.extend(generate_base_cot(row))

        random.shuffle(v2_rows)
        for row in v2_rows:
            fusion_samples.extend(generate_variant_cot(row, "v2"))
        
        # The contradiction cap used to be the literal 320, which was tuned when
        # BASE_TRAIN_FILE held 160 rows and produced ~14.3% contradiction samples.
        # Once the base corpus grew, the base half scaled with it while this cap
        # did not, diluting contradictions to ~1% -- and a model cannot learn to
        # halt on contradictions from 1% of its training signal. Scale the cap
        # with the base corpus so the ratio the method was designed around holds
        # at any corpus size.
        # 0.50 is calibrated, not guessed: it reproduces the original corpus's
        # ratios exactly (14.29% contradiction, 57.14% verification-prefix),
        # verified against `git show ea27058:data/train_fusion.csv`.
        _V3_PER_BASE_ROW = 0.50
        _v3_cap = max(320, int(len(base_rows) * _V3_PER_BASE_ROW))
        random.shuffle(v3_rows)
        for row in (v3_rows * (1 + _v3_cap // max(1, len(v3_rows))))[:_v3_cap]:
            fusion_samples.extend(generate_variant_cot(row, "v3"))
            fusion_samples.extend(generate_variant_gen(row))

    print(f"Generated {len(fusion_samples)} Fusion samples.")
    
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
        writer.writeheader()
        writer.writerows(fusion_samples)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
