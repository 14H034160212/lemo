
import csv
import random
import os
import pandas as pd

BASE_TRAIN_FILE = "data/train.csv"
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

    base_rows = list(csv.DictReader(open(BASE_TRAIN_FILE)))
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
        
        random.shuffle(v3_rows)
        for row in v3_rows[:320]: # Limit scale
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
