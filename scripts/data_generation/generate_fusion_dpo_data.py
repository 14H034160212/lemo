
import json
import random
import os
import pandas as pd
import csv

BASE_TRAIN_FILE = "data/train.csv"
MIXED_TRAIN_FILE = "data/train_mixed.csv"
OUTPUT_FILE = "data/train_fusion_dpo.jsonl"

def generate_fusion_dpo():
    if not os.path.exists(BASE_TRAIN_FILE) or not os.path.exists(MIXED_TRAIN_FILE):
        return

    base_rows = list(csv.DictReader(open(BASE_TRAIN_FILE)))
    mixed_df = pd.read_csv(MIXED_TRAIN_FILE)
    
    v2_rows = mixed_df[mixed_df['type'] == 'aug_variant2'].to_dict('records')
    v3_rows = mixed_df[mixed_df['type'] == 'aug_variant3'].to_dict('records')
    
    dpo_samples = []
    
    # 1. Base Logic Preferences
    for row in base_rows[:200]: 
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(' | ')
        answers = row['answers'].split(' | ')
        name = facts.split()[0]
        
        for q, a in zip(questions, answers):
            ans_text = "True" if a.strip() == "T" else "False"
            opp_text = "False" if ans_text == "True" else "True"
            
            # CoT preference (Conflict-Aware)
            dpo_samples.append({
                "prompt": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step.",
                "chosen": f"Reasoning: Step 1: Verify facts. Facts are consistent. Step 2: Apply rules. Rules imply {name} follows the target logical chain. Therefore, {q} is {ans_text}. Answer: {ans_text}",
                "rejected": f"Reasoning: Step 1: Verify facts. Facts are consistent. Step 2: Apply rules. Logic gap found. Answer: {opp_text}"
            })

    # 2. Variant 2 Preferences (Missing Rules)
    for row in v2_rows[:100]:
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(' | ')
        answers = row['answers'].split(' | ')
        
        for q, a in zip(questions, answers):
            if a.strip() == "F": 
                dpo_samples.append({
                    "prompt": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step.",
                    "chosen": "Reasoning: Step 1: Verify facts. Facts are consistent. Step 2: Apply rules. Necessary rule for this inference is missing. Answer: False",
                    "rejected": "Reasoning: Step 1: Verify facts. Facts are consistent. Step 2: Apply rules. Blindly following generic rules. Answer: True"
                })

    # 3. Variant 3 Preferences (Contradictions)
    for row in v3_rows[:100]:
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(' | ')
        answers = row['answers'].split(' | ')
        
        for q, a in zip(questions, answers):
            dpo_samples.append({
                "prompt": f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step.",
                "chosen": "Reasoning: Step 1: Verify facts. Conflict detected! The facts contain a direct contradiction. Step 2: Stop. Result is invalid. Answer: False",
                "rejected": "Reasoning: Step 1: Verify facts. Facts are consistent. Step 2: Apply rules. Rules imply true. Answer: True"
            })
            
    print(f"Generated {len(dpo_samples)} Fusion DPO samples.")
    with open(OUTPUT_FILE, "w") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_fusion_dpo()
