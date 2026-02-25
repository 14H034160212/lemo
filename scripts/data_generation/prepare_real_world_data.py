import os
import json
from datasets import load_dataset
import pandas as pd

# Create data directory if not exists
os.makedirs("data/real_world", exist_ok=True)

def prepare_logicnli():
    print("Downloading tasksource/LogicNLI...")
    try:
        # Load LogicNLI (it might be a subset of tasksource)
        # Using the direct path if available or via tasksource
        dataset = load_dataset("tasksource/LogicNLI", split="train[:500]") 
    except Exception as e:
        print(f"Direct load failed: {e}. Trying generic load...")
        # Fallback or alternative dataset if specific config fails
        return
    
    # LogicNLI structure usually has: premise, hypothesis, label
    # We need to map this to our format:
    # Input: Facts (Premise) | Rules (None/Implicit) | Question (Hypothesis)
    # Output: True/False/Unknown
    
    formatted_data = []
    for row in dataset:
        # Map labels: entailment=True, contradiction=False, neutral=False/Unknown
        # LogicNLI typically has entailment/contradiction
        label_map = {0: "True", 1: "False", 2: "False"} # Simplifying neutral to False for now or skip
        
        if row['label'] == -1: continue 
        
        target = label_map.get(row['label'], "False")
        
        # Conflict-Aware Template Construction
        # We treat Premise as Facts+Rules
        input_text = f"Facts: {row['premise']}\nRules: (Implicit in facts)\nQuestion: {row['hypothesis']}\nThink step by step."
        
        formatted_data.append({
            "input_text": input_text,
            "target_text": target, # Evaluation target
            "original_label": row['label']
        })
    
    df = pd.DataFrame(formatted_data)
    df.to_csv("data/real_world/logicnli_eval.csv", index=False)
    print(f"Saved {len(df)} LogicNLI samples to data/real_world/logicnli_eval.csv")

def prepare_mnli_contradiction():
    print("Downloading MNLI (Contradiction subset)...")
    dataset = load_dataset("glue", "mnli", split="validation_matched[:500]")
    
    formatted_data = []
    for row in dataset:
        # MNLI: 0=entailment, 1=neutral, 2=contradiction
        # We specifically want to see if our model handles contradictions (Label 2)
        
        if row['label'] == 2: # Contradiction
            target = "False"
            input_text = f"Facts: {row['premise']}\nQuestion: {row['hypothesis']}\nThink step by step."
            
            formatted_data.append({
                "input_text": input_text,
                "target_text": target,
                "type": "contradiction"
            })
        elif row['label'] == 0: # Entailment (Base Control)
            target = "True"
            input_text = f"Facts: {row['premise']}\nQuestion: {row['hypothesis']}\nThink step by step."

            formatted_data.append({
                "input_text": input_text,
                "target_text": target,
                "type": "entailment"
            })

    df = pd.DataFrame(formatted_data)
    df.to_csv("data/real_world/mnli_eval.csv", index=False)
    print(f"Saved {len(df)} MNLI samples to data/real_world/mnli_eval.csv")

if __name__ == "__main__":
    prepare_logicnli()
    prepare_mnli_contradiction()
