
import csv
import json
import random

INPUT_FILE = "data/train.csv"
OUTPUT_FILE = "data/train_cot.csv"

def generate_reasoning(row):
    facts = row['facts']
    rules = row['rules'].split(' | ')
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    
    # Simple semantic parsing for this specific dataset structure
    # Fact: "Name is color1 or color2"
    name = facts.split()[0]
    
    # We assume the standard chain:
    # Color -> Cold -> Rough -> Young -> Nice
    # And Rule 5: Young -> Cold (back loop)
    # Rule 3: not Young -> not Rough (contrapositive)
    
    samples = []
    
    for q, a in zip(questions, answers):
        # q: "Name is attribute."
        # Extract attribute
        parts = q.strip(".").split()
        target_attr = parts[-1] # cold, rough, young, nice, green, etc.
        is_negated = "not" in q
        
        chain = []
        chain.append(f"Fact: {facts}.")
        
        # Step 1: Colors imply Cold
        chain.append(f"Rule: If {name} is {rules[0].split()[3]} then {name} is cold.") # Green->Cold
        chain.append(f"Rule: If {name} is {rules[1].split()[3]} then {name} is cold.") # Blue->Cold
        chain.append(f"Therefore, {name} is cold.")
        
        # Step 2: Cold -> Rough
        chain.append(f"Rule: If {name} is cold then {name} is rough.")
        chain.append(f"Therefore, {name} is rough.")
        
        # Step 3: Rough -> Young
        chain.append(f"Rule: If {name} is rough then {name} is young.")
        chain.append(f"Therefore, {name} is young.")
        
        # Step 4: Young -> Nice
        chain.append(f"Rule: If {name} is young then {name} is nice.")
        chain.append(f"Therefore, {name} is nice.")
        
        reasoning = " ".join(chain)
        
        # Determine Answer
        # Valid attributes: cold, rough, young, nice
        valid_attrs = ["cold", "rough", "young", "nice"]
        
        final_answer = "False"
        if a.strip() == "T":
            final_answer = "True"
            # For True samples, the reasoning supports it.
            # We can cut the reasoning chain short if we want, but full chain is fine.
        else:
            # False sample.
            # E.g. "Anne is not cold" (False)
            # Reasoning proves Anne IS cold. So "Anne is not cold" is False.
            # E.g. "Anne is green" (False/Unknown - actually Unknown in strict logic if "Green or Blue", but dataset might label F?)
            # Let's trust the GT 'a'.
            pass

        full_output = f"Reasoning: {reasoning} Answer: {final_answer}"
        
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {row['rules']}\nQuestion: {q}\nThink step by step.",
            "target_text": full_output
        })
        
    return samples

def generate_cot_data():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    cot_samples = []
    for row in rows:
        if row['type'] in ['base_positive', 'base_negative']:
            cot_samples.extend(generate_reasoning(row))
            
    print(f"Generated {len(cot_samples)} CoT samples.")
    
    # Save as CSV with input_text, target_text (compatible with stage2_train script)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
        writer.writeheader()
        writer.writerows(cot_samples)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_cot_data()
