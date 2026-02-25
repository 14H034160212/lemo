
import csv
import json
import random
import os

# Source files from 80 training groups (avoiding test leakage)
BASE_TRAIN_FILE = "data/train.csv"
OUTPUT_FILE = "data/train_ra_cot.csv"

def generate_base_reasoning(row):
    facts = row['facts']
    rules_list = row['rules'].split(' | ')
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')
    name = facts.split()[0]
    
    samples = []
    for q, a in zip(questions, answers):
        parts = q.strip(".").split()
        target_attr = parts[-1]
        
        # Build standard chain
        chain = [f"Fact: {facts}."]
        chain.append(f"Rule 1 & 2: Colors ({rules_list[0].split()[3]}, {rules_list[1].split()[3]}) imply {name} is cold.")
        chain.append(f"Rule 3: Cold implies {name} is rough.")
        chain.append(f"Rule 4: Not young implies not rough (Contrapositive: Rough implies young).")
        chain.append(f"Rule 5: Young implies {name} is nice.")
        
        reasoning = " ".join(chain)
        final_answer = "True" if a.strip() == "T" else "False"
        
        samples.append({
            "input_text": f"Facts: {facts}\nRules: {row['rules']}\nQuestion: {q}\nThink step by step.",
            "target_text": f"Reasoning: {reasoning} Answer: {final_answer}"
        })
    return samples

def generate_variant2_reasoning(row):
    """Reasoning for missing rules."""
    facts = row['facts']
    rules_list = row['rules'].split(' | ')
    questions = row['questions'].split(' | ')
    name = facts.split()[0]
    
    # We simulate rule removal for training
    # Remove Rule 3 (Cold -> Rough)
    modified_rules = " | ".join(rules_list[:2] + rules_list[3:])
    
    samples = []
    # In Variant 2, queries about qualities AFTER the missing link should be False/Unknown
    for q in questions:
        if "rough" in q or "young" in q or "nice" in q:
            reasoning = (f"Reasoning: Fact: {facts}. Rules 1 & 2 confirm {name} is cold. "
                         f"However, the rule linking 'cold' to 'rough' is missing from the ruleset. "
                         f"Therefore, we cannot infer subsequent traits. Answer: False")
            samples.append({
                "input_text": f"Facts: {facts}\nRules: {modified_rules}\nQuestion: {q}\nThink step by step.",
                "target_text": reasoning
            })
    return samples

def generate_variant3_reasoning(row):
    """Reasoning for contradictions."""
    facts = row['facts']
    rules_list = row['rules'].split(' | ')
    # Facts: Anne is green or blue. 
    # Let's add a contradicting fact.
    name = facts.split()[0]
    contradicting_fact = f"{facts}. {name} is not cold."
    
    samples = []
    for q in row['questions'].split(' | '):
        if "cold" in q:
            reasoning = (f"Reasoning: Fact: {name} is not cold. Rules 1 & 2 state that if {name} is "
                         f"{rules_list[0].split()[3]} or {rules_list[1].split()[3]}, then {name} must be cold. "
                         f"The fact and the rules are in direct contradiction. In such cases, the status is False/Unknown. Answer: False")
            samples.append({
                "input_text": f"Facts: {contradicting_fact}\nRules: {row['rules']}\nQuestion: {q}\nThink step by step.",
                "target_text": reasoning
            })
    return samples

def main():
    if not os.path.exists(BASE_TRAIN_FILE):
        print(f"Error: {BASE_TRAIN_FILE} not found.")
        return

    all_samples = []
    with open(BASE_TRAIN_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} training groups...")
    
    for row in rows:
        # 1. Base CoT
        all_samples.extend(generate_base_reasoning(row))
        # 2. Variant 2 (Rule removal)
        all_samples.extend(generate_variant2_reasoning(row))
        # 3. Variant 3 (Contradiction)
        all_samples.extend(generate_variant3_reasoning(row))

    print(f"Generated {len(all_samples)} total RA-CoT samples.")
    
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"Successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
