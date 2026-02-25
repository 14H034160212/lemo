
import csv
import json

INPUT_FILE = "data/train_mixed.csv"
OUTPUT_FILE = "data/train_dpo.jsonl"

def format_prompt(facts, rules, question):
    input_text = f"""Given the following information:

Facts: {facts}

Rules:
{chr(10).join(f"- {rule}" for rule in rules.split(' | '))}

Question: {question}

Based on the facts and rules above, is this statement true or false?

Answer:"""
    return input_text

def generate_dpo_pairs():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    dpo_samples = []
    
    for row in rows:
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(' | ')
        answers = row['answers'].split(' | ') # "T | F | ..."
        
        for q, a in zip(questions, answers):
            prompt = format_prompt(facts, rules, q)
            
            chosen = ""
            rejected = ""
            
            if a.strip() == "T":
                chosen = "True"
                rejected = "False"
            else:
                chosen = "False"
                rejected = "True"
                
            dpo_samples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
            
    print(f"Generated {len(dpo_samples)} DPO pairs.")
    
    with open(OUTPUT_FILE, "w") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dpo_pairs()
