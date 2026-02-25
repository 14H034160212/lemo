
import csv
import json
import os

INPUT_FILE = "data/train_ra_cot.csv"
OUTPUT_FILE = "data/train_ra_cot_dpo.jsonl"

def generate_rejected_reasoning(target_text):
    """
    Creates a 'Rejected' version of the reasoning by introducing 
    common failure modes (skipping steps, wrong conclusions).
    """
    if "Answer: True" in target_text:
        # Turn it into a hallucinated False or shortcut True
        hallucination = target_text.replace("Answer: True", "Answer: False")
        hallucination = hallucination.replace("Rough implies young.", "Rough implies old.")
        return hallucination
    else:
        # Turn it into a hallucinated True
        hallucination = target_text.replace("Answer: False", "Answer: True")
        hallucination = "Reasoning: Fact matches rule. Answer: True" # Shortcut hallucination
        return hallucination

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    dpo_pairs = []
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['input_text']
            chosen = row['target_text']
            rejected = generate_rejected_reasoning(chosen)
            
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    print(f"Generated {len(dpo_pairs)} DPO pairs for RA-CoT.")
    
    with open(OUTPUT_FILE, "w") as f:
        for entry in dpo_pairs:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
