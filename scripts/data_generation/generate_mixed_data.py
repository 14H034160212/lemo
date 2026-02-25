
import csv
import random
import uuid

INPUT_FILE = "data/train.csv"
OUTPUT_FILE = "data/train_mixed.csv"

def generate_mixed_data():
    print(f"Reading from {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        
    base_rows = [row for row in all_rows if row["type"] == "base_positive"]
    print(f"Found {len(base_rows)} base_positive rows.")
    
    new_rows = []
    
    # 1. Keep original rows (including base_negative, etc)
    # Actually, let's keep ALL original rows first
    new_rows.extend(all_rows)
    
    # 2. Generate variants from base_rows
    # We want to augment the dataset. Let's add 100% of base rows as variants 
    # (or maybe 50% each to avoid exploding dataset size? Let's do 100% for now, dataset is small)
    
    for row in base_rows:
        # --- Variant 2: Essential Rule Deletion ---
        # Rule to remove: "If someone is cold then they are rough."
        rules = row["rules"].split(" | ")
        questions = row["questions"].split(" | ")
        
        # Find the critical rule index
        critical_idx = -1
        for i, r in enumerate(rules):
            if "cold" in r and "rough" in r and "not" not in r: # Simple heuristic
                critical_idx = i
                break
        
        if critical_idx != -1:
            # Create Variant 2 sample
            # Remove rule
            new_rules = [r for i, r in enumerate(rules) if i != critical_idx]
            
            # Ground Truth: T | F | F | F
            # Note: The original questions are Q1..Q4. We assume Q1=Cold, Q2=Rough...
            # This is hardcoded to the specific problem structure.
            # Q1 (Cold) -> True (still valid)
            # Q2 (Rough) -> False (link broken)
            # Q3 (Young) -> False
            # Q4 (Nice) -> False
            
            v2_row = row.copy()
            v2_row["group_id"] = str(uuid.uuid4())
            v2_row["type"] = "aug_variant2"
            v2_row["rules"] = " | ".join(new_rules)
            v2_row["answers"] = "T | F | F | F"
            
            new_rows.append(v2_row)
            
        # --- Variant 3: Contradiction ---
        # Add fact: "Name is not cold or not nice"
        # Find name
        facts = row["facts"]
        name = facts.split()[0] # "Anne is..."
        contradiction = f"{name} is not cold or not nice"
        
        v3_row = row.copy()
        v3_row["group_id"] = str(uuid.uuid4())
        v3_row["type"] = "aug_variant3"
        v3_row["facts"] = f"{facts} | {contradiction}"
        v3_row["answers"] = "F | F | F | F"
        
        new_rows.append(v3_row)
        
    print(f"Generated {len(new_rows) - len(all_rows)} new augmented samples.")
    print(f"Total samples: {len(new_rows)}")
    
    # Shuffle to mix
    random.shuffle(new_rows)
    
    # Write output
    fieldnames = reader.fieldnames
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_mixed_data()
