import pandas as pd
import json
import os

def convert_to_evals_jsonl(csv_path, output_path):
    print(f"Converting {csv_path} to {output_path}...")
    df = pd.read_csv(csv_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Combine Facts and Rules into the input
            facts = row['facts'].replace(' | ', '\n')
            rules = row['rules'].replace(' | ', '\n')
            
            # OpenAI Evals usually evaluate single questions. 
            # The CSV has multiple questions per row. We will split them into separate eval items.
            questions = row['questions'].split(' | ')
            answers = row['answers'].split(' | ')
            
            for q, a in zip(questions, answers):
                # Map 'T' to 'True' and 'F' to 'False' if needed, or keep as is.
                # Standard OpenAI 'match' eval expects the ideal string.
                ideal = "True" if a.strip() == 'T' else "False"
                
                # Format: {"input": [{"role": "user", "content": "..."}], "ideal": "..."}
                # For basic evals, it's often simpler: {"input": "...", "ideal": "..."}
                # But the chat format is more common now.
                
                input_text = f"Facts:\n{facts}\n\nRules:\n{rules}\n\nQuestion: {q.strip()}\nAnswer with 'True' or 'False'.\nReasoning:"
                
                json_record = {
                    "input": [{"role": "user", "content": input_text}],
                    "ideal": ideal
                }
                f.write(json.dumps(json_record) + '\n')

if __name__ == "__main__":
    data_dir = "/mnt/lemo/data"
    output_dir = "/mnt/lemo/evals_data"
    os.makedirs(output_dir, exist_ok=True)
    
    variants = [
        ("test_variant1.csv", "logic_stress_v1.jsonl"),
        ("test_variant2.csv", "logic_stress_v2.jsonl"),
        ("test_variant3.csv", "logic_stress_v3.jsonl"),
        ("test_variant4_equiv_multi.csv", "logic_stress_v4.jsonl")
    ]
    
    for csv, jsonl in variants:
        conv_path = os.path.join(data_dir, csv)
        if os.path.exists(conv_path):
            convert_to_evals_jsonl(conv_path, os.path.join(output_dir, jsonl))
        else:
            print(f"Warning: {conv_path} not found.")
