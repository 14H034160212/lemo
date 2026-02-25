
import argparse
import os
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'

DEFAULT_TEST_FILES = {
    "variant3": "data/test_variant3.csv",
    "variant2": "data/test_variant2.csv",
    "base": "data/test_base.csv",
}

def format_cot_prompt(facts, rules, question):
    input_text = f"Facts: {facts}\nRules: {rules}\nQuestion: {question}\nThink step by step."
    return input_text

def parse_cot_answer(text):
    text_lower = text.lower()
    if "answer: true" in text_lower:
        return "T"
    if "answer: false" in text_lower:
        return "F"
    if text_lower.endswith("true"):
        return "T"
    if text_lower.endswith("false"):
        return "F"
    return "F"

def evaluate_cot(model_path):
    print(f"Loading CoT model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    results_detailed = []
    summary_results = []
    
    MAX_SAMPLES = 50 # Speed optimization
    
    for split, filepath in DEFAULT_TEST_FILES.items():
        print(f"Evaluating {split} (First {MAX_SAMPLES} samples)...")
        ds = load_dataset("csv", data_files=filepath)["train"]
        
        correct = 0
        total = 0
        
        # Iterate 
        for i, row in enumerate(ds):
            if i >= MAX_SAMPLES:
                break
                
            facts = row['facts']
            rules = row['rules']
            questions = row['questions'].split(" | ")
            answers = row['answers'].split(" | ")
            
            for q, a in zip(questions, answers):
                prompt = format_cot_prompt(facts, rules, q)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=128, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = parse_cot_answer(generated)
                
                is_correct = (pred == a.strip())
                if is_correct:
                    correct += 1
                total += 1
                
                results_detailed.append({
                    "split": split,
                    "prompt": prompt.replace("\n", " "),
                    "generated": generated.replace("\n", " "),
                    "ground_truth": a.strip(),
                    "prediction": pred,
                    "correct": is_correct
                })
            
            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/{MAX_SAMPLES} groups. Current Acc: {correct/total:.4f}")

        acc = correct / total
        print(f"Split: {split}, Accuracy: {acc:.4f}")
        summary_results.append({"split": split, "accuracy": acc})
        
    preds_file = os.path.join(model_path, "detailed_predictions_fast.csv")
    with open(preds_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "prompt", "generated", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results_detailed)
    print(f"Detailed predictions saved to {preds_file}")

    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    evaluate_cot(args.model_dir)
