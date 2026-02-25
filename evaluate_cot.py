
import argparse
import os
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'

DEFAULT_TEST_FILES = {
    "base": "data/test_base.csv",
    "variant2": "data/test_variant2.csv",
    "variant3": "data/test_variant3.csv",
}

def format_cot_prompt(facts, rules, question):
    input_text = f"Facts: {facts}\nRules: {rules}\nQuestion: {question}\nThink step by step."
    return input_text

def parse_cot_answer(text):
    # Expects "Reasoning: ... Answer: True"
    text_lower = text.lower()
    if "answer: true" in text_lower:
        return "T"
    if "answer: false" in text_lower:
        return "F"
    # Fallback: look for generic true/false at end
    if text_lower.endswith("true"):
        return "T"
    if text_lower.endswith("false"):
        return "F"
    return "F" # Default to False if unclear (safe for Variant 2/3)

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
    
    for split, filepath in DEFAULT_TEST_FILES.items():
        print(f"Evaluating {split}...")
        ds = load_dataset("csv", data_files=filepath)["train"]
        
        correct = 0
        total = 0
        
        for row in ds:
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
                
                # Store prediction for CSV
                results_detailed.append({
                    "split": split,
                    "prompt": prompt.replace("\n", " "),
                    "generated": generated.replace("\n", " "),
                    "ground_truth": a.strip(),
                    "prediction": pred,
                    "correct": is_correct
                })
                
        acc = correct / total
        print(f"Split: {split}, Accuracy: {acc:.4f}")
        summary_results.append({"split": split, "accuracy": acc})
        
    # Save Detailed Predictions
    preds_file = os.path.join(model_path, "detailed_predictions.csv")
    with open(preds_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "prompt", "generated", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results_detailed)
    print(f"Detailed predictions saved to {preds_file}")

    # Save Summary
    with open(os.path.join(model_path, "accuracy_summary_cot.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_results)
        
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    evaluate_cot(args.model_dir)
