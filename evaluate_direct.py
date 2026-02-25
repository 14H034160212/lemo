
import argparse
import os
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'

def evaluate_direct(model_path, split="variant3"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    filepath = f"data/test_{split}.csv"
    print(f"Evaluating {split} (Direct)...")
    ds = load_dataset("csv", data_files=filepath)["train"]
    
    correct = 0
    total = 0
    
    for row in ds:
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(" | ")
        answers = row['answers'].split(" | ")
        
        for q, a in zip(questions, answers):
            # NO "Think step by step"
            prompt = f"Facts: {facts}\nRules: {rules}\nQuestion: {q}"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=10, 
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Simple parsing
            pred = "T" if "true" in generated else "F"
            
            is_correct = (pred == a.strip())
            if is_correct:
                correct += 1
            total += 1
            
    acc = correct / total
    print(f"Split: {split} (Direct), Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--split", default="variant3")
    args = parser.parse_args()
    evaluate_direct(args.model_dir, args.split)
