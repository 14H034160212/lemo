
import os
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'

def debug_v3(model_path):
    print(f"Loading CoT model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    ds = load_dataset("csv", data_files="data/test_variant3.csv")["train"]
    
    print("Debugging Variant 3 Predictions:")
    for i, row in enumerate(ds):
        if i >= 2: break # Check 2 groups (8 questions)
        
        facts = row['facts']
        rules = row['rules']
        questions = row['questions'].split(" | ")
        answers = row['answers'].split(" | ")
        
        print(f"\nGroup {i+1}:")
        print(f"Facts: {facts}")
        
        for q, a in zip(questions, answers):
            prompt = f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Q: {q}")
            print(f"GT: {a.strip()}")
            print(f"Pred: {generated}\n")

if __name__ == "__main__":
    debug_v3("trained_models/qwen_fusion_final_opt")
