
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'

def verify_template(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    splits = {
        "variant3": "data/test_variant3.csv",
        "base": "data/test_base.csv"
    }
    
    print("\nVerifying Template Adherence (Expecting 'Step 1: Verify facts')...")
    
    for split, filepath in splits.items():
        print(f"\nScanning {split}...")
        ds = load_dataset("csv", data_files=filepath)["train"]
        
        # Check first 5 samples
        for i, row in enumerate(ds):
            if i >= 5: break
            
            facts = row['facts']
            rules = row['rules']
            q = row['questions'].split(" | ")[0]
            
            prompt = f"Facts: {facts}\nRules: {rules}\nQuestion: {q}\nThink step by step."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            has_preamble = "Step 1: Verify facts" in generated
            status = "✅ PASS" if has_preamble else "❌ FAIL"
            print(f"Sample {i+1}: {status} | Output Start: {generated[:50]}...")

if __name__ == "__main__":
    verify_template("trained_models/qwen_fusion_sft_conflict_aware")
