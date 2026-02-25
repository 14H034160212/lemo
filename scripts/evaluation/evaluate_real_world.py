import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import os

# Models to evaluate: (Checkpoint Path, Base Model Path)
MODELS = {
    "Fusion_Conflict": ("./trained_models/qwen_stage2_dpo", "Qwen/Qwen2-1.5B"),
    "RealWorld_SFT": ("checkpoints/real_world_sft", "Qwen/Qwen2.5-0.5B-Instruct")
}

def evaluate_model(model_path, base_model_name, label_name, data_path, dataset_name):
    print(f"Evaluating {label_name} using base {base_model_name} on {dataset_name}...")
    
    # Load Model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print("Loaded adapter.")
        except Exception as e:
            print(f"Failed to load adapter: {e}")
            model = base_model # Fallback
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    df = pd.read_csv(data_path)
    correct = 0
    total = 0
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row['input_text']
        target = str(row['target_text']).strip()
        
        # Conflict-Aware Prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Verify facts before answering."},
            {"role": "user", "content": input_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split(text)[-1].strip()
        
        pred = "Unknown"
        if "Answer: True" in generated: pred = "True"
        if "Answer: False" in generated: pred = "False"
        if "Conflict Found" in generated or "Stop." in generated: pred = "False"
        
        is_correct = (pred.lower() == target.lower())
        correct += int(is_correct)
        total += 1
        
        results.append({
            "input": input_text,
            "target": target,
            "generated": generated,
            "prediction": pred,
            "correct": is_correct
        })
        
    acc = correct / total
    print(f"{label_name} on {dataset_name} Accuracy: {acc:.4f}")
    
    # Save results - append mode or unique name
    out_file = f"data/real_world/results_{label_name}_{dataset_name}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    del model
    del base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if not os.path.exists("data/real_world/logicnli_eval.csv"): exit()

    # Eval Fusion-Conflict on LogicNLI
    # path, base = MODELS["Fusion_Conflict"]
    # evaluate_model(path, base, "Fusion-Conflict", "data/real_world/logicnli_eval.csv", "LogicNLI")
    
    # Eval RealWorld-SFT on LogicNLI
    path, base = MODELS["RealWorld_SFT"]
    evaluate_model(path, base, "RealWorld-SFT", "data/real_world/logicnli_eval.csv", "LogicNLI")
