
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Environment setup
os.environ['HF_HOME'] = '.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '.cache/huggingface/datasets'
os.environ['TMPDIR'] = './tmp'

def train_fusion(train_file="data/train_fusion.csv",
                 output_dir="trained_models/qwen_fusion_sft_conflict_aware",
                 seed=42, model_id="Qwen/Qwen2-1.5B", batch_size=4,
                 grad_accum=4):
    """seed controls LoRA initialisation and data order.

    There was no seed argument before, so every run used the HuggingFace
    default of 42 and re-running produced bit-identical results. That made the
    single number we report indistinguishable from a lucky draw: with n=1 there
    is no way to tell a 14-point gap between two methods from seed noise.
    """
    print(f"  base model : {model_id}")
    print(f"  train file : {train_file}")
    print(f"  output dir : {output_dir}")
    
    print(f"Loading data from {train_file}...")
    df = pd.read_csv(train_file)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_func(examples):
        # Concatenate input and target for causal LM training
        full_texts = [f"{inp}\n{tgt}{tokenizer.eos_token}" for inp, tgt in zip(examples['input_text'], examples['target_text'])]
        model_inputs = tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)
        
        # Create labels (mask input part with -100)
        labels = []
        for i, text in enumerate(full_texts):
            input_ids = model_inputs['input_ids'][i]
            # Find the split point
            input_part = examples['input_text'][i]
            input_len = len(tokenizer.encode(input_part, add_special_tokens=True))
            
            label = [-100] * input_len + input_ids[input_len:]
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_ds = dataset.map(tokenize_func, batched=True, remove_columns=dataset.column_names)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        # The corpus is now 10x the size it was when num_train_epochs=3 was set
        # (112k rows vs 11.2k), so 3 epochs would be 21k steps against the ~2.1k
        # the recipe was tuned for. Cap the step count so the training budget
        # stays comparable while the model still sees the full, correctly
        # proportioned distribution (14.29% contradiction samples).
        num_train_epochs=3,
        max_steps=2100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting Fusion SFT training...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fusion SFT model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_fusion.csv")
    ap.add_argument("--output_dir",
                    default="trained_models/qwen_fusion_sft_conflict_aware")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", default="Qwen/Qwen2-1.5B",
                    help="backbone; the 8B run uses /data/shared/qwen3/Qwen3-8B")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4,
                    help="kept x batch_size constant across scales so the "
                         "effective batch, and therefore the step count, "
                         "matches the 1.5B runs")
    a = ap.parse_args()
    train_fusion(a.train_file, a.output_dir, a.seed, a.model_id,
                 a.batch_size, a.grad_accum)
