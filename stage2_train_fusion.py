
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
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TMPDIR'] = '/mnt/lemo/tmp'

def train_fusion():
    model_id = "Qwen/Qwen2-1.5B"
    train_file = "data/train_fusion.csv"
    output_dir = "trained_models/qwen_fusion_sft_conflict_aware"
    
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
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
    train_fusion()
