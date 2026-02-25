
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel

# Environment
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'

def get_log_probs(logits, labels):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
    mask = (labels != -100).float().view(logits.size(0), -1)
    log_probs = -(loss * mask).sum(dim=1)
    return log_probs

class CustomDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        policy_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        ).logits
        
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).logits
            
        policy_log_probs = get_log_probs(policy_logits, inputs["labels"])
        ref_log_probs = get_log_probs(ref_logits, inputs["labels"])
        
        logits_diff = policy_log_probs - ref_log_probs
        loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
        return (loss, policy_logits) if return_outputs else loss

def train_fusion_dpo():
    base_model_id = "Qwen/Qwen2-1.5B"
    sft_model_path = "trained_models/qwen_fusion_sft_opt"
    dpo_data_path = "data/train_fusion_dpo.jsonl"
    output_dir = "trained_models/qwen_fusion_final_opt"
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    def tokenize_func(examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # DPO typically needs chosen and rejected pairs
            # This simplified version uses a custom loss, let's format for standard DPO if possible
            pass
        return examples

    print("Loading model for DPO...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    ref_model = PeftModel.from_pretrained(ref_model, sft_model_path)
    ref_model.eval()

    from trl import DPOConfig

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        beta=0.1,
        max_length=512,
        max_prompt_length=256
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    print("Starting Fusion DPO training...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fusion Final model saved to {output_dir}")

if __name__ == "__main__":
    train_fusion_dpo()
