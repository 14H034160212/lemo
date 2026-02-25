
import os
import torch
import torch.nn.functional as F
import argparse
import json
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
)
from peft import PeftModel, get_peft_model, LoraConfig

os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'

def get_log_probs(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.view(logits.size(0), -1)
    mask = (shift_labels != -100).float().view(logits.size(0), -1)
    log_probs = -(loss * mask).sum(dim=1)
    return log_probs

class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.ref_model.eval()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            
        log_probs = get_log_probs(logits, labels)
        ref_log_probs = get_log_probs(ref_logits, labels)
        batch_size = input_ids.shape[0] // 2
        
        chosen_log_probs = log_probs[:batch_size]
        rejected_log_probs = log_probs[batch_size:]
        chosen_ref_log_probs = ref_log_probs[:batch_size]
        rejected_ref_log_probs = ref_log_probs[batch_size:]
        
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        return (losses.mean(), outputs) if return_outputs else losses.mean()

def encode_dpo_pair(sample, tokenizer, max_length=512):
    prompt = sample['prompt']
    def tokenize(resp):
        full = f"{prompt} {resp}"
        enc = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        labels = input_ids[:]
        p_enc = tokenizer(prompt, truncation=True, max_length=max_length)
        p_len = len(p_enc['input_ids'])
        for i in range(min(p_len, len(labels))): labels[i] = -100
        for i in range(len(labels)):
            if input_ids[i] == tokenizer.pad_token_id: labels[i] = -100
        return input_ids, attention_mask, labels

    c_ids, c_mask, c_labels = tokenize(sample['chosen'])
    r_ids, r_mask, r_labels = tokenize(sample['rejected'])
    return {'c_ids': c_ids, 'c_mask': c_mask, 'c_labels': c_labels, 
            'r_ids': r_ids, 'r_mask': r_mask, 'r_labels': r_labels}

def collate_dpo(batch):
    ids = [i['c_ids'] for i in batch] + [i['r_ids'] for i in batch]
    mask = [i['c_mask'] for i in batch] + [i['r_mask'] for i in batch]
    labels = [i['c_labels'] for i in batch] + [i['r_labels'] for i in batch]
    return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask), 'labels': torch.tensor(labels)}

def train_ra_cot_dpo():
    model_name = "Qwen/Qwen2-1.5B"
    sft_model_dir = "./trained_models/qwen_stage2_ra_cot_sft"
    output_dir = "./trained_models/qwen_ra_cot_final"
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset("json", data_files="data/train_ra_cot_dpo.jsonl")['train']
    ds = ds.map(lambda x: encode_dpo_pair(x, tokenizer), batched=False)
    
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    policy = PeftModel.from_pretrained(base, sft_model_dir, is_trainable=True)
    
    ref_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    ref = PeftModel.from_pretrained(ref_base, sft_model_dir)
    ref.eval()
    
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=1e-6, fp16=True, logging_steps=10, remove_unused_columns=False
    )
    
    trainer = DPOTrainer(ref_model=ref, model=policy, args=args, train_dataset=ds, data_collator=collate_dpo, tokenizer=tokenizer)
    trainer.train()
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_ra_cot_dpo()
