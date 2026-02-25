import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig

def text_formatting_func(example):
    # LogicNLI: premise, hypothesis, label (0=entailment, 1=contradiction, 2=neutral)
    output_texts = []
    
    label_map = {0: "True", 1: "False", 2: "False"} # Simplifying
    
    for premise, hypothesis, label in zip(example['premise'], example['hypothesis'], example['label']):
        if label == -1: continue # Skip invalid
        
        # Construct Conflict-Aware Trace
        if label == 1: # Contradiction
            reasoning = (f"Step 1: Check consistency. \n"
                         f"Analysis: The hypothesis '{hypothesis}' follows from the negation of the premise '{premise}' or contradicts it. \n"
                         f"Conflict detected. \n"
                         f"Step 2: Stop. Output False.")
            ans = "False"
        elif label == 0: # Entailment
            reasoning = (f"Step 1: Check consistency. \n"
                         f"Analysis: Premise and hypothesis are consistent. \n"
                         f"Step 2: Apply logic. The premise entails the hypothesis. Output True.")
            ans = "True"
        else: # Neutral
            reasoning = (f"Step 1: Check consistency. \n"
                         f"Analysis: Consistent but independent. \n"
                         f"Step 2: Apply logic. Cannot prove hypothesis from premise. Output False.")
            ans = "False"

        # Qwen Chat Format
        messages = [
            {"role": "system", "content": "You are a logical reasoner. Verify consistency first."},
            {"role": "user", "content": f"Facts: {premise}\nQuestion: {hypothesis}\nThink step by step."},
            {"role": "assistant", "content": f"{reasoning} Answer: {ans}"}
        ]
        
        # We need to format this as a single string for SFTTrainer usually? 
        # But Qwen tokenizer applies chat template. 
        # We will do simple string concat for simplicity in this demo script 
        # or use appropriate formatting if tokenizer available.
        
        # Simple format for generic SFT:
        text = f"<|im_start|>system\nYou are a logical reasoner. Verify consistency first.<|im_end|>\n<|im_start|>user\nFacts: {premise}\nQuestion: {hypothesis}\nThink step by step.<|im_end|>\n<|im_start|>assistant\n{reasoning} Answer: {ans}<|im_end|>\n"
        output_texts.append(text)
        
    return output_texts

def train():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "checkpoints/real_world_sft"
    
    print("Loading LogicNLI...")
    dataset = load_dataset("tasksource/LogicNLI", split="train[:2000]") # Subset for speed
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=100, # Quick demo training
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=text_formatting_func,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=512
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving...")
    trainer.save_model(output_dir)
    print("Done!")

if __name__ == "__main__":
    train()
