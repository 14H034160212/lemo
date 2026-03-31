"""
Stage 1 Training for Generative Models (Qwen/LLaMA)
Train models to generate missing rules given incomplete information.

Task: Input facts + masked_rules + question → Output: missing_rule
"""

import argparse
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/data/qbao775/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data/qbao775/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/data/qbao775/lemo/.cache/huggingface/transformers'

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Model configurations
MODEL_LIST = {
    "qwen": "Qwen/Qwen2-1.5B",
    "qwen3": "/data/shared/qwen3/Qwen3-8B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def format_prompt(input_text, target_text=None):
    """
    Format input and optionally target for training.

    Args:
        input_text: The prompt/input
        target_text: The expected output (for training)

    Returns:
        Full text for training
    """
    if target_text is not None:
        return f"{input_text} {target_text}"
    return input_text


def encode_sample(sample, tokenizer, max_length=512):
    """
    Encode a single sample for causal LM training.

    The key idea is:
    - Concatenate input_text + target_text
    - Mask the input part so loss is only computed on target
    """
    input_text = sample['input_text']
    target_text = sample['target_text']

    # Full text
    full_text = format_prompt(input_text, target_text)

    # Tokenize full text
    full_encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,  # Will pad in collator
    )

    # Tokenize input only (to find where to mask)
    input_encoding = tokenizer(
        format_prompt(input_text, ""),
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_length = len(input_encoding['input_ids'])

    # Create labels: -100 for input part, actual tokens for target part
    labels = full_encoding['input_ids'].copy()
    labels[:input_length] = [-100] * input_length  # Mask input, only compute loss on target

    return {
        'input_ids': full_encoding['input_ids'],
        'attention_mask': full_encoding['attention_mask'],
        'labels': labels,
    }


def build_lora_config(model_key: str = "qwen") -> LoraConfig:
    """
    Build LoRA configuration for decoder-only models.
    Qwen3 has q_norm/k_norm layers that conflict with LoRA on q_proj/k_proj,
    so we use v/o/gate/up/down projections instead.
    """
    if model_key == "qwen3":
        target_modules = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora


def train_stage1_generative(
    model_key: str = "qwen",
    train_data_path: str = "data/stage1_train_generative.csv",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """
    Stage 1 Training for generative models: Rule prediction.

    Args:
        model_key: Model type (qwen or llama)
        train_data_path: Path to stage1 training data CSV
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
    """
    if model_key not in MODEL_LIST:
        raise ValueError(f"Model {model_key} not supported. Choose from: {list(MODEL_LIST.keys())}")

    model_name = MODEL_LIST[model_key]
    if output_dir is None:
        output_dir = f"./trained_models/{model_key}_stage1_gen"

    print(f"=" * 80)
    print(f"Stage 1 Generative Training - Rule Prediction")
    print(f"=" * 80)
    print(f"▶ Base model: {model_name}")
    print(f"▶ Training data: {train_data_path}")
    print(f"▶ Output directory: {output_dir}")
    print(f"=" * 80)

    # ------------------ tokenizer ------------------
    print(f"\n▶ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️ pad_token was missing, set to eos_token: {tokenizer.pad_token}")

    # ------------------ dataset ------------------
    print(f"\n▶ Loading training data from: {train_data_path}")
    dataset = load_dataset("csv", data_files=train_data_path)["train"]
    print(f"  Loaded {len(dataset)} samples")

    # Encode dataset
    print(f"▶ Encoding dataset...")
    encoded_data = []
    for sample in dataset:
        encoded = encode_sample(sample, tokenizer, max_length=max_length)
        encoded_data.append(encoded)

    dataset = Dataset.from_list(encoded_data)
    print(f"  Encoded {len(dataset)} samples")

    # ------------------ model ------------------
    print(f"\n▶ Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # ------------------ LoRA ------------------
    print(f"▶ Applying LoRA...")
    lora_config = build_lora_config(model_key)
    model = get_peft_model(model, lora_config)
    print(f"  LoRA config: {lora_config}")
    model.print_trainable_parameters()

    # ------------------ data collator ------------------
    # Custom data collator that handles labels padding
    def custom_data_collator(features):
        # Find max length
        max_length = max(len(f['input_ids']) for f in features)

        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        for f in features:
            pad_len = max_length - len(f['input_ids'])
            # Pad input_ids with pad_token_id
            batch['input_ids'].append(f['input_ids'] + [tokenizer.pad_token_id] * pad_len)
            # Pad attention_mask with 0
            batch['attention_mask'].append(f['attention_mask'] + [0] * pad_len)
            # Pad labels with -100 (ignore index)
            batch['labels'].append(f['labels'] + [-100] * pad_len)

        return {
            'input_ids': torch.tensor(batch['input_ids']),
            'attention_mask': torch.tensor(batch['attention_mask']),
            'labels': torch.tensor(batch['labels']),
        }

    data_collator = custom_data_collator

    # ------------------ training args ------------------
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=20,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=2,
    )

    # ------------------ trainer ------------------
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # ------------------ train ------------------
    print("\n🚀 Starting Stage 1 generative training...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Stage 1 generative model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Generative Training")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "qwen3", "llama"],
        help="Generative model to fine-tune",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/stage1_train_generative.csv",
        help="Path to stage1 training data CSV (generative format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: trained_models/{model}_stage1_gen)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    train_stage1_generative(
        model_key=args.model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
