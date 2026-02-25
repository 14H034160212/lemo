"""
Stage 1 Training: Train on incomplete rules data
Train models to reason with incomplete information (variant2/variant3 style data)
"""

import argparse
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Model configurations
MODEL_LIST = {
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen2-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def encode(sample, tokenizer):
    """
    Encode each row's multiple question-answer pairs into independent samples.
    Each question corresponds to a label (T=1, F=0)
    """
    facts = sample["facts"]
    rules = sample["rules"]
    questions = sample["questions"].split(" | ")
    answers = sample["answers"].split(" | ")

    # Create multiple training samples (one per question)
    expanded_samples = []
    for q, a in zip(questions, answers):
        text = facts + " " + rules + " " + q
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        expanded_samples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": 1 if a.strip() == "T" else 0,
        })

    return expanded_samples


def build_lora_config(model_name: str) -> LoraConfig:
    """
    Build LoRA configuration based on model architecture.
    - For decoder-only (Qwen / LLaMA/TinyLlama): q/k/v/o_proj
    - For encoder-only (BERT): query / value
    """
    lower_name = model_name.lower()
    if "llama" in lower_name or "qwen" in lower_name or "tinyllama" in lower_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["query", "value"]

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    return lora


def train_stage1(
    model_key: str = "bert",
    train_data_path: str = "data/stage1_train_combined.csv",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
):
    """
    Stage 1 Training: Train on incomplete rules data.

    Args:
        model_key: Model type to use
        train_data_path: Path to stage1 training data CSV
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    model_name = MODEL_LIST[model_key]
    if output_dir is None:
        output_dir = f"./trained_models/{model_key}_stage1"

    print(f"=" * 80)
    print(f"Stage 1 Training")
    print(f"=" * 80)
    print(f"▶ Base model: {model_name}")
    print(f"▶ Training data: {train_data_path}")
    print(f"▶ Output directory: {output_dir}")
    print(f"=" * 80)

    # ------------------ tokenizer ------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if missing (common for Qwen / LLaMA)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️ pad_token was missing, set to eos_token: {tokenizer.pad_token}")

    # ------------------ dataset ------------------
    print(f"\n▶ Loading training data from: {train_data_path}")
    dataset = load_dataset("csv", data_files=train_data_path)["train"]
    print(f"  Loaded {len(dataset)} rows")

    # Expand each row into multiple samples (one per question)
    expanded_data = []
    for sample in dataset:
        expanded_samples = encode(sample, tokenizer)
        expanded_data.extend(expanded_samples)

    print(f"  Expanded to {len(expanded_data)} training samples")

    # Convert back to HuggingFace Dataset
    from datasets import Dataset
    dataset = Dataset.from_list(expanded_data)

    # ------------------ model ------------------
    print(f"\n▶ Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # ------------------ LoRA ------------------
    lora_config = build_lora_config(model_name)
    model = get_peft_model(model, lora_config)
    print(f"▶ LoRA enabled: {lora_config}")
    print(f"  Trainable parameters: {model.print_trainable_parameters()}")

    # ------------------ training args ------------------
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=20,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # ------------------ trainer ------------------
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    # ------------------ train ------------------
    print("\n🚀 Starting Stage 1 training...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Stage 1 model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "qwen", "llama"],
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/stage1_train_combined.csv",
        help="Path to stage1 training data CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: trained_models/{model}_stage1)",
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
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    args = parser.parse_args()

    train_stage1(
        model_key=args.model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
