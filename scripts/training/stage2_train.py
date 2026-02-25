"""
Stage 2 Training: Continue training from Stage 1 or train with mixed data
Two modes:
1. Continue from Stage 1 model: Load stage1 checkpoint and continue training on original data
2. Mixed data training: Train on combined stage1 + original data (from scratch or from stage1)
"""

import argparse
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel

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


def load_and_prepare_data(
    tokenizer,
    original_data_path="data/train.csv",
    stage1_data_path="data/stage1_train_combined.csv",
    use_mixed=False,
):
    """
    Load and prepare training data.

    Args:
        tokenizer: Tokenizer instance
        original_data_path: Path to original training data
        stage1_data_path: Path to stage1 training data
        use_mixed: If True, combine both datasets

    Returns:
        Prepared HuggingFace Dataset
    """
    datasets_to_load = []

    # Always load original data for stage2
    if os.path.exists(original_data_path):
        print(f"▶ Loading original training data: {original_data_path}")
        original_ds = load_dataset("csv", data_files=original_data_path)["train"]
        datasets_to_load.append(original_ds)
        print(f"  Original data: {len(original_ds)} rows")
    else:
        raise FileNotFoundError(f"Original training data not found: {original_data_path}")

    # Optionally add stage1 data
    if use_mixed:
        if os.path.exists(stage1_data_path):
            print(f"▶ Loading stage1 training data: {stage1_data_path}")
            stage1_ds = load_dataset("csv", data_files=stage1_data_path)["train"]
            datasets_to_load.append(stage1_ds)
            print(f"  Stage1 data: {len(stage1_ds)} rows")
        else:
            print(f"⚠️  Stage1 data not found: {stage1_data_path}, using only original data")

    # Concatenate datasets if using mixed mode
    if len(datasets_to_load) > 1:
        combined_ds = concatenate_datasets(datasets_to_load)
        print(f"▶ Combined dataset: {len(combined_ds)} rows")
        dataset = combined_ds
    else:
        dataset = datasets_to_load[0]

    # Expand each row into multiple samples (one per question)
    expanded_data = []
    for sample in dataset:
        expanded_samples = encode(sample, tokenizer)
        expanded_data.extend(expanded_samples)

    print(f"▶ Expanded to {len(expanded_data)} training samples")

    return Dataset.from_list(expanded_data)


def train_stage2(
    model_key: str = "bert",
    from_stage1: bool = False,
    stage1_model_dir: str = None,
    use_mixed_data: bool = False,
    original_data_path: str = "data/train.csv",
    stage1_data_path: str = "data/stage1_train_combined.csv",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
):
    """
    Stage 2 Training: Continue from Stage 1 or train with mixed data.

    Args:
        model_key: Model type to use
        from_stage1: If True, load from stage1 checkpoint
        stage1_model_dir: Path to stage1 model (if from_stage1=True)
        use_mixed_data: If True, train on stage1 + original data
        original_data_path: Path to original training data
        stage1_data_path: Path to stage1 training data
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    model_name = MODEL_LIST[model_key]

    # Determine stage1 model directory
    if stage1_model_dir is None:
        stage1_model_dir = f"./trained_models/{model_key}_stage1"

    # Determine output directory
    if output_dir is None:
        if use_mixed_data:
            output_dir = f"./trained_models/{model_key}_stage2_mixed"
        else:
            output_dir = f"./trained_models/{model_key}_stage2"

    print(f"=" * 80)
    print(f"Stage 2 Training")
    print(f"=" * 80)
    print(f"▶ Base model: {model_name}")
    print(f"▶ From stage1 checkpoint: {from_stage1}")
    if from_stage1:
        print(f"  Stage1 model dir: {stage1_model_dir}")
    print(f"▶ Use mixed data: {use_mixed_data}")
    print(f"▶ Output directory: {output_dir}")
    print(f"=" * 80)

    # ------------------ tokenizer ------------------
    if from_stage1 and os.path.exists(stage1_model_dir):
        print(f"\n▶ Loading tokenizer from stage1: {stage1_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)
    else:
        print(f"\n▶ Loading tokenizer from base model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️ pad_token was missing, set to eos_token: {tokenizer.pad_token}")

    # ------------------ dataset ------------------
    print(f"\n▶ Preparing training data...")
    dataset = load_and_prepare_data(
        tokenizer,
        original_data_path=original_data_path,
        stage1_data_path=stage1_data_path,
        use_mixed=use_mixed_data,
    )

    # ------------------ model ------------------
    if from_stage1 and os.path.exists(stage1_model_dir):
        print(f"\n▶ Loading model from stage1 checkpoint: {stage1_model_dir}")
        # Load base model first
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        # Load PEFT adapter on top
        model = PeftModel.from_pretrained(base_model, stage1_model_dir, is_trainable=True)
        print(f"  ✅ Stage1 PEFT model loaded successfully")
    else:
        print(f"\n▶ Loading base model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )

        # Apply LoRA if starting fresh
        lora_config = build_lora_config(model_name)
        model = get_peft_model(model, lora_config)
        print(f"▶ LoRA enabled: {lora_config}")

    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # Print trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
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
    print("\n🚀 Starting Stage 2 training...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Stage 2 model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "qwen", "llama"],
        help="Base model type",
    )
    parser.add_argument(
        "--from_stage1",
        action="store_true",
        help="Load from stage1 checkpoint and continue training",
    )
    parser.add_argument(
        "--stage1_model_dir",
        type=str,
        default=None,
        help="Path to stage1 model directory (default: trained_models/{model}_stage1)",
    )
    parser.add_argument(
        "--mixed_data",
        action="store_true",
        help="Use mixed data (stage1 + original)",
    )
    parser.add_argument(
        "--original_data",
        type=str,
        default="data/train.csv",
        help="Path to original training data",
    )
    parser.add_argument(
        "--stage1_data",
        type=str,
        default="data/stage1_train_combined.csv",
        help="Path to stage1 training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: trained_models/{model}_stage2[_mixed])",
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

    train_stage2(
        model_key=args.model,
        from_stage1=args.from_stage1,
        stage1_model_dir=args.stage1_model_dir,
        use_mixed_data=args.mixed_data,
        original_data_path=args.original_data,
        stage1_data_path=args.stage1_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
