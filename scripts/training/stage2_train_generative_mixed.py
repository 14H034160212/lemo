"""
Stage 2 Training for Generative Models
Mixed training: Rule prediction + T/F prediction

This script:
1. Loads stage1 model (rule prediction)
2. Converts original T/F data to generative format
3. Trains on mixed data (stage1 + stage2 tasks)
"""

import argparse
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import pandas as pd

# Model configurations
MODEL_LIST = {
    "qwen": "Qwen/Qwen2-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def convert_tf_to_generative(sample):
    """
    Convert original T/F classification data to generative format.

    Input: facts + rules + questions + answers
    Output: For each Q-A pair, create input->output format
    """
    facts = sample['facts']
    rules = sample['rules']
    questions = sample['questions'].split(' | ')
    answers = sample['answers'].split(' | ')

    samples = []
    for q, a in zip(questions, answers):
        input_text = f"""Given the following information:

Facts: {facts}

Rules:
{chr(10).join(f"- {rule}" for rule in rules.split(' | '))}

Question: {q}

Based on the facts and rules above, is this statement true or false?

Answer:"""
        target_text = "True" if a.strip() == "T" else "False"

        samples.append({
            'input_text': input_text,
            'target_text': target_text,
        })

    return samples


def format_prompt(input_text, target_text=None):
    """Format input and optionally target for training."""
    if target_text is not None:
        return f"{input_text} {target_text}"
    return input_text


def encode_sample(sample, tokenizer, max_length=512):
    """
    Encode a single sample for causal LM training.
    Mask the input part so loss is only computed on target.
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
        padding=False,
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
    labels[:input_length] = [-100] * input_length

    return {
        'input_ids': full_encoding['input_ids'],
        'attention_mask': full_encoding['attention_mask'],
        'labels': labels,
    }


def build_lora_config() -> LoraConfig:
    """Build LoRA configuration for decoder-only models."""
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora


def train_stage2_generative(
    model_key: str = "qwen",
    stage1_model_dir: str = None,
    from_stage1: bool = False,
    use_mixed_data: bool = True,
    original_data_path: str = "data/train.csv",
    stage1_data_path: str = "data/stage1_train_generative.csv",
    output_dir: str = None,
    epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_length: int = 512,
):
    """
    Stage 2 Training for generative models: Mixed training.

    Args:
        model_key: Model type (qwen or llama)
        stage1_model_dir: Path to stage1 model
        from_stage1: If True, load from stage1 checkpoint
        use_mixed_data: If True, train on stage1 + original data
        original_data_path: Path to original training data
        stage1_data_path: Path to stage1 training data
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
    """
    if model_key not in MODEL_LIST:
        raise ValueError(f"Model {model_key} not supported. Choose from: {list(MODEL_LIST.keys())}")

    model_name = MODEL_LIST[model_key]

    # Determine stage1 model directory
    if stage1_model_dir is None:
        stage1_model_dir = f"./trained_models/{model_key}_stage1_gen"

    # Determine output directory
    if output_dir is None:
        if use_mixed_data:
            output_dir = f"./trained_models/{model_key}_stage2_mixed"
        else:
            output_dir = f"./trained_models/{model_key}_stage2"

    print(f"=" * 80)
    print(f"Stage 2 Generative Training - Mixed Tasks")
    print(f"=" * 80)
    print(f"▶ Base model: {model_name}")
    print(f"▶ From stage1: {from_stage1}")
    if from_stage1:
        print(f"  Stage1 model: {stage1_model_dir}")
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------ dataset ------------------
    print(f"\n▶ Preparing training data...")
    all_samples = []

    # Load original T/F data and convert to generative format
    print(f"  Loading original data: {original_data_path}")
    if os.path.exists(original_data_path):
        df_original = pd.read_csv(original_data_path)
        print(f"  Original data: {len(df_original)} rows")

        # Convert to generative format
        for _, row in df_original.iterrows():
            samples = convert_tf_to_generative(row)
            all_samples.extend(samples)

        print(f"  Converted to {len(all_samples)} T/F prediction samples")

    # Optionally add stage1 data (rule prediction)
    if use_mixed_data and os.path.exists(stage1_data_path):
        print(f"  Loading stage1 data: {stage1_data_path}")
        df_stage1 = pd.read_csv(stage1_data_path)
        print(f"  Stage1 data: {len(df_stage1)} rows")

        # Add to samples
        for _, row in df_stage1.iterrows():
            all_samples.append({
                'input_text': row['input_text'],
                'target_text': row['target_text'],
            })

        print(f"  Total samples (T/F + rule prediction): {len(all_samples)}")

    # Shuffle
    import random
    random.shuffle(all_samples)

    # Encode dataset
    print(f"▶ Encoding {len(all_samples)} samples...")
    encoded_data = []
    for sample in all_samples:
        encoded = encode_sample(sample, tokenizer, max_length=max_length)
        encoded_data.append(encoded)

    dataset = Dataset.from_list(encoded_data)
    print(f"  Encoded {len(dataset)} samples")

    # ------------------ model ------------------
    if from_stage1 and os.path.exists(stage1_model_dir):
        print(f"\n▶ Loading model from stage1: {stage1_model_dir}")
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        # Load PEFT adapter on top
        model = PeftModel.from_pretrained(base_model, stage1_model_dir, is_trainable=True)
        print(f"  ✅ Stage1 PEFT model loaded")
    else:
        print(f"\n▶ Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Apply LoRA if starting fresh
        lora_config = build_lora_config()
        model = get_peft_model(model, lora_config)
        print(f"  LoRA applied: {lora_config}")

    model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, 'print_trainable_parameters'):
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
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
    )

    # ------------------ trainer ------------------
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # ------------------ train ------------------
    print("\n🚀 Starting Stage 2 generative training...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Stage 2 generative model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Generative Training")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "llama"],
        help="Model type",
    )
    parser.add_argument(
        "--from_stage1",
        action="store_true",
        help="Load from stage1 checkpoint",
    )
    parser.add_argument(
        "--stage1_model_dir",
        type=str,
        default=None,
        help="Path to stage1 model (default: trained_models/{model}_stage1_gen)",
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
        default="data/stage1_train_generative.csv",
        help="Path to stage1 training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )

    args = parser.parse_args()

    train_stage2_generative(
        model_key=args.model,
        stage1_model_dir=args.stage1_model_dir,
        from_stage1=args.from_stage1,
        use_mixed_data=args.mixed_data,
        original_data_path=args.original_data,
        stage1_data_path=args.stage1_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
