"""
Stage 1 Training for BERT: Multiple Choice Rule Selection
Train BERT to select the correct missing rule from candidates.

Task: Given context (facts + masked_rules + question) and rule candidates,
      select the correct missing rule.
"""

import argparse
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
import torch

# Model configuration
MODEL_NAME = "bert-base-uncased"


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator for multiple choice tasks.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def encode_sample(sample, tokenizer, max_length=256):
    """
    Encode a sample for multiple choice.

    For each candidate, we create: [CLS] context [SEP] candidate [SEP]
    """
    context = sample['context']
    candidates = [
        sample.get('candidate_0', ''),
        sample.get('candidate_1', ''),
        sample.get('candidate_2', ''),
        sample.get('candidate_3', ''),
    ]

    # Filter out empty candidates
    candidates = [c for c in candidates if c]

    # Encode each choice
    encoded_choices = []
    for candidate in candidates:
        # BERT input: [CLS] context [SEP] candidate [SEP]
        encoded = tokenizer(
            context,
            candidate,
            truncation=True,
            max_length=max_length,
            padding=False,  # Will pad in collator
        )
        encoded_choices.append(encoded)

    # Transpose to get lists of input_ids, attention_mask, etc.
    return {
        'input_ids': [choice['input_ids'] for choice in encoded_choices],
        'attention_mask': [choice['attention_mask'] for choice in encoded_choices],
        'labels': int(sample['correct_answer']),
    }


def build_lora_config() -> LoraConfig:
    """
    Build LoRA configuration for BERT.
    """
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",  # Sequence classification
    )
    return lora


def train_stage1_bert(
    train_data_path: str = "data/stage1_train_bert.csv",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 256,
):
    """
    Stage 1 Training for BERT: Multiple choice rule selection.

    Args:
        train_data_path: Path to stage1 training data CSV (BERT format)
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
    """
    if output_dir is None:
        output_dir = "./trained_models/bert_stage1_mc"

    print(f"=" * 80)
    print(f"Stage 1 BERT Training - Multiple Choice Rule Selection")
    print(f"=" * 80)
    print(f"▶ Base model: {MODEL_NAME}")
    print(f"▶ Training data: {train_data_path}")
    print(f"▶ Output directory: {output_dir}")
    print(f"=" * 80)

    # ------------------ tokenizer ------------------
    print(f"\n▶ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ------------------ dataset ------------------
    print(f"\n▶ Loading training data from: {train_data_path}")
    dataset = load_dataset("csv", data_files=train_data_path)["train"]
    print(f"  Loaded {len(dataset)} samples")

    # Encode dataset
    print(f"▶ Encoding dataset...")
    encoded_data = []
    for sample in dataset:
        try:
            encoded = encode_sample(sample, tokenizer, max_length=max_length)
            encoded_data.append(encoded)
        except Exception as e:
            print(f"Warning: Failed to encode sample: {e}")
            continue

    dataset = Dataset.from_list(encoded_data)
    print(f"  Encoded {len(dataset)} samples")

    # ------------------ model ------------------
    print(f"\n▶ Loading model: {MODEL_NAME}")
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    # ------------------ LoRA ------------------
    print(f"▶ Applying LoRA...")
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    print(f"  LoRA config: {lora_config}")
    model.print_trainable_parameters()

    # ------------------ data collator ------------------
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

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
    print("\n🚀 Starting Stage 1 BERT training (multiple choice)...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Stage 1 BERT model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 BERT Training (Multiple Choice)")
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/stage1_train_bert.csv",
        help="Path to stage1 training data CSV (BERT format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: trained_models/bert_stage1_mc)",
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
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    train_stage1_bert(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
