# train.py
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# 使用开放可直接访问的模型
MODEL_LIST = {
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen2-1.5B",
    # 用 TinyLlama 代替 Meta 的 gated Llama-3
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def encode(sample, tokenizer):
    """
    把一行样本转成模型输入：
    输入 = facts + rules + questions
    label = 如果 answers 里面包含 'T' 就设为 1，否则 0
    你可以按需要修改这个逻辑。
    """
    text = sample["facts"] + " " + sample["rules"] + " " + sample["questions"]
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    sample["input_ids"] = enc["input_ids"]
    sample["attention_mask"] = enc["attention_mask"]

    # 这里是一个简单示例：只要有 T 就算 True
    sample["labels"] = 1 if "T" in sample["answers"] else 0
    return sample


def build_lora_config(model_name: str) -> LoraConfig:
    """
    根据模型类型选择合适的 LoRA target_modules
    - 对于 decoder-only（Qwen / LLaMA/TinyLlama）使用 q/k/v/o_proj
    - 对于 BERT 这类 encoder-only，使用 query / value
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


def train(model_key: str = "bert"):
    model_name = MODEL_LIST[model_key]
    print(f"▶ Using base model: {model_name}")

    # ------------------ tokenizer ------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen / LLaMA / TinyLlama 有可能没有 pad_token，这里显式设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️ pad_token was missing, set pad_token = eos_token = {tokenizer.pad_token}")

    # ------------------ dataset ------------------
    dataset = load_dataset("csv", data_files="train.csv")["train"]

    dataset = dataset.map(
        lambda x: encode(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # ------------------ model ------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # 确保模型也知道 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # ------------------ LoRA ------------------
    lora_config = build_lora_config(model_name)
    model = get_peft_model(model, lora_config)
    print("▶ LoRA enabled with config:", lora_config)

    # ------------------ training args ------------------
    output_dir = f"./trained_models/{model_key}"
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=20,
        remove_unused_columns=False,  # 我们已经手动控制输入列
    )

    # ------------------ trainer ------------------
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    # ------------------ train ------------------
    print("🚀 Start training...")
    trainer.train()

    # ------------------ save ------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "qwen", "llama"],
        help="Which base model to fine-tune",
    )
    args = parser.parse_args()

    train(args.model)
