# evaluate.py
import argparse
import os
import csv

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Must match train.py
MODEL_LIST = {
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen2-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Default test files
DEFAULT_TEST_FILES = {
    "base": "test_base.csv",
    "variant1": "test_variant1.csv",
    "variant2": "test_variant2.csv",
    "variant3": "test_variant3.csv",
    "variant4": "test_variant4.csv",
}


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_single(model, tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    return "T" if pred == 1 else "F"


def describe_change(split_name: str) -> str:
    """
    Return a human-readable description of what was changed
    for this variant (or 'none' for base).
    """
    if split_name == "base":
        return "none"
    elif split_name == "variant1":
        return "removed redundant rule: 'If someone is young then they are cold.'"
    elif split_name == "variant2":
        return "removed key rule: 'If someone is cold then they are rough.'"
    elif split_name == "variant3":
        return "changed facts: added '<name> is not cold or not nice'"
    elif split_name == "variant4":
        return "replaced first rule with its contrapositive"
    else:
        return "unknown"


def eval_and_save(model, tokenizer, filename, model_key, split_name, device):
    """
    Evaluate on one CSV file AND save predictions into a CSV.
    Each row in the output corresponds to ONE question, with
    context (facts+rules), GT, prediction, and which rule was changed.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test file not found: {filename}")

    ds = load_dataset("csv", data_files=filename)["train"]

    total, correct = 0, 0
    output_rows = []

    output_csv = f"{model_key}_{split_name}_predictions.csv"
    changed_desc = describe_change(split_name)

    for row in ds:
        facts = row["facts"]
        rules = row["rules"]
        questions = row["questions"].split(" | ")
        answers = row["answers"].split(" | ")

        for q, truth in zip(questions, answers):
            # Actual input given to the model for this question
            text = facts + " " + rules + " " + q
            pred = predict_single(model, tokenizer, text, device)

            output_rows.append({
                "group_id": row["group_id"],
                "type": split_name,
                "facts": facts,
                "rules": rules,
                "question": q,
                "ground_truth": truth,
                "prediction": pred,
                "changed_rule": changed_desc,
            })

            if pred == truth:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0

    # Save prediction CSV with context + change description
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "type",
                "facts",
                "rules",
                "question",
                "ground_truth",
                "prediction",
                "changed_rule",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"📄 Predictions saved to: {output_csv}")

    return acc, total, correct


def main(model_key: str):
    model_dir = f"./trained_models/{model_key}"
    base_model_name = MODEL_LIST[model_key]

    print(f"▶ Loading model from: {model_dir}")
    print(f"▶ Base model: {base_model_name}")

    device = build_device()
    print(f"▶ Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Ensure pad token exists (same logic as in train.py)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = {}

    print("\n===== Detailed Evaluation Per Split =====")

    for split_name, filename in DEFAULT_TEST_FILES.items():
        acc, total, correct = eval_and_save(
            model, tokenizer, filename, model_key, split_name, device
        )
        results[split_name] = acc
        print(f"[{split_name}] {filename}")
        print(f"  samples (questions): {total}")
        print(f"  correct: {correct}")
        print(f"  accuracy: {acc:.4f}")
        print("-" * 40)

    # ------- Base vs Variants summary table -------
    print("\n===== Base vs Variants Accuracy Table =====")
    base_acc = results["base"]

    header = f"{'Split':<10} | {'Accuracy':>9} | {'Δ vs base':>9}"
    print(header)
    print("-" * len(header))

    for split in ["base", "variant1", "variant2", "variant3", "variant4"]:
        acc = results[split]
        delta = acc - base_acc
        delta_str = f"{delta:+.3f}" if split != "base" else "0.000"
        print(f"{split:<10} | {acc:>9.4f} | {delta_str:>9}")

    print("\n✅ Evaluation FINISHED.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "qwen", "llama"])
    args = parser.parse_args()

    main(args.model)
