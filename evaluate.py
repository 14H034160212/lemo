# evaluate.py
import argparse
import os
import csv

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_LIST = {
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen2-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# All test splits, including multi-law variant4
DEFAULT_TEST_FILES = {
    "base": "test_base.csv",
    "variant1": "test_variant1.csv",
    "variant2": "test_variant2.csv",
    "variant3": "test_variant3.csv",
    "variant4_equiv_contrapositive": "test_variant4_equiv_contrapositive.csv",
    "variant4_equiv_double_negation": "test_variant4_equiv_double_negation.csv",
    "variant4_equiv_implication": "test_variant4_equiv_implication.csv",
    "variant4_equiv_demorgan": "test_variant4_equiv_demorgan.csv",
    "variant4_equiv_identity": "test_variant4_equiv_identity.csv",
    "variant4_equiv_commutativity": "test_variant4_equiv_commutativity.csv",
    "variant4_equiv_multi": "test_variant4_equiv_multi.csv",
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


def describe_change(split_name: str, laws_used: str, law_count: int) -> str:
    """
    Human-readable description of what changed for this split.
    For multi-law cases, we embed the law list and count.
    """
    if split_name == "base":
        return "none"
    if split_name == "variant1":
        return "removed redundant rule: 'If someone is young then they are cold.'"
    if split_name == "variant2":
        return "removed key rule: 'If someone is cold then they are rough.'"
    if split_name == "variant3":
        return "changed facts: added '<name> is not cold or not nice'"

    if split_name.startswith("variant4_equiv_"):
        if split_name == "variant4_equiv_multi":
            return f"multiple logical equivalence laws applied (count={law_count}): {laws_used}"
        base = split_name.replace("variant4_equiv_", "")
        return f"logical equivalence law applied: {base}"

    return "unknown"


def eval_and_save(model, tokenizer, filename, model_key, split_name, device, out_dir):
    """
    Evaluate on one CSV file AND save predictions into a CSV.
    Each row in the output corresponds to ONE question.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test file not found: {filename}")

    ds = load_dataset("csv", data_files=filename)["train"]

    total, correct = 0, 0
    output_rows = []

    os.makedirs(out_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, f"{model_key}_{split_name}_predictions.csv")

    for row in ds:
        facts = row["facts"]
        rules = row["rules"]
        questions = row["questions"].split(" | ")
        answers = row["answers"].split(" | ")
        laws_used = row.get("equiv_laws_used", "") or ""
        law_list = [x for x in laws_used.split(",") if x]
        law_count = len(law_list)

        changed_desc = describe_change(split_name, laws_used, law_count)

        for q, truth in zip(questions, answers):
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
                "equiv_laws_used": laws_used,
                "equiv_law_count": law_count,
                "changed_rule": changed_desc,
            })

            if pred == truth:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0

    # Save prediction CSV
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
                "equiv_laws_used",
                "equiv_law_count",
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions_dir = os.path.join(model_dir, "predictions")
    results = {}

    print("\n===== Detailed Evaluation Per Split =====")

    for split_name, filename in DEFAULT_TEST_FILES.items():
        acc, total, correct = eval_and_save(
            model,
            tokenizer,
            filename,
            model_key,
            split_name,
            device,
            predictions_dir,
        )
        results[split_name] = acc
        print(f"[{split_name}] {filename}")
        print(f"  samples (questions): {total}")
        print(f"  correct: {correct}")
        print(f"  accuracy: {acc:.4f}")
        print("-" * 40)

    # ------- summary table -------
    print("\n===== Base vs Variants Accuracy Table =====")
    base_acc = results.get("base", 0.0)

    header = f"{'Split':<35} | {'Accuracy':>9} | {'Δ vs base':>9}"
    print(header)
    print("-" * len(header))

    ordered_splits = [
        "base",
        "variant1",
        "variant2",
        "variant3",
        "variant4_equiv_contrapositive",
        "variant4_equiv_double_negation",
        "variant4_equiv_implication",
        "variant4_equiv_demorgan",
        "variant4_equiv_identity",
        "variant4_equiv_commutativity",
        "variant4_equiv_multi",
    ]

    for split in ordered_splits:
        if split not in results:
            continue
        acc = results[split]
        delta = acc - base_acc
        delta_str = f"{delta:+.3f}" if split != "base" else "0.000"
        print(f"{split:<35} | {acc:>9.4f} | {delta_str:>9}")

    print("\n✅ Evaluation FINISHED.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "qwen", "llama"])
    args = parser.parse_args()

    main(args.model)
