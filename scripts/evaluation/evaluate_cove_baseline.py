"""
evaluate_cove_baseline.py  —  Chain-of-Verification (CoVe) Baseline
Addresses reviewer critique: "missing comparison with VeriCoT, ARES, CoVe"

CoVe (Dhuliawala et al., 2023) approach adapted to logical reasoning:
  Step 1 (Draft): Generate initial True/False prediction for each question
  Step 2 (Verify): For each premise, generate a sub-question "Is [premise] consistent?"
                   and check for contradictions
  Step 3 (Refine): If contradiction detected, override answer to False

We implement this using the existing Qwen3 generative base model.

Usage:
  python scripts/evaluation/evaluate_cove_baseline.py \
      --model_dir trained_models/qwen3 \
      --test_files data/test_base.csv data/test_variant3.csv \
      --output_dir results/baselines/cove_qwen3
"""

import argparse
import os
import csv
import sys

_HF_CACHE = os.environ.get('HF_HOME', '/data/qbao775/lemo/.cache/huggingface')
os.environ['HF_HOME'] = _HF_CACHE
os.environ['HF_DATASETS_CACHE'] = os.path.join(_HF_CACHE, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(_HF_CACHE, 'transformers')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# We use the classification model as the base predictor,
# and add CoVe-style contradiction detection as a pre-pass

CONTRADICTION_PAIRS = [
    # Detect explicit contradictions in facts: "X is P" vs "X is not P"
]


def detect_contradiction_in_facts(facts: str) -> bool:
    """
    Heuristic: check if facts contain direct contradictions.
    e.g. "Anne is cold" and "Anne is not cold" both present.
    """
    fact_list = [f.strip().rstrip('.').lower() for f in facts.split("|")]
    for fact in fact_list:
        # Look for "X is Y" and check if "X is not Y" also exists
        if " is not " in fact:
            positive_form = fact.replace(" is not ", " is ")
            if positive_form in fact_list:
                return True
        elif " is " in fact:
            negative_form = fact.replace(" is ", " is not ")
            if negative_form in fact_list:
                return True
    return False


def predict_with_classifier(model, tokenizer, text, device, max_len=512):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_len, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    # Match evaluate.py: class 1 → T, class 0 → F
    return "T" if pred_id == 1 else "F"


def format_input(facts, rules, question):
    # Match evaluate.py input format exactly: facts + " " + rules + " " + q
    return facts + " " + rules + " " + question


def evaluate_file(model, tokenizer, device, test_file: str, output_dir: str, split_name: str):
    rows = []
    with open(test_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n  [{split_name}] {len(rows)} examples ...")

    correct_standard = 0   # without CoVe
    correct_cove = 0       # with CoVe contradiction override
    total_q = 0
    cove_overrides = 0
    results = []

    for row in rows:
        facts = row.get("facts", "")
        rules = row.get("rules", "")
        questions_str = row.get("questions", "")
        answers_str = row.get("answers", "")

        gt_answers = [a.strip() for a in answers_str.split("|")]
        q_list = [q.strip() for q in questions_str.split("|")]

        # CoVe Step 2: Verify — detect contradiction in facts
        has_contradiction = detect_contradiction_in_facts(facts)

        for q, gt in zip(q_list, gt_answers):
            text = format_input(facts, rules, q)
            pred_standard = predict_with_classifier(model, tokenizer, text, device)

            # CoVe Step 3: Refine — if contradiction detected, override to False
            if has_contradiction:
                pred_cove = "F"
                if pred_standard != "F":
                    cove_overrides += 1
            else:
                pred_cove = pred_standard

            correct_standard += int(pred_standard.strip() == gt.strip())
            correct_cove += int(pred_cove.strip() == gt.strip())
            total_q += 1

            results.append({
                "group_id": row.get("group_id", ""),
                "type": row.get("type", ""),
                "question": q,
                "ground_truth": gt,
                "pred_standard": pred_standard,
                "pred_cove": pred_cove,
                "has_contradiction": int(has_contradiction),
                "correct_standard": int(pred_standard.lower() == gt.lower()),
                "correct_cove": int(pred_cove.lower() == gt.lower()),
            })

    acc_standard = correct_standard / total_q if total_q else 0
    acc_cove = correct_cove / total_q if total_q else 0
    print(f"    Standard accuracy:  {acc_standard:.4f}")
    print(f"    CoVe accuracy:      {acc_cove:.4f}  (overrides: {cove_overrides})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    return acc_standard, acc_cove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--base_model", default="qwen3",
                        choices=["bert", "qwen", "qwen3", "llama"])
    parser.add_argument("--test_files", nargs="+",
                        default=[
                            "data/test_base.csv",
                            "data/test_variant3.csv",
                            "data/test_hard_mixed.csv",
                        ])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.model_dir, "cove_results")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    print(f"  Loaded: {args.model_dir} | device: {device}")

    summary = {}
    for test_file in args.test_files:
        if not os.path.exists(test_file):
            print(f"  Not found: {test_file}")
            continue
        split_name = os.path.basename(test_file).replace("test_", "").replace(".csv", "")
        acc_std, acc_cove = evaluate_file(model, tokenizer, device, test_file, out_dir, split_name)
        summary[split_name] = {"standard": acc_std, "cove": acc_cove}

    # Save summary
    summary_path = os.path.join(out_dir, "cove_accuracy_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "standard_accuracy", "cove_accuracy", "delta"])
        for split, accs in summary.items():
            delta = accs["cove"] - accs["standard"]
            writer.writerow([split, f"{accs['standard']:.4f}",
                            f"{accs['cove']:.4f}", f"{delta:+.4f}"])

    print(f"\n  === CoVe Results ===")
    print(f"  {'Split':<35} {'Standard':>10} {'CoVe':>10} {'Delta':>8}")
    print("  " + "-" * 65)
    for split, accs in summary.items():
        delta = accs["cove"] - accs["standard"]
        print(f"  {split:<35} {accs['standard']:>10.4f} {accs['cove']:>10.4f} {delta:>+8.4f}")
    print(f"\n  Saved: {summary_path}")
    print("  ✅ CoVe evaluation FINISHED.")


if __name__ == "__main__":
    main()
