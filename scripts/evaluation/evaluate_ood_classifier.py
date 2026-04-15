"""
evaluate_ood_classifier.py
Run Qwen3 (or any sequence-classification) model on OOD datasets:
  - LogicNLI (data/real_world/logicnli_eval.csv)
  - MNLI     (data/real_world/mnli_eval.csv)

Usage:
  python scripts/evaluation/evaluate_ood_classifier.py \
      --model_dir trained_models/qwen3_rlvf \
      --base_model qwen3 \
      --output_dir trained_models/qwen3_rlvf/ood_results
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

BASE_MODELS = {
    "bert":  "bert-base-uncased",
    "qwen":  "Qwen/Qwen2-1.5B",
    "qwen3": "/data/shared/qwen3/Qwen3-8B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

OOD_DATASETS = {
    "logicnli": "data/real_world/logicnli_eval.csv",
    "mnli":     "data/real_world/mnli_eval.csv",
}

# Label mapping: target_text → True/False for classification
# LogicNLI / MNLI: "True" → positive class, "False" → negative class
# Some rows may have "entailment", "contradiction", "neutral" → map accordingly
def normalize_label(label: str) -> str:
    """Normalize to 'T'/'F' to match evaluate.py convention."""
    label = label.strip().lower()
    if label in ["true", "entailment", "t", "1", "yes"]:
        return "T"
    return "F"


def load_model(model_dir, base_model_key):
    base_model_name = BASE_MODELS[base_model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Loaded: {model_dir} | base: {base_model_name} | device: {device}")
    return model, tokenizer, device


def predict(model, tokenizer, text, device, max_len=512):
    """Match evaluate.py: class 1 → T, class 0 → F."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_len, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    return "T" if pred_id == 1 else "F"


def evaluate_dataset(model, tokenizer, device, dataset_path, dataset_name, out_dir):
    print(f"\n  [{dataset_name}] Evaluating {dataset_path} ...")
    rows = []
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"    Total samples: {len(rows)}")

    correct = 0
    results = []
    for i, row in enumerate(rows):
        text = row.get("input_text", "")
        label = normalize_label(row.get("target_text", ""))
        pred = predict(model, tokenizer, text, device)
        is_correct = pred.strip().lower() == label.strip().lower()
        correct += int(is_correct)
        results.append({**row, "prediction": pred, "correct": int(is_correct)})
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(rows)} done ...")

    accuracy = correct / len(rows) if rows else 0.0
    print(f"    Accuracy: {accuracy:.4f} ({correct}/{len(rows)})")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_predictions.csv")
    if results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"    Saved to: {out_path}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to trained model directory")
    parser.add_argument("--base_model", required=True, choices=list(BASE_MODELS.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(OOD_DATASETS.keys()),
                        help="Which datasets to evaluate")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.model_dir, "ood_results")
    model, tokenizer, device = load_model(args.model_dir, args.base_model)

    summary = {}
    for ds_name in args.datasets:
        if ds_name not in OOD_DATASETS:
            print(f"  Unknown dataset: {ds_name}, skipping.")
            continue
        ds_path = OOD_DATASETS[ds_name]
        if not os.path.exists(ds_path):
            print(f"  Dataset not found: {ds_path}, skipping.")
            continue
        acc = evaluate_dataset(model, tokenizer, device, ds_path, ds_name, out_dir)
        summary[ds_name] = acc

    # Save summary
    summary_path = os.path.join(out_dir, "ood_accuracy_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "accuracy"])
        for ds, acc in summary.items():
            writer.writerow([ds, f"{acc:.4f}"])
    print(f"\n  OOD Summary saved to: {summary_path}")
    print("\n  === OOD Results ===")
    for ds, acc in summary.items():
        print(f"    {ds}: {acc:.4f}")
    print("\n  ✅ OOD Evaluation FINISHED.")


if __name__ == "__main__":
    main()
