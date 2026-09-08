"""
evaluate_folio.py
Evaluate trained SEQ_CLS checkpoints (Stage-1 SFT baseline vs full Fusion-Conflict
RLVF pipeline) on the external FOLIO benchmark (tasksource/folio).
Addresses reviewer critique: "experiments rely primarily on a synthetic
constructed benchmark ... lack validation on standard logical reasoning
datasets such as FOLIO, ZebraLogic".

Input format matches training exactly (see scripts/training/stage4_train_rlvf.py):
    text = premises + " " + conclusion
fed directly into the LoRA sequence-classification head, label 1=True/0=False.
FOLIO's "Uncertain" label has no analogue in our binary T/F benchmark, so
(matching how MNLI's "neutral" class was handled in prepare_real_world_data.py)
those rows are excluded and the drop count is reported.
"""

import argparse
import json
import os

os.environ.setdefault('HF_HOME', '.cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', '.cache/huggingface/datasets')
os.environ.setdefault('TRANSFORMERS_CACHE', '.cache/huggingface/transformers')

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODELS = {
    "qwen_stage1": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen"},
    "qwen_rlvf": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen_rlvf/checkpoint-3000"},
    "qwen3_rlvf": {"base": "/data/shared/qwen3/Qwen3-8B", "adapter": "trained_models/qwen3_rlvf/checkpoint-10000"},
}


def load_folio(max_samples=None):
    # Combine train+validation for a larger evaluation sample (811 True/False
    # rows total vs. 134 in validation alone); FOLIO's "test" split has no
    # public labels, so train+validation is the full labeled pool available.
    ds = concatenate_datasets([
        load_dataset("tasksource/folio", split="train"),
        load_dataset("tasksource/folio", split="validation"),
    ])
    rows = []
    dropped_uncertain = 0
    for r in ds:
        label = r["label"]
        if label == "Uncertain":
            dropped_uncertain += 1
            continue
        rows.append({
            "premises": r["premises"].replace("\n", " "),
            "conclusion": r["conclusion"],
            "label": 1 if label == "True" else 0,
        })
    if max_samples:
        rows = rows[:max_samples]
    return rows, dropped_uncertain


def _patched_adapter_dir(adapter_dir, cache={}):
    """PEFT's saved adapter_config.json has modules_to_save=null even though the
    trained classifier-head weights ARE present in adapter_model.safetensors
    (confirmed: the tensor is there, just not wired up for restore without this).
    Patch a temp copy of the config so PeftModel.from_pretrained actually restores
    the trained 'score' head instead of leaving it randomly initialized."""
    if adapter_dir in cache:
        return cache[adapter_dir]
    tmp_dir = os.path.join("/tmp", "folio_eval_patched_adapters", adapter_dir.replace("/", "_"))
    os.makedirs(tmp_dir, exist_ok=True)
    import shutil, json as _json
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        shutil.copy(os.path.join(adapter_dir, fname), os.path.join(tmp_dir, fname))
    cfg_path = os.path.join(tmp_dir, "adapter_config.json")
    cfg = _json.load(open(cfg_path))
    cfg["modules_to_save"] = ["score"]
    _json.dump(cfg, open(cfg_path, "w"))
    cache[adapter_dir] = tmp_dir
    return tmp_dir


def evaluate(model_key, rows, device):
    cfg = MODELS[model_key]
    print(f"\n=== {model_key}: base={cfg['base']} adapter={cfg['adapter']} ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base"], num_labels=2, torch_dtype=torch.bfloat16
    ).to(device)
    # Required for Qwen2ForSequenceClassification's pooling to find the real
    # last-token position instead of a padding position (see investigation notes).
    base_model.config.pad_token_id = tokenizer.pad_token_id
    patched_dir = _patched_adapter_dir(cfg["adapter"])
    model = PeftModel.from_pretrained(base_model, patched_dir).to(device)
    model.eval()

    correct = 0
    results = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            text = row["premises"] + " " + row["conclusion"]
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=512,
                             return_tensors="pt").to(device)
            logits = model(**enc).logits
            pred = int(torch.argmax(logits, dim=-1).item())
            is_correct = int(pred == row["label"])
            correct += is_correct
            results.append({"i": i, "gt": row["label"], "pred": pred, "correct": is_correct})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rows)} running acc={correct/(i+1):.4f}")

    acc = correct / len(rows) if rows else 0.0
    print(f"  FINAL {model_key} FOLIO accuracy: {acc:.4f} ({correct}/{len(rows)})")

    del model, base_model
    torch.cuda.empty_cache()
    return acc, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--output", default="results/folio_eval_summary.json")
    args = parser.parse_args()

    rows, dropped = load_folio(args.max_samples)
    print(f"FOLIO: {len(rows)} True/False rows loaded ({dropped} Uncertain rows dropped).")

    summary = {"n_rows": len(rows), "n_dropped_uncertain": dropped, "results": {}}
    for key in args.models:
        acc, _ = evaluate(key, rows, args.device)
        summary["results"][key] = acc

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
