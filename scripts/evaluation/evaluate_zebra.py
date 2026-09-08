"""
evaluate_zebra.py
Evaluate trained SEQ_CLS checkpoints on ZebraLogic puzzles solved by our own
constraint solver (scripts/evaluation/zebra_solver.py), since ZebraLogicBench's
official public test split withholds ground-truth answers.

For each solved puzzle we build exactly two balanced binary probes:
  - a True probe using the solver-derived correct choice
  - a False probe using one randomly-sampled incorrect choice
Input format matches training (see stage4_train_rlvf.py):
    text = <full puzzle text> + " Question: " + <the T/F assertion>
"""

import argparse
import json
import os
import random

os.environ.setdefault('HF_HOME', '.cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', '.cache/huggingface/datasets')
os.environ.setdefault('TRANSFORMERS_CACHE', '.cache/huggingface/transformers')

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODELS = {
    "qwen_stage1": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen"},
    "qwen_rlvf": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen_rlvf/checkpoint-3000"},
    "qwen3_rlvf": {"base": "/data/shared/qwen3/Qwen3-8B", "adapter": "trained_models/qwen3_rlvf/checkpoint-10000"},
}


def _compact_clues(puzzle_text):
    """Strip markdown/backticks/scene-setting boilerplate; keep only the bare
    clue sentences, pipe-joined -- mirroring the 'facts + rules' training
    format (short declarative sentences, no numbering, no headers)."""
    clue_lines = [re.sub(r"^\s*\d+\.\s*", "", l).strip().rstrip(".")
                  for l in puzzle_text.split("\n") if re.match(r"^\s*\d+\.", l)]
    return " | ".join(clue_lines)


def build_probes(solved, seed=0, compact=True):
    rng = random.Random(seed)
    probes = []
    for item in solved:
        house_match = re.search(r"House (\d+)", item["question"])
        if not house_match:
            continue
        house = house_match.group(1)
        attr_match = re.match(r"What is (\w+) of", item["question"])
        attr = attr_match.group(1) if attr_match else "attribute"
        correct = item["answer"]
        wrong_choices = [c for c in item["choices"] if c != correct]
        if not wrong_choices:
            continue
        wrong = rng.choice(wrong_choices)

        if compact:
            base_text = _compact_clues(item["puzzle"])
        else:
            base_text = item["puzzle"]
        probes.append({
            "text": f"{base_text} Q: is the {attr} of the person in house {house} {correct}?",
            "label": 1,
        })
        probes.append({
            "text": f"{base_text} Q: is the {attr} of the person in house {house} {wrong}?",
            "label": 0,
        })
    return probes


def _patched_adapter_dir(adapter_dir, cache={}):
    if adapter_dir in cache:
        return cache[adapter_dir]
    tmp_dir = os.path.join("/tmp", "zebra_eval_patched_adapters", adapter_dir.replace("/", "_"))
    os.makedirs(tmp_dir, exist_ok=True)
    import shutil
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        shutil.copy(os.path.join(adapter_dir, fname), os.path.join(tmp_dir, fname))
    cfg_path = os.path.join(tmp_dir, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    cfg["modules_to_save"] = ["score"]
    json.dump(cfg, open(cfg_path, "w"))
    cache[adapter_dir] = tmp_dir
    return tmp_dir


def evaluate(model_key, probes, device, max_length=512):
    cfg = MODELS[model_key]
    print(f"\n=== {model_key}: base={cfg['base']} adapter={cfg['adapter']} ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base"], num_labels=2, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    patched_dir = _patched_adapter_dir(cfg["adapter"])
    model = PeftModel.from_pretrained(base_model, patched_dir).to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, p in enumerate(probes):
            enc = tokenizer(p["text"], truncation=True, padding="max_length",
                             max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            pred = int(torch.argmax(logits, dim=-1).item())
            correct += int(pred == p["label"])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(probes)} running acc={correct/(i+1):.4f}")

    acc = correct / len(probes) if probes else 0.0
    print(f"  FINAL {model_key} ZebraLogic (solver-derived) accuracy: {acc:.4f} ({correct}/{len(probes)})")
    del model, base_model
    torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solved_json", default="results/zebra_solved.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/zebra_eval_summary.json")
    args = parser.parse_args()

    solved = json.load(open(args.solved_json))
    probes = build_probes(solved)
    print(f"ZebraLogic: {len(solved)} solver-derived puzzles -> {len(probes)} balanced T/F probes.")

    summary = {"n_puzzles": len(solved), "n_probes": len(probes), "results": {}}
    for key in args.models:
        acc = evaluate(key, probes, args.device)
        summary["results"][key] = acc

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
