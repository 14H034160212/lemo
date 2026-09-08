"""
evaluate_real_world_v2.py

Re-run of the LogicNLI / MNLI-contradiction transfer evaluation reported in the
paper's out-of-distribution transfer table.

Why this script exists: the original run's checkpoints are gone
(`trained_models/qwen_stage2_dpo` and `checkpoints/real_world_sft` no longer
exist), and the only surviving prediction files under data/real_world/ record
0.000 accuracy with every prediction collapsed to "Unknown". This script
re-evaluates the checkpoints that DO survive, using exactly the same protocol
as evaluate_folio.py / evaluate_zebra.py so the numbers are directly comparable
to the FOLIO and ZebraLogic transfer results.

Protocol (identical to evaluate_folio.py):
    text  = <input_text from the prepared eval CSV>
    model = LoRA SEQ_CLS head on the frozen backbone, label 1=True / 0=False
    metric = per-row accuracy against target_text

Data: data/real_world/logicnli_eval.csv (500 rows, 4 balanced label groups)
      data/real_world/mnli_eval.csv     (349 rows, contradiction + entailment)
Neither file's rows were used in training these checkpoints (they are trained
on the synthetic LEMO corpus only), so this is a zero-shot transfer measurement.
"""

import argparse
import json
import os

os.environ.setdefault('HF_HOME', '.cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', '.cache/huggingface/datasets')
os.environ.setdefault('TRANSFORMERS_CACHE', '.cache/huggingface/transformers')

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODELS = {
    "qwen_stage1": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen"},
    "qwen_lire": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen_lire"},
    "qwen_rlvf": {"base": "Qwen/Qwen2-1.5B", "adapter": "trained_models/qwen_rlvf/checkpoint-3000"},
    "qwen3_rlvf": {"base": "/data/shared/qwen3/Qwen3-8B", "adapter": "trained_models/qwen3_rlvf/checkpoint-10000"},
}

DATASETS = {
    # As-shipped file: prepare_real_world_data.py mapped LogicNLI's STRING labels
    # through an int-keyed dict with a "False" default, collapsing all 500 rows to
    # False. Kept here so the size of that artefact is measurable, not hidden.
    "logicnli_asshipped": "data/real_world/logicnli_eval.csv",
    # Same 500 rows with labels rebuilt from the preserved original_label column:
    # entailment -> True; contradiction / neutral / self_contradiction -> False.
    "logicnli_fixed": "data/real_world/logicnli_eval_fixed.csv",
    "mnli_con": "data/real_world/mnli_eval.csv",
}


def load_rows(path, subset_col=None, subset_val=None):
    df = pd.read_csv(path)
    if subset_col and subset_col in df.columns and subset_val is not None:
        df = df[df[subset_col] == subset_val]
    rows = []
    for _, r in df.iterrows():
        label = str(r["target_text"]).strip()
        rows.append({
            "text": str(r["input_text"]),
            "label": 1 if label == "True" else 0,
            "group": str(r.get("original_label", r.get("type", ""))),
        })
    return rows


def _patched_adapter_dir(adapter_dir, cache={}):
    """Same fix as evaluate_folio.py: PEFT saves modules_to_save=null even though
    the trained classifier-head weights are present in adapter_model.safetensors.
    Patch a temp copy so the trained 'score' head is actually restored instead of
    being left randomly initialised."""
    if adapter_dir in cache:
        return cache[adapter_dir]
    tmp_dir = os.path.join("/tmp", "rw_eval_patched_adapters", adapter_dir.replace("/", "_"))
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


def evaluate(model_key, dataset_rows, device, out_dir):
    cfg = MODELS[model_key]
    print(f"\n=== {model_key}: base={cfg['base']} adapter={cfg['adapter']} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base"], num_labels=2, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, _patched_adapter_dir(cfg["adapter"])).to(device)
    model.eval()

    out = {}
    with torch.no_grad():
        for ds_name, rows in dataset_rows.items():
            correct = 0
            recs = []
            for i, row in enumerate(rows):
                enc = tokenizer(row["text"], truncation=True, padding="max_length",
                                max_length=512, return_tensors="pt").to(device)
                logits = model(**enc).logits
                pred = int(torch.argmax(logits, dim=-1).item())
                ok = int(pred == row["label"])
                correct += ok
                recs.append({"i": i, "group": row["group"], "gt": row["label"],
                             "pred": pred, "correct": ok})
                if (i + 1) % 100 == 0:
                    print(f"  [{ds_name}] {i+1}/{len(rows)} running acc={correct/(i+1):.4f}", flush=True)
            acc = correct / len(rows) if rows else 0.0
            print(f"  FINAL {model_key} / {ds_name}: {acc:.4f} ({correct}/{len(rows)})", flush=True)
            out[ds_name] = acc
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(recs).to_csv(
                os.path.join(out_dir, f"{model_key}_{ds_name}_predictions.csv"), index=False)
            # per-group breakdown, so a collapsed all-one-class predictor is visible
            g = pd.DataFrame(recs).groupby("group")["pred"].agg(["mean", "count"])
            print(f"    per-group mean predicted label:\n{g}", flush=True)

    del model, base_model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--out_dir", default="results/real_world_rerun")
    ap.add_argument("--output", default="results/real_world_rerun_summary.json")
    args = ap.parse_args()

    dataset_rows = {name: load_rows(path) for name, path in DATASETS.items()}
    for name, rows in dataset_rows.items():
        pos = sum(r["label"] for r in rows)
        print(f"{name}: {len(rows)} rows | True={pos} False={len(rows)-pos} "
              f"| majority-class baseline={max(pos, len(rows)-pos)/len(rows):.4f}")

    summary = {
        "datasets": {k: {"n": len(v),
                         "n_true": sum(r["label"] for r in v),
                         "majority_baseline": max(sum(r["label"] for r in v),
                                                  len(v) - sum(r["label"] for r in v)) / len(v)}
                     for k, v in dataset_rows.items()},
        "results": {},
    }
    for key in args.models:
        if not os.path.exists(MODELS[key]["adapter"]):
            print(f"!! skipping {key}: adapter missing at {MODELS[key]['adapter']}")
            continue
        summary["results"][key] = evaluate(key, dataset_rows, args.device, args.out_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote", args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
