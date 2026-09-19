"""
Out-of-distribution transfer, measured on the GENERATION path.

Why this exists
---------------
The paper's OOD transfer table reports a uniformly negative result: the trained
checkpoints do not transfer to LogicNLI / MNLI / FOLIO / ZebraLogic. Every one
of those numbers was produced by `evaluate_real_world_v2.py`, `evaluate_folio.py`
and `evaluate_zebra.py`, which all load the checkpoint as an
`AutoModelForSequenceClassification` LoRA head.

The September 2026 audit found that the two pipeline components implemented on
that classification path (LIRE, RLVF) fail on contradiction detection, while the
one component implemented on the generation path (Fusion-LRA, CAUSAL_LM) works.
The per-label breakdown of the OOD runs shows the same signature: the heads do
not transfer badly, they collapse to a constant.

    qwen_rlvf  (1.5B)  LogicNLI: predicts False on 500/500 items
    qwen3_rlvf (8B)    LogicNLI: predicts True  on >91% of every label group

A constant predictor tells us nothing about reasoning. So the OOD table may be
measuring the classification head rather than the prior we actually trained.
This script re-runs the same datasets through the generation path, where the
model must emit a reasoning trace and an explicit "Answer: True/False".

Reported per dataset, so a collapse stays visible:
    accuracy, majority-class baseline, rho_T (fraction predicted True),
    parse rate, and a per-original-label breakdown.
"""

import argparse
import json
import os
import re
import sys

_HF_CACHE = os.environ.get("HF_HOME", ".cache/huggingface")
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_CACHE, "transformers")

import pandas as pd
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.evaluation.evaluate_generative import generate_answers_batch, parse_answer
from scripts.utils.eval_provenance import (
    collect_provenance, write_provenance, sanity_columns, warn_if_uninformative,
)

# The CSV files already store their text in the LEMO prompt format
# ("Facts: ...\nQuestion: ...\nThink step by step."), so they feed the
# generation path unchanged. FOLIO is built into the same shape below.
CSV_DATASETS = {
    "logicnli": "data/real_world/logicnli_eval_fixed.csv",
    "mnli_con": "data/real_world/mnli_eval.csv",
}


def load_csv_rows(path):
    df = pd.read_csv(path)
    group_col = "original_label" if "original_label" in df.columns else (
        "type" if "type" in df.columns else None)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "prompt": str(r["input_text"]),
            "gold": "T" if str(r["target_text"]).strip() == "True" else "F",
            "group": str(r[group_col]) if group_col else "",
        })
    return rows


def load_folio(max_samples=None):
    """FOLIO, wrapped in the LEMO prompt format.

    'Uncertain' has no analogue in a binary T/F benchmark, so those items are
    dropped -- the same filter evaluate_folio.py applies, which keeps this
    directly comparable to the classification-path number in the paper.
    """
    from datasets import load_dataset, concatenate_datasets
    ds = concatenate_datasets([
        load_dataset("tasksource/folio", split="train"),
        load_dataset("tasksource/folio", split="validation"),
    ])
    rows = []
    for r in ds:
        label = r["label"]
        if label == "Uncertain":
            continue
        premises = r["premises"].replace("\n", " ")
        rows.append({
            "prompt": f"Facts: {premises}\nQuestion: {r['conclusion']}\n"
                      f"Think step by step.",
            "gold": "T" if label == "True" else "F",
            "group": str(label),
        })
        if max_samples and len(rows) >= max_samples:
            break
    return rows


_STRICT_RE = re.compile(r"\banswer\s*[:=]\s*(true|false)\b", re.I)


def strict_parse(text):
    """Return "T"/"F" only when the model actually emitted its answer protocol.

    `parse_answer()` in evaluate_generative.py ends in a `return "F"` default.
    In distribution that is harmless -- the trained models almost always emit
    "Answer: True/False". Out of distribution they do not: on LogicNLI this
    checkpoint finishes its trace and then writes

        "Answer: Baird is not serious."

    which matches no branch and silently becomes "F". Scoring that way
    manufactures exactly the constant-False signature the classification heads
    showed, so it cannot be used to decide whether the generation path
    transfers. We therefore report the parse rate first, accuracy over the
    parsed subset, and the F-default accuracy separately.
    """
    m = _STRICT_RE.search(text)
    if m:
        return "T" if m.group(1).lower() == "true" else "F"
    head = text.strip().lower()
    if head.startswith("true"):
        return "T"
    if head.startswith("false"):
        return "F"
    return None


def forced_answers(model, tokenizer, prompts, traces, device, batch_size=24):
    """Read the answer off the trace the model already wrote, for every item.

    Free generation only states an explicit answer on 7-46% of out-of-distribution
    items, so accuracy over the parsed subset is computed on a set the model
    selected for itself. This appends "\nAnswer:" to the trace it already
    produced and compares the next-token probability of " True" against
    " False". The reasoning is untouched -- only the read-out changes -- and
    coverage is 100% by construction, so no selection effect survives.
    """
    t_id = tokenizer(" True", add_special_tokens=False)["input_ids"][0]
    f_id = tokenizer(" False", add_special_tokens=False)["input_ids"][0]
    if t_id == f_id:
        raise RuntimeError("' True'/' False' share a first token; forced read-out invalid")
    texts = [p + t + "\nAnswer:" for p, t in zip(prompts, traces)]

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"          # last real token lands at index -1
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=1024).to(device)
                logits = model(**enc).logits[:, -1, :].float()
                out.extend("T" if a > b else "F" for a, b in
                           zip(logits[:, t_id].tolist(), logits[:, f_id].tolist()))
    finally:
        tokenizer.padding_side = original_side
    return out


def evaluate_dataset(name, rows, model, tokenizer, device, args, out_dir):
    prompts = [r["prompt"] for r in rows]
    print(f"\n--- {name}: {len(rows)} rows", flush=True)
    gens = generate_answers_batch(model, tokenizer, prompts, device,
                                  max_new_tokens=args.max_new_tokens,
                                  batch_size=args.batch_size)
    gold = [r["gold"] for r in rows]
    strict = [strict_parse(g) for g in gens]              # None when unparseable
    lenient = [parse_answer(g) for g in gens]             # paper protocol, F-default

    n_parsed = sum(p is not None for p in strict)
    strict_rate = n_parsed / len(rows)
    parsed_pairs = [(p, g) for p, g in zip(strict, gold) if p is not None]
    acc_parsed = (sum(p == g for p, g in parsed_pairs) / len(parsed_pairs)
                  if parsed_pairs else float("nan"))
    rho_t_parsed = (sum(p == "T" for p, _ in parsed_pairs) / len(parsed_pairs)
                    if parsed_pairs else float("nan"))

    forced = None
    if args.forced_answer:
        forced = forced_answers(model, tokenizer, prompts, gens, device,
                                batch_size=max(8, args.batch_size // 2))
        f_correct = sum(p == g for p, g in zip(forced, gold))
        f_acc = f_correct / len(rows)
        f_rho = sum(p == "T" for p in forced) / len(forced)
        f_sanity = sanity_columns(gold, forced)

    correct = sum(p == g for p, g in zip(lenient, gold))
    acc = correct / len(rows)
    sanity = sanity_columns(gold, lenient, generated=gens)
    rho_t = sum(p == "T" for p in lenient) / len(lenient)

    print(f"  protocol kept    : {strict_rate:.4f}  "
          f"({n_parsed}/{len(rows)} emitted an explicit Answer: True/False)")
    print(f"  acc | parsed only: {acc_parsed:.4f}  (n={n_parsed}, "
          f"rho_T={rho_t_parsed:.4f})")
    print(f"  acc | F-default  : {acc:.4f}  ({correct}/{len(rows)}, "
          f"rho_T={rho_t:.4f})  <- paper protocol")
    if forced is not None:
        print(f"  acc | forced     : {f_acc:.4f}  ({f_correct}/{len(rows)}, "
              f"rho_T={f_rho:.4f})  <- 100% coverage, no selection")
        warn_if_uninformative(name + "/forced", f_acc, f_sanity)
    print(f"  majority baseline: {sanity['majority_class_baseline']:.4f}")
    warn_if_uninformative(name, acc, sanity)
    if strict_rate < 0.80:
        print(f"  !! [{name}] the F-default accuracy above is NOT a reasoning "
              f"measurement: {1-strict_rate:.1%} of traces never stated an "
              f"answer and were scored as False by the parser's fallback.",
              flush=True)

    recs = pd.DataFrame({
        "group": [r["group"] for r in rows],
        "gold": gold,
        "pred_strict": ["" if p is None else p for p in strict],
        "pred_lenient": lenient,
        "protocol_kept": [int(p is not None) for p in strict],
        "pred_forced": forced if forced is not None else [""] * len(rows),
        "correct": [int(p == g) for p, g in zip(lenient, gold)],
        "generated": gens,
    })
    os.makedirs(out_dir, exist_ok=True)
    recs.to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)

    if recs["group"].nunique() > 1:
        brk = recs.groupby("group").agg(
            n=("pred_lenient", "size"),
            protocol=("protocol_kept", "mean"),
            rho_T=("pred_lenient", lambda s: (s == "T").mean()),
            acc=("correct", "mean"))
        print("  per-label breakdown (rho_T at 0 or 1 means collapsed):")
        print("    " + brk.round(3).to_string().replace("\n", "\n    "))

    return {"n": len(rows),
            "protocol_kept": round(strict_rate, 4),
            "n_parsed": n_parsed,
            "accuracy_parsed_only": (None if parsed_pairs is None or not parsed_pairs
                                     else round(acc_parsed, 4)),
            "rho_T_parsed": (None if not parsed_pairs else round(rho_t_parsed, 4)),
            "accuracy_forced": (round(f_acc, 4) if forced is not None else None),
            "rho_T_forced": (round(f_rho, 4) if forced is not None else None),
            "accuracy_f_default": round(acc, 4),
            "rho_T_f_default": round(rho_t, 4),
            "majority_baseline": sanity["majority_class_baseline"],
            "pred_concentration": sanity["pred_concentration"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="trained_models/qwen_fusion_sft_conflict_aware",
                    help="LoRA CAUSAL_LM checkpoint, e.g. trained_models/qwen_fusion_sft_conflict_aware")
    ap.add_argument("--datasets", nargs="+",
                    default=["logicnli", "mnli_con", "folio"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--base_only", metavar="HF_ID",
                    help="control: run the untrained backbone, no adapter. If the "
                         "base model already scores what the trained checkpoint "
                         "scores, the transfer belongs to the backbone and the "
                         "read-out protocol, not to our training.")
    ap.add_argument("--forced_answer", action="store_true",
                    help="also read the answer off every trace (100%% coverage)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    tag = ("BASE_" + os.path.basename(args.base_only.rstrip("/"))
           if args.base_only else os.path.basename(args.model_dir.rstrip("/")))
    out_dir = args.out_dir or f"results/ood_generative/{tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {}
    for name in args.datasets:
        if name in CSV_DATASETS:
            data[name] = load_csv_rows(CSV_DATASETS[name])
        elif name == "folio":
            try:
                data[name] = load_folio()
            except Exception as e:
                print(f"!! skipping folio: {e}", flush=True)
        else:
            print(f"!! unknown dataset {name}", flush=True)
    if args.max_rows:
        data = {k: v[:args.max_rows] for k, v in data.items()}

    for k, v in data.items():
        pos = sum(r["gold"] == "T" for r in v)
        print(f"{k}: {len(v)} rows | T={pos} F={len(v)-pos} "
              f"| majority baseline={max(pos, len(v)-pos)/len(v):.4f}")

    print(f"\nLoading {args.model_dir} on the generation path (CAUSAL_LM)", flush=True)
    if args.base_only:
        from transformers import AutoModelForCausalLM
        print(f"CONTROL: untrained backbone {args.base_only}, no adapter")
        tokenizer = AutoTokenizer.from_pretrained(args.base_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_only, torch_dtype=torch.bfloat16).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    results = {}
    for name, rows in data.items():
        results[name] = evaluate_dataset(name, rows, model, tokenizer,
                                         device, args, out_dir)

    summary_path = os.path.join(out_dir, "summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"model_dir": args.model_dir, "path": "generative",
                   "results": results}, f, indent=2)
    write_provenance(summary_path, collect_provenance(
        {k: CSV_DATASETS.get(k, "huggingface:tasksource/folio") for k in data},
        model_dir=args.model_dir,
        extra={"eval_path": "generative", "max_new_tokens": args.max_new_tokens}))

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
