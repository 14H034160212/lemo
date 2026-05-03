"""
evaluate_gemma_local.py
Local-inference evaluation of Gemma 3 4B-IT on the LEMO benchmark.
Matches the prompt template, parser, and CSV output schema of
evaluate_frontier_api.py so Gemma numbers are directly comparable to
the GPT-4o frontier baseline.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_gemma_local.py \
      --model google/gemma-3-4b-it \
      --test_files data/test_base.csv data/test_variant2.csv data/test_variant3.csv \
                   data/test_hard_mixed.csv data/test_variant4_equiv_double_negation.csv \
      --output_dir results/frontier/gemma-3-4b-it \
      --max_samples 200
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


SYSTEM_PROMPT = """You are a logical reasoning assistant. Given facts and rules, answer each question with exactly "True" or "False".

IMPORTANT: If the facts contain contradictions (e.g., "X is cold" AND "X is not cold"), use the contradiction to determine the answer: contradictory premises make ANY statement False.

Always answer with ONLY "True" or "False" for each question, in order."""


def build_user_prompt(facts: str, rules: str, questions: str) -> str:
    q_list = [q.strip() for q in questions.split("|")]
    q_formatted = "\n".join(q_list)
    return (
        f"Facts: {facts}\n\n"
        f"Rules: {rules}\n\n"
        f"Questions:\n{q_formatted}\n\n"
        f"Answer each question (True/False), one per line, in order."
    )


def parse_answers(response_text: str, num_questions: int) -> list:
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
    answers = []
    for line in lines:
        line_lower = line.lower()
        if "true" in line_lower and "false" not in line_lower:
            answers.append("T")
        elif "false" in line_lower:
            answers.append("F")
        if len(answers) == num_questions:
            break
    while len(answers) < num_questions:
        answers.append("?")
    return answers[:num_questions]


def make_call_fn(model_id: str, max_new_tokens: int = 256):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  loading {model_id} (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print("  model ready.", flush=True)

    # Some chat templates (Gemma) merge system into first user turn.
    # We try the "system" role first; if the template rejects it we fall back.
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True,
        )
        supports_system = True
    except Exception:
        supports_system = False

    @torch.no_grad()
    def call(facts, rules, questions):
        user_msg = build_user_prompt(facts, rules, questions)
        if supports_system:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        else:
            # merge system into user (Gemma-style)
            messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_msg}]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return call


def evaluate_file(call_fn, test_file, output_dir, split_name, max_samples=None):
    rows = []
    with open(test_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_samples:
        rows = rows[:max_samples]

    print(f"\n  [{split_name}] {len(rows)} rows ...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_predictions.csv")

    correct, total_q, results = 0, 0, []
    t0 = time.time()
    for i, row in enumerate(rows):
        facts = row.get("facts", "")
        rules = row.get("rules", "")
        questions_str = row.get("questions", "")
        ground_truth_str = row.get("answers", "")

        gt_answers = [a.strip() for a in ground_truth_str.split("|")]
        num_q = len(gt_answers)

        try:
            response = call_fn(facts, rules, questions_str)
            pred_answers = parse_answers(response, num_q)
        except Exception as e:
            print(f"    inference error at row {i}: {e}", flush=True)
            pred_answers = ["?"] * num_q
            response = f"ERROR: {e}"

        q_list = [q.strip() for q in questions_str.split("|")]
        for q, gt, pred in zip(q_list, gt_answers, pred_answers):
            is_correct = (gt.strip() == pred.strip())
            correct += int(is_correct)
            total_q += 1
            results.append({
                "group_id": row.get("group_id", ""),
                "type": row.get("type", ""),
                "question": q,
                "ground_truth": gt,
                "prediction": pred,
                "correct": int(is_correct),
            })

        if (i + 1) % 25 == 0:
            acc = correct / total_q if total_q else 0
            dt = time.time() - t0
            print(f"    {i+1}/{len(rows)} | running acc: {acc:.4f} | "
                  f"{dt:.0f}s elapsed", flush=True)

    accuracy = correct / total_q if total_q else 0.0
    print(f"    Final accuracy: {accuracy:.4f} ({correct}/{total_q} questions)",
          flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "type", "question",
                                               "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"    Saved: {out_path}", flush=True)
    return accuracy, correct, total_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--test_files", nargs="+",
                        default=["data/test_base.csv",
                                 "data/test_variant2.csv",
                                 "data/test_variant3.csv",
                                 "data/test_hard_mixed.csv",
                                 "data/test_variant4_equiv_double_negation.csv"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    out_dir = args.output_dir or f"results/frontier/{args.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Model: {args.model}\n  Output: {out_dir}\n  "
          f"max_samples per split: {args.max_samples}", flush=True)

    call_fn = make_call_fn(args.model, max_new_tokens=args.max_new_tokens)

    summary_rows = []
    for test_file in args.test_files:
        if not os.path.exists(test_file):
            print(f"  File not found: {test_file}, skipping.", flush=True)
            continue
        split_name = Path(test_file).stem.replace("test_", "")
        acc, c, t = evaluate_file(call_fn, test_file, out_dir, split_name,
                                  max_samples=args.max_samples)
        summary_rows.append((split_name, c, t, acc))

    summary_path = os.path.join(out_dir, "accuracy_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "correct", "total", "accuracy"])
        for split, c, t, acc in summary_rows:
            w.writerow([split, c, t, f"{acc:.4f}"])

    print(f"\n  === {args.model} Results ===", flush=True)
    for split, c, t, acc in summary_rows:
        print(f"    {split}: {acc:.4f} ({c}/{t})", flush=True)
    print(f"\n  Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
