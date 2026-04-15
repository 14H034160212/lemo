"""
evaluate_frontier_api.py
Evaluate frontier models (Claude, GPT-4o) on the benchmark via API.
Addresses reviewer critique: "frontier models not tested"

Usage:
  # Claude (Anthropic API):
  ANTHROPIC_API_KEY=sk-ant-... python scripts/evaluation/evaluate_frontier_api.py \
      --provider anthropic --model claude-opus-4-6 \
      --test_files data/test_base.csv data/test_variant3.csv \
      --output_dir results/frontier/claude-opus-4-6

  # GPT-4o (OpenAI API):
  OPENAI_API_KEY=sk-... python scripts/evaluation/evaluate_frontier_api.py \
      --provider openai --model gpt-4o \
      --test_files data/test_base.csv data/test_variant3.csv \
      --output_dir results/frontier/gpt-4o
"""

import argparse
import os
import csv
import time
import json
import sys
from pathlib import Path

# =========================================================
# PROMPT TEMPLATE
# =========================================================

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
    """Extract True/False answers from model response."""
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
    # Pad if short
    while len(answers) < num_questions:
        answers.append("?")
    return answers[:num_questions]


# =========================================================
# API CLIENTS
# =========================================================

def call_anthropic(client, model: str, facts: str, rules: str, questions: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(facts, rules, questions)}]
    )
    return message.content[0].text


def call_openai(client, model: str, facts: str, rules: str, questions: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(facts, rules, questions)},
        ]
    )
    return response.choices[0].message.content


# =========================================================
# EVALUATION
# =========================================================

def evaluate_file(call_fn, test_file: str, output_dir: str, split_name: str,
                  max_samples: int = None, delay: float = 0.5):
    rows = []
    with open(test_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_samples:
        rows = rows[:max_samples]

    print(f"\n  [{split_name}] {len(rows)} rows ...")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_predictions.csv")

    correct = 0
    total_q = 0
    results = []

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
            print(f"    API error at row {i}: {e}")
            pred_answers = ["?"] * num_q
            response = f"ERROR: {e}"
            time.sleep(5)

        q_list = [q.strip() for q in questions_str.split("|")]
        for qi, (q, gt, pred) in enumerate(zip(q_list, gt_answers, pred_answers)):
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

        if (i + 1) % 50 == 0:
            acc = correct / total_q if total_q else 0
            print(f"    {i+1}/{len(rows)} | running acc: {acc:.4f}")

        time.sleep(delay)

    accuracy = correct / total_q if total_q else 0.0
    print(f"    Final accuracy: {accuracy:.4f} ({correct}/{total_q} questions)")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "type", "question",
                                               "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"    Saved: {out_path}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    parser.add_argument("--model", required=True, help="e.g. claude-opus-4-6, gpt-4o")
    parser.add_argument("--test_files", nargs="+",
                        default=["data/test_base.csv", "data/test_variant3.csv",
                                 "data/test_hard_mixed.csv"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max rows per split (default 200 to limit API cost)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls")
    args = parser.parse_args()

    out_dir = args.output_dir or f"results/frontier/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    # Init API client
    if args.provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("Install anthropic: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        call_fn = lambda f, r, q: call_anthropic(client, args.model, f, r, q)

    elif args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("Install openai: pip install openai")
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        call_fn = lambda f, r, q: call_openai(client, args.model, f, r, q)

    print(f"\n  Provider: {args.provider} | Model: {args.model}")
    print(f"  Max samples per split: {args.max_samples}")
    print(f"  Output: {out_dir}\n")

    summary = {}
    for test_file in args.test_files:
        if not os.path.exists(test_file):
            print(f"  File not found: {test_file}, skipping.")
            continue
        split_name = Path(test_file).stem.replace("test_", "")
        acc = evaluate_file(call_fn, test_file, out_dir, split_name,
                           max_samples=args.max_samples, delay=args.delay)
        summary[split_name] = acc

    # Save summary
    summary_path = os.path.join(out_dir, "accuracy_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "accuracy"])
        for split, acc in summary.items():
            writer.writerow([split, f"{acc:.4f}"])

    print(f"\n  === {args.model} Results ===")
    for split, acc in summary.items():
        print(f"    {split}: {acc:.4f}")
    print(f"\n  Summary saved to: {summary_path}")
    print("  ✅ Frontier evaluation FINISHED.")


if __name__ == "__main__":
    main()
