"""
evaluate_verify_guard_baseline.py
A genuine neuro-symbolic "verify-then-correct" baseline: an LLM reasons about
facts+rules+questions with NO special contradiction-handling instructions
(naive prompt), and an independent external symbolic checker
(scripts/utils/forward_chain.detect_contradiction) inspects the raw facts
for a logical contradiction; if one is found, every answer for that row is
overridden to "False" (matching the benchmark's conservative semantics),
otherwise the LLM's own raw answer is used unmodified.

This is what eMVG asked for: a comparison against a neuro-symbolic
verification pipeline (in the spirit of VeriCoT-style CoT validation),
implemented honestly rather than by re-running a symbolic oracle that
duplicates the benchmark's own label generator wholesale (which would be
circular: see scripts/evaluation/evaluate_vericot_symbolic.py, which is a
relabeled forward-chaining prover, not a fair comparator, and is NOT used
here). The symbolic component here only ever performs one well-defined,
narrow, mechanical check -- "are these stated facts self-contradictory?" --
and never determines the Base/V2 answers itself; those come entirely from
the LLM's own unaided reasoning, so the comparison is not circular on those
splits.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.forward_chain import detect_contradiction

NAIVE_SYSTEM_PROMPT = """You are a logical reasoning assistant. Given facts and rules, answer each question with exactly "True" or "False".

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


def call_openai(client, model, facts, rules, questions):
    token_kwarg = {"max_completion_tokens": 256} if model.startswith("gpt-5") else {"max_tokens": 256}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(facts, rules, questions)},
        ],
        **token_kwarg,
    )
    return response.choices[0].message.content


def call_anthropic(client, model, facts, rules, questions):
    message = client.messages.create(
        model=model, max_tokens=1024, system=NAIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(facts, rules, questions)}],
    )
    return next((b.text for b in message.content if b.type == "text"), "")


def evaluate_file(call_fn, test_file, output_dir, split_name, max_samples=None, delay=0.3):
    with open(test_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_samples:
        rows = rows[:max_samples]

    print(f"\n  [{split_name}] {len(rows)} rows ...")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_predictions.csv")

    correct = 0
    total_q = 0
    n_guarded = 0
    results = []

    for i, row in enumerate(rows):
        facts, rules, questions_str = row.get("facts", ""), row.get("rules", ""), row.get("questions", "")
        gt_answers = [a.strip() for a in row.get("answers", "").split("|")]
        num_q = len(gt_answers)

        try:
            response = call_fn(facts, rules, questions_str)
            raw_answers = parse_answers(response, num_q)
        except Exception as e:
            print(f"    API error at row {i}: {e}")
            raw_answers = ["?"] * num_q
            time.sleep(5)

        contradiction = detect_contradiction(facts, rules)
        if contradiction:
            n_guarded += 1
            final_answers = ["F"] * num_q
        else:
            final_answers = raw_answers

        q_list = [q.strip() for q in questions_str.split("|")]
        for q, gt, pred in zip(q_list, gt_answers, final_answers):
            is_correct = (gt.strip() == pred.strip())
            correct += int(is_correct)
            total_q += 1
            results.append({"group_id": row.get("group_id", ""), "question": q, "ground_truth": gt,
                             "raw_llm_answer": raw_answers[q_list.index(q)] if q in q_list else "?",
                             "guard_fired": contradiction, "prediction": pred, "correct": int(is_correct)})

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(rows)} | running acc: {correct/total_q:.4f} | guard fired on {n_guarded} rows")
        time.sleep(delay)

    accuracy = correct / total_q if total_q else 0.0
    print(f"    Final accuracy: {accuracy:.4f} ({correct}/{total_q}); guard fired on {n_guarded}/{len(rows)} rows")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "question", "ground_truth", "raw_llm_answer",
                                               "guard_fired", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_files", nargs="+", default=["data/test_base.csv", "data/test_variant2.csv", "data/test_variant3.csv"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = args.output_dir or f"results/verify_guard/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        call_fn = lambda f, r, q: call_anthropic(client, args.model, f, r, q)
    else:
        from openai import OpenAI
        client = OpenAI()
        call_fn = lambda f, r, q: call_openai(client, args.model, f, r, q)

    print(f"\n  Provider: {args.provider} | Model: {args.model} | Max samples/split: {args.max_samples}\n")
    summary = {}
    for test_file in args.test_files:
        if not os.path.exists(test_file):
            print(f"  File not found: {test_file}, skipping.")
            continue
        split_name = Path(test_file).stem.replace("test_", "")
        acc = evaluate_file(call_fn, test_file, out_dir, split_name, max_samples=args.max_samples, delay=args.delay)
        summary[split_name] = acc

    summary_path = os.path.join(out_dir, "accuracy_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "accuracy"])
        for split, acc in summary.items():
            writer.writerow([split, f"{acc:.4f}"])
    print(f"\n  === {args.model} (naive LLM + external contradiction guard) Results ===")
    for split, acc in summary.items():
        print(f"    {split}: {acc:.4f}")
    print("  FINISHED.")


if __name__ == "__main__":
    main()
