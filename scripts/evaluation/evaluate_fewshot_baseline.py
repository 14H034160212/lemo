"""
evaluate_fewshot_baseline.py
Verification-first few-shot prompting baseline (no training) — the strong
inference-time comparator reviewer 2hi4 asked for: "a strong inference-time
baseline that explicitly instructs the model to normalize rules, check
consistency, identify missing dependencies, and halt before deduction, with
few-shot examples covering V2/V3/V4."

Same protocol/parsing as scripts/evaluation/evaluate_frontier_api.py so numbers
are directly comparable to the existing zero-shot GPT-4o/Gemma baseline.
"""

import argparse
import os
import csv
import time
import sys
from pathlib import Path

SYSTEM_PROMPT = """You are a careful logical-reasoning assistant. Before answering, ALWAYS follow this procedure:
Step 1: Normalize the rules (rewrite each rule in plain if-then form).
Step 2: Check the facts for any internal contradiction (e.g. "X is P" and "X is not P").
Step 3: If a contradiction exists, HALT deduction — every question about this instance is False.
Step 4: If no contradiction, check that every fact needed to answer each question is actually derivable from the given rules (do not assume a missing link). If a required rule is absent, the question is False.
Step 5: Otherwise, deduce the answer by chaining the rules forward.

Always end with your final answers as "True" or "False" for each question, one per line, in order. Show your step 1-5 reasoning first, then the final answers."""

FEWSHOT_EXAMPLES = [
    # Variant 2 style: essential rule removed -> some conclusions no longer derivable -> False
    {
        "facts": "Frank is green or purple",
        "rules": "If someone is green then they are cold. | If someone is purple then they are cold. | "
                 "If someone is cold then they are rough. | If someone is young then they are cold. | "
                 "If someone is young then they are nice.",
        "questions": "Q1: Frank is cold. | Q2: Frank is rough. | Q3: Frank is young. | Q4: Frank is nice.",
        "answer": "Step 1: Rules -> green->cold, purple->cold, cold->rough, young->cold, young->nice.\n"
                  "Step 2: Facts (green or purple) contain no contradiction.\n"
                  "Step 3: No contradiction, continue.\n"
                  "Step 4: cold is derivable (green/purple->cold), rough is derivable (cold->rough). "
                  "But no rule derives 'young' from the given facts -- young is NOT derivable, "
                  "so 'nice' (which requires young) is also NOT derivable.\n"
                  "Step 5: cold=True, rough=True, young=False (not derivable), nice=False (not derivable).\n"
                  "True\nTrue\nFalse\nFalse",
    },
    # Variant 3 style: contradiction injected -> halt -> all False
    {
        "facts": "Frank is green or purple | Frank is not cold",
        "rules": "If someone is green then they are cold. | If someone is purple then they are cold. | "
                 "If someone is cold then they are rough. | If someone is not young then they are not rough. | "
                 "If someone is young then they are cold. | If someone is young then they are nice.",
        "questions": "Q1: Frank is cold. | Q2: Frank is rough. | Q3: Frank is young. | Q4: Frank is nice.",
        "answer": "Step 1: Rules normalized: green->cold, purple->cold, cold->rough, not(young)->not(rough), young->cold, young->nice.\n"
                  "Step 2: Facts say Frank is green-or-purple (implies cold), but ALSO explicitly 'Frank is not cold'. "
                  "This is a direct contradiction (cold and not-cold).\n"
                  "Step 3: Contradiction detected -> HALT. Every question about this instance is False.\n"
                  "False\nFalse\nFalse\nFalse",
    },
    # Variant 4 style: logic-preserving rewrite (De Morgan) of a rule -> same answers as the original
    {
        "facts": "Frank is green or purple",
        "rules": "If someone is not green and not purple then they are not cold. | If someone is green then they are cold. | "
                 "If someone is purple then they are cold. | If someone is cold then they are rough. | "
                 "If someone is not young then they are not rough. | If someone is young then they are cold. | "
                 "If someone is young then they are nice.",
        "questions": "Q1: Frank is cold. | Q2: Frank is rough. | Q3: Frank is young. | Q4: Frank is nice.",
        "answer": "Step 1: The first rule 'not green and not purple -> not cold' is the De Morgan-equivalent "
                  "contrapositive of 'green or purple -> cold' -- it is the SAME logical content as the other two rules, "
                  "just rephrased, not a new independent constraint.\n"
                  "Step 2: Facts (green or purple) contain no contradiction.\n"
                  "Step 3: No contradiction, continue.\n"
                  "Step 4: cold is derivable, rough is derivable (cold->rough), and here 'not(young)->not(rough)' "
                  "combined with rough=True and the contrapositive gives young=True, and young->nice gives nice=True.\n"
                  "Step 5: A logic-preserving rewrite must not change the answers versus the equivalent original rule set.\n"
                  "True\nTrue\nTrue\nTrue",
    },
]


def build_fewshot_messages(facts: str, rules: str, questions: str) -> list:
    msgs = []
    for ex in FEWSHOT_EXAMPLES:
        user_msg = f"Facts: {ex['facts']}\n\nRules: {ex['rules']}\n\nQuestions:\n" + \
                   "\n".join(q.strip() for q in ex["questions"].split("|")) + \
                   "\n\nAnswer each question (True/False), one per line, in order."
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    q_list = [q.strip() for q in questions.split("|")]
    final_user = f"Facts: {facts}\n\nRules: {rules}\n\nQuestions:\n" + "\n".join(q_list) + \
                 "\n\nAnswer each question (True/False), one per line, in order."
    msgs.append({"role": "user", "content": final_user})
    return msgs


def parse_answers(response_text: str, num_questions: int) -> list:
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
    answers = []
    for line in lines:
        line_lower = line.lower()
        if "true" in line_lower and "false" not in line_lower:
            answers.append("T")
        elif "false" in line_lower:
            answers.append("F")
    # keep only the LAST num_questions T/F tokens (final answers, not the reasoning trace)
    answers = answers[-num_questions:] if len(answers) >= num_questions else answers
    while len(answers) < num_questions:
        answers.append("?")
    return answers[:num_questions]


def call_anthropic(client, model, facts, rules, questions):
    messages = build_fewshot_messages(facts, rules, questions)
    message = client.messages.create(model=model, max_tokens=1024, system=SYSTEM_PROMPT, messages=messages)
    return next((b.text for b in message.content if b.type == "text"), "")


def call_openai(client, model, facts, rules, questions):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_fewshot_messages(facts, rules, questions)
    token_kwarg = {"max_completion_tokens": 1024} if model.startswith("gpt-5") else {"max_tokens": 1024}
    response = client.chat.completions.create(model=model, messages=messages, **token_kwarg)
    return response.choices[0].message.content


def evaluate_file(call_fn, test_file, output_dir, split_name, max_samples=None, delay=0.5):
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
        facts, rules, questions_str = row.get("facts", ""), row.get("rules", ""), row.get("questions", "")
        gt_answers = [a.strip() for a in row.get("answers", "").split("|")]
        num_q = len(gt_answers)
        try:
            response = call_fn(facts, rules, questions_str)
            pred_answers = parse_answers(response, num_q)
        except Exception as e:
            print(f"    API error at row {i}: {e}")
            pred_answers = ["?"] * num_q
            time.sleep(5)

        q_list = [q.strip() for q in questions_str.split("|")]
        for q, gt, pred in zip(q_list, gt_answers, pred_answers):
            is_correct = (gt.strip() == pred.strip())
            correct += int(is_correct)
            total_q += 1
            results.append({"group_id": row.get("group_id", ""), "question": q, "ground_truth": gt,
                             "prediction": pred, "correct": int(is_correct)})
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(rows)} | running acc: {correct/total_q:.4f}")
        time.sleep(delay)

    accuracy = correct / total_q if total_q else 0.0
    print(f"    Final accuracy: {accuracy:.4f} ({correct}/{total_q})")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "question", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_files", nargs="+", default=["data/test_base.csv", "data/test_variant2.csv", "data/test_variant3.csv"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = args.output_dir or f"results/fewshot/{args.model}"
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
    print(f"\n  === {args.model} (few-shot verification-first) Results ===")
    for split, acc in summary.items():
        print(f"    {split}: {acc:.4f}")
    print("  FINISHED.")


if __name__ == "__main__":
    main()
