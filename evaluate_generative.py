"""
Evaluation script for generative models (Qwen/LLaMA)
Evaluate models trained to generate True/False answers.
"""

import argparse
import os
import csv

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_LIST = {
    "qwen": "Qwen/Qwen2-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# All test splits
DEFAULT_TEST_FILES = {
    "base": "data/test_base.csv",
    "variant1": "data/test_variant1.csv",
    "variant2": "data/test_variant2.csv",
    "variant3": "data/test_variant3.csv",
    "variant4_equiv_contrapositive": "data/test_variant4_equiv_contrapositive.csv",
    "variant4_equiv_double_negation": "data/test_variant4_equiv_double_negation.csv",
    "variant4_equiv_implication": "data/test_variant4_equiv_implication.csv",
    "variant4_equiv_demorgan": "data/test_variant4_equiv_demorgan.csv",
    "variant4_equiv_identity": "data/test_variant4_equiv_identity.csv",
    "variant4_equiv_commutativity": "data/test_variant4_equiv_commutativity.csv",
    "variant4_equiv_multi": "data/test_variant4_equiv_multi.csv",
}


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_question_prompt(facts, rules, question):
    """
    Format a question into the prompt format used during training.
    """
    prompt = f"""Given the following information:

Facts: {facts}

Rules:
{chr(10).join(f"- {rule}" for rule in rules.split(' | '))}

Question: {question}

Based on the facts and rules above, is this statement true or false?

Answer:"""
    return prompt


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=256):
    """
    Generate answer using the model.

    Returns:
        Generated text (potentially including reasoning and a final answer)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for deterministic results
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated tokens (excluding input)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def parse_answer(generated_text):
    """
    Parse generated text to extract T or F.

    Handles variations like "True", "true", "T", "False", "false", "F"
    looking specifically for the first or last occurrences if reasoning is present.
    """
    # Clean up the text
    text = generated_text.strip()
    
    # Check if text starts with True/False (our intended format)
    text_lower = text.lower()
    
    # Priority 1: Check start of string for direct answer
    if text_lower.startswith("true"):
        return "T"
    if text_lower.startswith("false"):
        return "F"
    
    # Priority 2: Look for 'true' or 'false' words explicitly (handling reasoning)
    # Often models end with "Therefore the answer is True." or similar.
    # We'll check for the word 'false' first because it's the more common "correct" answer in tricky variants.
    if "false" in text_lower:
        return "F"
    if "true" in text_lower:
        # Check if "true" is used as an answer or just in rules discussion
        # Models often say "Step 1: If someone is blue then they are true..." (Wait, rules use 'cold')
        # Let's be smarter: split by lines and look for "Answer:" or "Therefore"
        lines = text.split('\n')
        for line in reversed(lines):
            line_l = line.lower()
            if "answer is true" in line_l or "statement is true" in line_l or (line_l.strip().startswith("true")):
                return "T"
            if "answer is false" in line_l or "statement is false" in line_l or (line_l.strip().startswith("false")):
                return "F"
                
    # Fallback to general containment check if still unsure
    if "true" in text_lower: return "T"
    if "false" in text_lower: return "F"

    return "F"


def describe_change(split_name: str, laws_used: str, law_count: int) -> str:
    """Human-readable description of what changed for this split."""
    if split_name == "base":
        return "none"
    if split_name == "variant1":
        return "removed redundant rule: 'If someone is young then they are cold.'"
    if split_name == "variant2":
        return "removed key rule: 'If someone is cold then they are rough.'"
    if split_name == "variant3":
        return "changed facts: added '<name> is not cold or not nice'"

    if split_name.startswith("variant4_equiv_"):
        if split_name == "variant4_equiv_multi":
            return f"multiple logical equivalence laws applied (count={law_count}): {laws_used}"
        base = split_name.replace("variant4_equiv_", "")
        return f"logical equivalence law applied: {base}"

    return "unknown"


def eval_and_save(model, tokenizer, filename, model_key, split_name, device, out_dir):
    """
    Evaluate on one CSV file AND save predictions.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test file not found: {filename}")

    ds = load_dataset("csv", data_files=filename)["train"]

    total, correct = 0, 0
    output_rows = []

    os.makedirs(out_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, f"{model_key}_{split_name}_predictions.csv")

    for row in ds:
        facts = row["facts"]
        rules = row["rules"]
        questions = row["questions"].split(" | ")
        answers = row["answers"].split(" | ")
        laws_used = row.get("equiv_laws_used", "") or ""
        law_list = [x for x in laws_used.split(",") if x]
        law_count = len(law_list)

        changed_desc = describe_change(split_name, laws_used, law_count)

        for q, truth in zip(questions, answers):
            # Format prompt
            prompt = format_question_prompt(facts, rules, q)

            # Generate answer
            generated = generate_answer(model, tokenizer, prompt, device)

            # Parse to T/F
            pred = parse_answer(generated)

            output_rows.append({
                "group_id": row["group_id"],
                "type": split_name,
                "facts": facts,
                "rules": rules,
                "question": q,
                "ground_truth": truth,
                "generated_text": generated,
                "prediction": pred,
                "equiv_laws_used": laws_used,
                "equiv_law_count": law_count,
                "changed_rule": changed_desc,
            })

            if pred == truth:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0

    # Save prediction CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "type",
                "facts",
                "rules",
                "question",
                "ground_truth",
                "generated_text",
                "prediction",
                "equiv_laws_used",
                "equiv_law_count",
                "changed_rule",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"📄 Predictions saved to: {output_csv}")

    return acc, total, correct


def main(model_key: str, model_dir: str = None):
    # Allow custom model directory or use default
    if model_dir is None:
        model_dir = f"./trained_models/{model_key}_stage2_mixed"

    print(f"=" * 80)
    print(f"Generative Model Evaluation")
    print(f"=" * 80)
    print(f"▶ Loading model from: {model_dir}")
    print(f"▶ Model type: {model_key}")

    device = build_device()
    print(f"▶ Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    predictions_dir = os.path.join(model_dir, "predictions")
    results = []

    print("\n===== Detailed Evaluation Per Split =====")

    for split_name, filename in DEFAULT_TEST_FILES.items():
        output_csv = os.path.join(predictions_dir, f"{model_key}_{split_name}_predictions.csv")
        
        # Load ground truth dataset
        ds = load_dataset("csv", data_files=filename)["train"]
        
        # Calculate total EXPECTED predictions (sum of questions in each row)
        expected_total = 0
        for row in ds:
             questions = row["questions"].split(" | ")
             expected_total += len(questions)
        
        # Check if predictions already exist and are complete
        if os.path.exists(output_csv):
            print(f"\n[{split_name}] Found existing predictions: {output_csv}")
            try:
                with open(output_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if len(rows) == expected_total:
                    print(f"  ✅ File is complete ({len(rows)} predictions). Calculating accuracy from existing file...")
                    correct = sum(1 for r in rows if r["prediction"] == r["ground_truth"])
                    acc = correct / expected_total if expected_total > 0 else 0.0
                    results.append({
                        "split": split_name,
                        "accuracy": acc,
                        "correct": correct,
                        "total": expected_total
                    })
                    print(f"  accuracy: {acc:.4f}")
                    continue
                else:
                    print(f"  ⚠️ File incomplete ({len(rows)}/{expected_total}). Re-running evaluation...")
            except Exception as e:
                print(f"  ⚠️ Error reading file: {e}. Re-running evaluation...")
        
        print(f"\n[{split_name}] Evaluating...")
        acc, total, correct = eval_and_save(
            model,
            tokenizer,
            filename,
            model_key,
            split_name,
            device,
            predictions_dir,
        )
        results.append({
            "split": split_name,
            "accuracy": acc,
            "correct": correct,
            "total": total
        })
        print(f"  samples (questions): {total}")
        print(f"  correct: {correct}")
        print(f"  accuracy: {acc:.4f}")
        print("-" * 40)

    # Summary table
    print("\n===== Base vs Variants Accuracy Table =====")
    base_acc = next((r["accuracy"] for r in results if r["split"] == "base"), 0.0)

    header = f"{'Split':<35} | {'Accuracy':>9} | {'Δ vs base':>9}"
    print(header)
    print("-" * len(header))
    
    summary_csv_path = os.path.join(model_dir, "accuracy_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "accuracy", "delta_vs_base", "correct", "total"])
        writer.writeheader()

        ordered_splits = [
            "base",
            "variant1",
            "variant2",
            "variant3",
            "variant4_equiv_contrapositive",
            "variant4_equiv_double_negation",
            "variant4_equiv_implication",
            "variant4_equiv_demorgan",
            "variant4_equiv_identity",
            "variant4_equiv_commutativity",
            "variant4_equiv_multi",
        ]

        for split in ordered_splits:
            res = next((r for r in results if r["split"] == split), None)
            if not res:
                continue
                
            acc = res["accuracy"]
            delta = acc - base_acc
            delta_str = f"{delta:+.3f}" if split != "base" else "0.000"
            print(f"{split:<35} | {acc:>9.4f} | {delta_str:>9}")
            
            writer.writerow({
                "split": split,
                "accuracy": acc,
                "delta_vs_base": delta,
                "correct": res["correct"],
                "total": res["total"]
            })
            
    print(f"\n📄 Accuracy summary saved to: {summary_csv_path}")

    print("\n✅ Evaluation FINISHED.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "llama"],
                        help="Model type (qwen or llama)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Custom model directory")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["stage1_gen", "stage2", "stage2_mixed"],
                        help="Shortcut to evaluate stage models")
    args = parser.parse_args()

    # Handle stage shortcuts
    if args.stage:
        model_dir = f"./trained_models/{args.model}_{args.stage}"
    else:
        model_dir = args.model_dir

    main(args.model, model_dir=model_dir)
