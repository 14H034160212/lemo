"""
Stage 1 Data Generation V2: Rule Prediction Task
Generate training data for masked rule prediction (similar to BERT MLM).

For generative models (Qwen/LLaMA):
- Input: facts + masked_rules + question → Output: missing_rule

For discriminative models (BERT):
- Input: facts + masked_rules + question + rule_candidates → Output: correct_rule_index
"""

import random
import csv
import uuid
import argparse
import pandas as pd

random.seed(42)

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen"]


def identify_critical_rules(rules):
    """
    Identify critical rules that can be removed.
    Returns list of (rule_index, rule_text, rule_type) tuples.
    """
    critical_rules = []

    # Rule index 2: "If someone is cold then they are rough." - most critical
    if len(rules) > 2 and "cold" in rules[2] and "rough" in rules[2]:
        critical_rules.append((2, rules[2], "cold_rough"))

    # Rule index 4: "If someone is young then they are cold." - redundant
    if len(rules) > 4 and "young" in rules[4] and "cold" in rules[4]:
        critical_rules.append((4, rules[4], "young_cold"))

    # Rule index 3: contrapositive
    if len(rules) > 3 and "not young" in rules[3] and "not rough" in rules[3]:
        critical_rules.append((3, rules[3], "contrapositive"))

    return critical_rules


def generate_rule_candidates(target_rule, all_rules, num_candidates=4):
    """
    Generate rule candidates for multiple choice (for BERT).

    Args:
        target_rule: The correct rule
        all_rules: All available rules from the sample
        num_candidates: Total number of candidates (including correct one)

    Returns:
        List of candidate rules, correct answer index
    """
    candidates = [target_rule]

    # Add other rules as distractors
    other_rules = [r for r in all_rules if r != target_rule]

    # Add some random variations
    distractors = []

    # Use existing rules as distractors
    distractors.extend(other_rules[:min(2, len(other_rules))])

    # Generate some wrong rules
    if "cold" in target_rule and "rough" in target_rule:
        distractors.append("If someone is cold then they are nice.")
        distractors.append("If someone is rough then they are cold.")
    elif "young" in target_rule and "cold" in target_rule:
        distractors.append("If someone is young then they are rough.")
        distractors.append("If someone is cold then they are young.")
    else:
        # Generic distractors for other rule types
        distractors.append("If someone is happy then they are sad.")
        distractors.append("If someone is tall then they are short.")

    # Ensure we have enough distractors
    while len(distractors) < num_candidates - 1:
        distractors.append(f"If someone is property{len(distractors)} then they are state{len(distractors)}.")

    # Limit to num_candidates - 1
    distractors = distractors[:num_candidates - 1]

    # Combine and shuffle
    candidates.extend(distractors)

    # Shuffle
    indices = list(range(len(candidates)))
    random.shuffle(indices)
    shuffled_candidates = [candidates[i] for i in indices]

    # Find correct answer index
    correct_idx = indices.index(0)

    return shuffled_candidates, correct_idx


def format_for_generative(facts, masked_rules, question, missing_rule):
    """
    Format data for generative models (Qwen/LLaMA).

    Returns:
        input_text: Prompt for the model
        target_text: Expected output (the missing rule)
    """
    # Create a natural language prompt
    input_text = f"""Given the following information:

Facts: {facts}

Rules:
{chr(10).join(f"- {rule}" for rule in masked_rules.split(' | '))}

Question: {question}

One critical rule is missing from the rules above. Based on the facts and question, what is the missing rule?

Missing rule:"""

    target_text = missing_rule

    return input_text, target_text


def format_for_bert(facts, masked_rules, question, candidates, correct_idx):
    """
    Format data for BERT (multiple choice).

    Returns:
        input_text: Context
        candidate_texts: List of candidate rules
        correct_idx: Index of correct rule
    """
    # Context
    context = f"Facts: {facts} Rules: {masked_rules} Question: {question}"

    return context, candidates, correct_idx


def generate_stage1_samples(base_sample, name, format_type="generative"):
    """
    Generate stage1 samples for rule prediction.

    Args:
        base_sample: Original training sample
        name: Person name
        format_type: "generative" or "bert"

    Returns:
        List of training samples
    """
    facts = base_sample['facts']
    rules = base_sample['rules'].split(' | ')
    questions = base_sample['questions'].split(' | ')

    critical_rules = identify_critical_rules(rules)

    samples = []

    for rule_idx, rule_text, rule_type in critical_rules:
        # Create masked rules (remove the critical rule)
        masked_rules = [r for i, r in enumerate(rules) if i != rule_idx]
        masked_rules_text = ' | '.join(masked_rules)

        # Use first question for simplicity (can use all questions)
        question = questions[0] if questions else "Q1: Unknown"

        if format_type == "generative":
            # Format for generative models
            input_text, target_text = format_for_generative(
                facts, masked_rules_text, question, rule_text
            )

            samples.append({
                'group_id': str(uuid.uuid4()),
                'type': f'stage1_generative_{rule_type}',
                'input_text': input_text,
                'target_text': target_text,
                'facts': facts,
                'masked_rules': masked_rules_text,
                'question': question,
                'missing_rule': rule_text,
            })

        elif format_type == "bert":
            # Format for BERT (multiple choice)
            candidates, correct_idx = generate_rule_candidates(rule_text, rules)
            context, cand_list, correct = format_for_bert(
                facts, masked_rules_text, question, candidates, correct_idx
            )

            samples.append({
                'group_id': str(uuid.uuid4()),
                'type': f'stage1_bert_{rule_type}',
                'context': context,
                'candidate_0': cand_list[0] if len(cand_list) > 0 else "",
                'candidate_1': cand_list[1] if len(cand_list) > 1 else "",
                'candidate_2': cand_list[2] if len(cand_list) > 2 else "",
                'candidate_3': cand_list[3] if len(cand_list) > 3 else "",
                'correct_answer': correct,
                'facts': facts,
                'masked_rules': masked_rules_text,
                'question': question,
                'missing_rule': rule_text,
            })

    return samples


def write_rows(path, rows, header):
    """Write rows to CSV file"""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def generate_stage1_data_v2(
    train_csv_path="data/train.csv",
    output_prefix="data/stage1_train",
    num_samples=200,
    format_type="generative"
):
    """
    Generate Stage 1 training data for rule prediction.

    Args:
        train_csv_path: Path to original training data
        output_prefix: Prefix for output files
        num_samples: Number of samples to generate
        format_type: "generative" (for Qwen/LLaMA) or "bert" (for BERT)
    """
    print(f"=" * 80)
    print(f"Stage 1 Data Generation V2 - Rule Prediction")
    print(f"=" * 80)
    print(f"Format type: {format_type}")
    print(f"Loading training data from: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    # Filter only base_positive samples
    base_samples = df[df['type'] == 'base_positive'].copy()
    print(f"Found {len(base_samples)} base positive samples")

    if len(base_samples) == 0:
        print("Warning: No base_positive samples found. Using all samples.")
        base_samples = df.copy()

    # Generate samples
    all_samples = []

    # Sample from base data
    sample_indices = list(range(len(base_samples)))
    if len(sample_indices) > num_samples:
        sample_indices = random.sample(sample_indices, num_samples)

    for idx in sample_indices:
        sample = base_samples.iloc[idx]

        # Extract name from facts
        facts_text = sample['facts']
        name = facts_text.split()[0] if facts_text else random.choice(NAMES)

        # Generate samples
        samples = generate_stage1_samples(sample, name, format_type=format_type)
        all_samples.extend(samples)

    print(f"\nGenerated {len(all_samples)} samples")

    # Define headers based on format type
    if format_type == "generative":
        header = ["group_id", "type", "input_text", "target_text",
                  "facts", "masked_rules", "question", "missing_rule"]
    else:  # bert
        header = ["group_id", "type", "context",
                  "candidate_0", "candidate_1", "candidate_2", "candidate_3",
                  "correct_answer", "facts", "masked_rules", "question", "missing_rule"]

    # Save data
    output_path = f"{output_prefix}_{format_type}.csv"
    write_rows(output_path, all_samples, header)
    print(f"Saved to: {output_path}")

    # Print sample
    print("\n" + "="*80)
    print("Sample:")
    print("="*80)
    if all_samples:
        sample = all_samples[0]
        if format_type == "generative":
            print(f"Input:\n{sample['input_text']}\n")
            print(f"Target: {sample['target_text']}")
        else:
            print(f"Context: {sample['context']}\n")
            print("Candidates:")
            for i in range(4):
                cand = sample.get(f'candidate_{i}', '')
                if cand:
                    marker = " ← CORRECT" if i == sample['correct_answer'] else ""
                    print(f"  {i}. {cand}{marker}")

    print("\n✅ Stage 1 data generation V2 complete!")
    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 training data V2")
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to original training CSV")
    parser.add_argument("--output_prefix", type=str, default="data/stage1_train",
                        help="Prefix for output files")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of base samples to use")
    parser.add_argument("--format", type=str, default="generative",
                        choices=["generative", "bert"],
                        help="Data format: 'generative' for Qwen/LLaMA, 'bert' for BERT")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Generate data
    generate_stage1_data_v2(
        train_csv_path=args.train_csv,
        output_prefix=args.output_prefix,
        num_samples=args.num_samples,
        format_type=args.format
    )
