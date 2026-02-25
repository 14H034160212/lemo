"""
Stage 1 Data Generation: Incomplete Rules Training Data
Generate training data where critical rules are removed, teaching the model to reason with incomplete information.

This creates variant2-style training data:
- Remove critical rules that break the inference chain
- Update answers accordingly (some T -> F due to missing rules)
"""

import random
import csv
import uuid
import argparse

random.seed(42)

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen"]


def identify_critical_rules(rules):
    """
    Identify critical rules that can be removed to create variant2-style samples.
    Returns list of (rule_index, rule_text) tuples.
    """
    critical_rules = []

    # Rule index 2: "If someone is cold then they are rough." - most critical
    if len(rules) > 2 and "cold" in rules[2] and "rough" in rules[2]:
        critical_rules.append((2, rules[2], "cold_rough"))

    # Rule index 4: "If someone is young then they are cold." - redundant with color->cold
    if len(rules) > 4 and "young" in rules[4] and "cold" in rules[4]:
        critical_rules.append((4, rules[4], "young_cold"))

    # Rule index 3: "If someone is not young then they are not rough."
    if len(rules) > 3 and "not young" in rules[3] and "not rough" in rules[3]:
        critical_rules.append((3, rules[3], "contrapositive"))

    return critical_rules


def compute_answers_after_removal(rule_type, name):
    """
    Compute the answers after removing a specific rule.
    Based on variant logic from data_gen.py
    """
    if rule_type == "cold_rough":
        # Removing "cold -> rough" breaks Q2, Q3, Q4
        return ["T", "F", "F", "F"]
    elif rule_type == "young_cold":
        # Removing "young -> cold" (redundant rule) - all still work
        return ["T", "T", "T", "T"]
    elif rule_type == "contrapositive":
        # Removing contrapositive rule might affect reasoning
        return ["T", "T", "F", "T"]
    else:
        # Default: assume all fail
        return ["F", "F", "F", "F"]


def generate_stage1_variant2_data(base_sample, name):
    """
    Generate variant2-style data by removing critical rules.
    Similar to variant2() in data_gen.py
    """
    facts = base_sample['facts']
    rules = base_sample['rules'].split(' | ')
    questions = base_sample['questions'].split(' | ')

    critical_rules = identify_critical_rules(rules)

    variant2_samples = []

    for rule_idx, rule_text, rule_type in critical_rules:
        # Create copy and remove the critical rule
        masked_rules = [r for i, r in enumerate(rules) if i != rule_idx]

        # Compute new answers based on removed rule
        new_answers = compute_answers_after_removal(rule_type, name)

        variant2_samples.append({
            'group_id': str(uuid.uuid4()),
            'type': f'stage1_variant2_{rule_type}',
            'facts': facts,
            'rules': ' | '.join(masked_rules),
            'questions': ' | '.join(questions),
            'answers': ' | '.join(new_answers),
            'equiv_laws_used': '',
            'removed_rule': rule_text,
            'removed_rule_type': rule_type,
        })

    return variant2_samples


def generate_stage1_variant3_data(base_sample, name):
    """
    Generate variant3-style data by adding contradictory facts.
    Similar to variant3() in data_gen.py
    """
    facts = base_sample['facts']
    rules = base_sample['rules']
    questions = base_sample['questions'].split(' | ')

    # Add contradictory fact
    contradictory_fact = f"{name} is not cold or not nice"
    augmented_facts = f"{facts} | {contradictory_fact}"

    # All answers become F due to contradiction
    new_answers = ["F", "F", "F", "F"]

    return [{
        'group_id': str(uuid.uuid4()),
        'type': 'stage1_variant3',
        'facts': augmented_facts,
        'rules': rules,
        'questions': ' | '.join(questions),
        'answers': ' | '.join(new_answers),
        'equiv_laws_used': '',
        'added_contradiction': contradictory_fact,
    }]


def write_rows(path, rows, header):
    """Write rows to CSV file"""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def generate_stage1_data(train_csv_path="data/train.csv",
                        output_prefix="data/stage1_train",
                        num_samples=200):
    """
    Generate Stage 1 training data from base training data.
    Creates variant2 and variant3 style samples for training.
    """
    import pandas as pd

    print(f"Loading training data from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)

    # Filter only base_positive samples
    base_samples = df[df['type'] == 'base_positive'].copy()
    print(f"Found {len(base_samples)} base positive samples")

    if len(base_samples) == 0:
        print("Warning: No base_positive samples found. Using all samples.")
        base_samples = df.copy()

    # Generate variant2 and variant3 samples
    all_variant2_samples = []
    all_variant3_samples = []

    # Sample from base data
    sample_indices = list(range(len(base_samples)))
    if len(sample_indices) > num_samples:
        sample_indices = random.sample(sample_indices, num_samples)

    for idx in sample_indices:
        sample = base_samples.iloc[idx]

        # Extract name from facts
        facts_text = sample['facts']
        name = facts_text.split()[0] if facts_text else random.choice(NAMES)

        # Generate variant2 samples (multiple per base sample, one for each critical rule)
        variant2_samples = generate_stage1_variant2_data(sample, name)
        all_variant2_samples.extend(variant2_samples)

        # Generate variant3 samples
        variant3_samples = generate_stage1_variant3_data(sample, name)
        all_variant3_samples.extend(variant3_samples)

    print(f"\nGenerated {len(all_variant2_samples)} variant2 samples")
    print(f"Generated {len(all_variant3_samples)} variant3 samples")

    # Define header (compatible with train.csv format)
    header = ["group_id", "type", "facts", "rules", "questions", "answers", "equiv_laws_used"]

    # Save variant2 samples
    variant2_path = f"{output_prefix}_variant2.csv"
    # Remove extra columns for compatibility
    variant2_clean = [{k: v for k, v in row.items() if k in header} for row in all_variant2_samples]
    write_rows(variant2_path, variant2_clean, header)
    print(f"Saved variant2 data to: {variant2_path}")

    # Save variant3 samples
    variant3_path = f"{output_prefix}_variant3.csv"
    variant3_clean = [{k: v for k, v in row.items() if k in header} for row in all_variant3_samples]
    write_rows(variant3_path, variant3_clean, header)
    print(f"Saved variant3 data to: {variant3_path}")

    # Save combined dataset (variant2 + variant3 mixed)
    combined_samples = variant2_clean + variant3_clean
    random.shuffle(combined_samples)
    combined_path = f"{output_prefix}_combined.csv"
    write_rows(combined_path, combined_samples, header)
    print(f"Saved combined data to: {combined_path}")

    # Print sample
    print("\n" + "="*80)
    print("Sample from Variant 2 (Incomplete Rules):")
    print("="*80)
    if variant2_clean:
        sample = variant2_clean[0]
        print(f"Facts: {sample['facts']}")
        print(f"Rules: {sample['rules']}")
        print(f"Questions: {sample['questions']}")
        print(f"Answers: {sample['answers']}")

    print("\n" + "="*80)
    print("Sample from Variant 3 (Contradictory Facts):")
    print("="*80)
    if variant3_clean:
        sample = variant3_clean[0]
        print(f"Facts: {sample['facts']}")
        print(f"Rules: {sample['rules']}")
        print(f"Questions: {sample['questions']}")
        print(f"Answers: {sample['answers']}")

    print("\n✅ Stage 1 data generation complete!")
    return variant2_clean, variant3_clean, combined_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 training data")
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to original training CSV")
    parser.add_argument("--output_prefix", type=str, default="data/stage1_train",
                        help="Prefix for output files")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of base samples to use for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Generate data
    generate_stage1_data(
        train_csv_path=args.train_csv,
        output_prefix=args.output_prefix,
        num_samples=args.num_samples
    )
