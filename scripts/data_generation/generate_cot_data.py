
import csv
import json
import random

INPUT_FILE = "data/train.csv"
OUTPUT_FILE = "data/train_cot.csv"

def generate_reasoning(row):
    row_type = row.get('type', 'base_positive')
    facts = row['facts']
    rules = row['rules'].split(' | ')
    questions = row['questions'].split(' | ')
    answers = row['answers'].split(' | ')

    name = facts.split()[0]

    samples = []

    for q, a in zip(questions, answers):
        parts = q.strip(".").split()
        is_negated = "not" in parts
        target_attr = parts[-1]

        final_answer = "True" if a.strip() == "T" else "False"

        if row_type in ('base_positive',):
            # Full chain: color → cold → rough → young → nice
            chain = []
            chain.append(f"Fact: {facts}.")
            chain.append(f"By the color rules, {name} is cold.")
            chain.append(f"Since cold → rough, {name} is rough.")
            chain.append(f"Since rough → young (contrapositive of not-young → not-rough), {name} is young.")
            chain.append(f"Since young → nice, {name} is nice.")
            chain.append(f"Therefore '{q}' is {final_answer}.")
            reasoning = " ".join(chain)

        elif row_type == 'base_negative':
            chain = []
            chain.append(f"Fact: {facts}.")
            chain.append(f"By the color rules, {name} is cold.")
            chain.append(f"Since cold → rough, {name} is rough.")
            chain.append(f"Since rough → young, {name} is young.")
            chain.append(f"Since young → nice, {name} is nice.")
            if "not cold" in q:
                chain.append(f"But we derived cold=True, so '{q}' is False.")
            elif "not rough" in q:
                chain.append(f"But we derived rough=True, so '{q}' is False.")
            else:
                chain.append(f"The property in '{q}' is not derivable from given facts and rules, so it is False.")
            chain.append(f"Therefore '{q}' is {final_answer}.")
            reasoning = " ".join(chain)

        elif row_type in ('variant2',):
            # One critical rule is missing; determine which and reason accordingly
            rules_list = row['rules'].split(' | ')
            has_cold_rough = any("cold" in r and "rough" in r and "not" not in r for r in rules_list)
            has_rough_young = any("not young" in r and "not rough" in r for r in rules_list)
            has_young_nice = any("young" in r and "nice" in r and "not" not in r for r in rules_list)
            chain = []
            chain.append(f"Fact: {facts}.")
            chain.append(f"By the color rules, {name} is cold. cold=True.")
            if not has_cold_rough:
                chain.append(f"The rule 'cold → rough' is absent. rough cannot be derived. rough=False.")
                chain.append(f"Without rough, young cannot be derived. young=False. nice=False.")
            elif not has_rough_young:
                chain.append(f"cold → rough applies: rough=True.")
                chain.append(f"The rule 'not young → not rough' is absent, so rough → young fails. young=False. nice=False.")
            else:
                chain.append(f"cold → rough: rough=True. rough → young (via contrapositive): young=True.")
                chain.append(f"The rule 'young → nice' is absent. nice=False.")
            chain.append(f"Therefore '{q}' is {final_answer}.")
            reasoning = " ".join(chain)

        elif row_type in ('variant3',):
            # A contradicting fact was added; determine which one and reason accordingly
            facts_str = facts
            chain = []
            chain.append(f"Fact: {facts_str}.")
            if "not cold" in facts_str and "not nice" not in facts_str:
                chain.append(f"An explicit fact states {name} is not cold. cold=False.")
                chain.append(f"Without cold, rough cannot be derived. rough=False. young=False. nice=False.")
            elif "not rough" in facts_str:
                chain.append(f"An explicit fact states {name} is not rough. rough=False.")
                chain.append(f"Without rough, young cannot be derived. young=False. nice=False.")
                chain.append(f"Note: cold is still derivable from color rules. cold=True.")
            elif "not nice" in facts_str:
                chain.append(f"An explicit fact states {name} is not nice. nice=False.")
                chain.append(f"The chain still holds: cold=True, rough=True, young=True; only nice is directly negated.")
            else:
                chain.append(f"A contradicting fact overrides the normal chain.")
            chain.append(f"Therefore '{q}' is {final_answer}.")
            reasoning = " ".join(chain)

        else:
            # hard_mixed or other types: simple answer based on GT
            chain = []
            chain.append(f"Fact: {facts}. Rules: {row['rules']}.")
            chain.append(f"After applying the rules carefully, the answer for '{q}' is {final_answer}.")
            reasoning = " ".join(chain)

        full_output = f"Reasoning: {reasoning} Answer: {final_answer}"

        samples.append({
            "input_text": f"Facts: {facts}\nRules: {row['rules']}\nQuestion: {q}\nThink step by step.",
            "target_text": full_output
        })

    return samples


def generate_cot_data():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cot_samples = []
    type_counts = {}
    for row in rows:
        t = row.get('type', 'unknown')
        samples = generate_reasoning(row)
        cot_samples.extend(samples)
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Generated {len(cot_samples)} CoT samples from {len(rows)} rows.")
    print("  Types included:", dict(sorted(type_counts.items())))

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
        writer.writeheader()
        writer.writerows(cot_samples)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_cot_data()
