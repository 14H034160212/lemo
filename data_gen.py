# data_gen.py
import random
import csv
import uuid

random.seed(42)

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen"]
COLORS = ["green", "blue", "red", "yellow"]


def make_rule(premise, conclusion):
    return f"If someone is {premise} then they are {conclusion}."


def implication_equiv(rule):
    """
    Convert:
    If someone is X then they are Y.
    Into:
    If someone is not Y then they are not X.
    """
    s = rule.strip().removeprefix("If someone is ").removesuffix(".")
    p, _, q = s.partition(" then they are ")
    return f"If someone is not {q} then they are not {p}."


def generate_base(name):
    color = random.choice(COLORS)
    other_color = random.choice([c for c in COLORS if c != color])

    facts = [f"{name} is {color} or {other_color}"]

    rules = [
        make_rule(color, "cold"),
        make_rule(other_color, "cold"),
        make_rule("cold", "rough"),
        make_rule("not young", "not rough"),
        make_rule("young", "cold"),
        make_rule("young", "nice"),
    ]

    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice."
    ]

    answers = ["T", "T", "T", "T"]
    return facts, rules, questions, answers


# ========== Variants ==========

def variant1(facts, rules, name):
    # Drop redundant rule: "young -> cold"
    rules_v1 = rules[:4] + rules[5:]
    return facts, rules_v1, [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice."
    ], ["T", "T", "T", "T"]


def variant2(facts, rules, name):
    # Drop KEY rule: cold → rough
    rules_v2 = rules.copy()
    rules_v2.remove(rules[2])
    return facts, rules_v2, [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice."
    ], ["T", "F", "F", "F"]


def variant3(facts, rules, name):
    # Add contradictory fact
    facts_v3 = facts + [f"{name} is not cold or not nice"]
    return facts_v3, rules, [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice."
    ], ["F", "F", "F", "F"]


def variant4(facts, rules, name):
    # Apply logical equivalence (contrapositive)
    rules_v4 = rules.copy()
    rules_v4[0] = implication_equiv(rules[0])
    return facts, rules_v4, [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice."
    ], ["T", "T", "T", "T"]


# ========== Output Files ==========

train_file = "train.csv"
test_base_file = "test_base.csv"

test_files = {
    "variant1": "test_variant1.csv",
    "variant2": "test_variant2.csv",
    "variant3": "test_variant3.csv",
    "variant4": "test_variant4.csv"
}

# ========== Storage ==========

base_rows = []                     # collect base first, then split into train/test
variant_rows = {k: [] for k in test_files}

num_examples = 100


# ========== Generate Data ==========

for _ in range(num_examples):
    name = random.choice(NAMES)
    group_id = str(uuid.uuid4())

    facts, rules, questions, answers = generate_base(name)

    # Store base example
    base_rows.append({
        "group_id": group_id,
        "type": "base",
        "facts": " | ".join(facts),
        "rules": " | ".join(rules),
        "questions": " | ".join(questions),
        "answers": " | ".join(answers)
    })

    # Generate variants for test
    variants = {
        "variant1": variant1(facts, rules, name),
        "variant2": variant2(facts, rules, name),
        "variant3": variant3(facts, rules, name),
        "variant4": variant4(facts, rules, name),
    }

    for vname, (f, r, q, a) in variants.items():
        variant_rows[vname].append({
            "group_id": group_id,
            "type": vname,
            "facts": " | ".join(f),
            "rules": " | ".join(r),
            "questions": " | ".join(q),
            "answers": " | ".join(a)
        })


# ========== Split base into train/test ==========

random.shuffle(base_rows)
split = int(len(base_rows) * 0.8)

train_rows = base_rows[:split]
test_base_rows = base_rows[split:]

# Write train.csv
with open(train_file, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=train_rows[0].keys())
    w.writeheader()
    w.writerows(train_rows)

# Write test_base.csv
with open(test_base_file, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=test_base_rows[0].keys())
    w.writeheader()
    w.writerows(test_base_rows)

# Write variant files
for vname, fname in test_files.items():
    rows = variant_rows[vname]
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

print("✅ Generated:")
print(f"  train.csv        ({len(train_rows)} rows)")
print(f"  test_base.csv    ({len(test_base_rows)} rows)")
for v, fpath in test_files.items():
    print(f"  {fpath} ({len(variant_rows[v])} rows)")
