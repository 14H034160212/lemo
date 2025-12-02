# data_gen.py
import random
import csv
import uuid

random.seed(42)

NAMES = ["Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen"]
COLORS = ["green", "blue", "red", "yellow"]

# All logical equivalence law names
EQ_LAWS = [
    "contrapositive",
    "double_negation",
    "implication",
    "demorgan",
    "identity",
    "commutativity",
]


def rule(p, q):
    return f"If someone is {p} then they are {q}."


# =========================================================
# LOGICAL EQUIVALENCE TRANSFORMATIONS
# =========================================================

def contraposition(p, q):
    return f"If someone is not {q} then they are not {p}."

def double_negation(p, q):
    return f"If someone is not not {p} then they are not not {q}."

def implication_law(p, q):
    # P -> Q  ≡  not P or Q
    return f"Someone is not {p} or they are {q}."

def identity_law(p, q):
    # P -> Q  ≡  not not P -> Q
    return f"If someone is not not {p} then they are {q}."

def commutativity_or(p, q):
    # P or Q  ≡  Q or P
    return f"If someone is {q} or {p} then they are cold."

def demorgan_law(p, q):
    # not (P or Q)  ≡ not P and not Q
    return f"If someone is not {p} and not {q} then they are not cold."


# =========================================================
# BASE GENERATION
# =========================================================

def generate_base(name):
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])

    facts = [f"{name} is {color1} or {color2}"]

    rules = [
        rule(color1, "cold"),
        rule(color2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]

    answers = ["T", "T", "T", "T"]
    return facts, rules, questions, answers, (color1, color2)


# =========================================================
# ORIGINAL VARIANTS (1–3)
# =========================================================

def variant1(facts, rules, name):
    r = rules[:4] + rules[5:]
    return facts, r, [f"Q1: {name} is cold.",
                      f"Q2: {name} is rough.",
                      f"Q3: {name} is young.",
                      f"Q4: {name} is nice."], ["T","T","T","T"]

def variant2(facts, rules, name):
    r = rules.copy()
    r.remove(rules[2])  # remove cold -> rough
    return facts, r, [f"Q1: {name} is cold.",
                      f"Q2: {name} is rough.",
                      f"Q3: {name} is young.",
                      f"Q4: {name} is nice."], ["T","F","F","F"]

def variant3(facts, rules, name):
    f = facts + [f"{name} is not cold or not nice"]
    return f, rules, [f"Q1: {name} is cold.",
                      f"Q2: {name} is rough.",
                      f"Q3: {name} is young.",
                      f"Q4: {name} is nice."], ["F","F","F","F"]


# =========================================================
# VARIANT 4 — SINGLE-LAW EQUIVALENTS
# =========================================================

def variant_equiv_single(facts, rules, color_pair):
    c1, c2 = color_pair

    eq_variants = {}

    eq_variants["contrapositive"] = [
        contraposition(c1, "cold"),
        rule(c2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    eq_variants["double_negation"] = [
        double_negation(c1, "cold"),
        rule(c2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    eq_variants["implication"] = [
        implication_law(c1, "cold"),
        rule(c2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    eq_variants["identity"] = [
        identity_law(c1, "cold"),
        rule(c2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    # Commutativity + DeMorgan use composite color conditions
    eq_variants["commutativity"] = [
        commutativity_or(c1, c2),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    eq_variants["demorgan"] = [
        demorgan_law(c1, c2),
        rule(c1, "cold"),
        rule(c2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]

    return eq_variants


# =========================================================
# VARIANT 4 — MULTI-LAW EQUIVALENTS (2–5 rules)
# =========================================================

def variant_equiv_multi(facts, rules, color_pair):
    """
    Start from base rules, then *add* 2–5 extra rules that are
    logical equivalents of the first rule (color1 -> cold) using
    different equivalence laws.
    This preserves semantics but increases rule redundancy.
    """
    c1, c2 = color_pair
    laws = EQ_LAWS.copy()
    random.shuffle(laws)
    k = random.randint(2, min(5, len(laws)))
    selected = laws[:k]

    new_rules = rules.copy()

    for law in selected:
        if law == "contrapositive":
            new_rules.append(contraposition(c1, "cold"))
        elif law == "double_negation":
            new_rules.append(double_negation(c1, "cold"))
        elif law == "implication":
            new_rules.append(implication_law(c1, "cold"))
        elif law == "identity":
            new_rules.append(identity_law(c1, "cold"))
        elif law == "commutativity":
            new_rules.append(commutativity_or(c1, c2))
        elif law == "demorgan":
            new_rules.append(demorgan_law(c1, c2))

    laws_used = ",".join(selected)
    return facts, new_rules, laws_used


# =========================================================
# GENERATE + SAVE
# =========================================================

train_rows = []
test_base_rows = []
variant1_rows = []
variant2_rows = []
variant3_rows = []
equiv_rows_single = {law: [] for law in EQ_LAWS}
equiv_rows_multi = []  # multi-law variant4

base_examples = []
NUM = 100

for _ in range(NUM):
    name = random.choice(NAMES)
    gid = str(uuid.uuid4())
    facts, rules, questions, answers, cp = generate_base(name)
    base_examples.append((gid, name, facts, rules, questions, answers, cp))

random.shuffle(base_examples)
split = int(0.8 * NUM)
train_part = base_examples[:split]
test_part = base_examples[split:]


def write_rows(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# common header with extra column for equivalence info
header = ["group_id","type","facts","rules","questions","answers","equiv_laws_used"]

# base train / test
for gid,name,facts,rules,q,a,cp in train_part:
    train_rows.append({
        "group_id": gid,
        "type": "base",
        "facts": " | ".join(facts),
        "rules": " | ".join(rules),
        "questions": " | ".join(q),
        "answers": " | ".join(a),
        "equiv_laws_used": "",
    })

for gid,name,facts,rules,q,a,cp in test_part:
    test_base_rows.append({
        "group_id": gid,
        "type": "base",
        "facts": " | ".join(facts),
        "rules": " | ".join(rules),
        "questions": " | ".join(q),
        "answers": " | ".join(a),
        "equiv_laws_used": "",
    })

# variants 1–3
for gid,name,facts,rules,q,a,cp in base_examples:
    f1, r1, q1, a1 = variant1(facts, rules, name)
    variant1_rows.append({
        "group_id": gid,
        "type": "variant1",
        "facts": " | ".join(f1),
        "rules": " | ".join(r1),
        "questions": " | ".join(q1),
        "answers": " | ".join(a1),
        "equiv_laws_used": "",
    })

    f2, r2, q2, a2 = variant2(facts, rules, name)
    variant2_rows.append({
        "group_id": gid,
        "type": "variant2",
        "facts": " | ".join(f2),
        "rules": " | ".join(r2),
        "questions": " | ".join(q2),
        "answers": " | ".join(a2),
        "equiv_laws_used": "",
    })

    f3, r3, q3, a3 = variant3(facts, rules, name)
    variant3_rows.append({
        "group_id": gid,
        "type": "variant3",
        "facts": " | ".join(f3),
        "rules": " | ".join(r3),
        "questions": " | ".join(q3),
        "answers": " | ".join(a3),
        "equiv_laws_used": "",
    })

# variant4 single-law files
for gid,name,facts,rules,q,a,cp in base_examples:
    eqs = variant_equiv_single(facts, rules, cp)
    for law, rlist in eqs.items():
        equiv_rows_single[law].append({
            "group_id": gid,
            "type": f"equiv_{law}",
            "facts": " | ".join(facts),
            "rules": " | ".join(rlist),
            "questions": " | ".join(q),
            "answers": " | ".join(a),
            "equiv_laws_used": law,
        })

# variant4 multi-law file
for gid,name,facts,rules,q,a,cp in base_examples:
    f_multi, r_multi, laws_used = variant_equiv_multi(facts, rules, cp)
    equiv_rows_multi.append({
        "group_id": gid,
        "type": "equiv_multi",
        "facts": " | ".join(f_multi),
        "rules": " | ".join(r_multi),
        "questions": " | ".join(q),
        "answers": " | ".join(a),
        "equiv_laws_used": laws_used,
    })

# write core files
write_rows("train.csv", train_rows, header)
write_rows("test_base.csv", test_base_rows, header)
write_rows("test_variant1.csv", variant1_rows, header)
write_rows("test_variant2.csv", variant2_rows, header)
write_rows("test_variant3.csv", variant3_rows, header)

# single-law variant4 files
for law, rows in equiv_rows_single.items():
    write_rows(f"test_variant4_equiv_{law}.csv", rows, header)

# multi-law variant4 file
write_rows("test_variant4_equiv_multi.csv", equiv_rows_multi, header)

print("✔ Data generation complete!")
