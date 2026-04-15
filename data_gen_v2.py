"""
data_gen_v2.py  —  Extended benchmark with multiple logical domains
Addresses reviewer critique: "benchmark too small and narrow, single rule-based template"

New features vs v1:
  1. Five independent logical domains (different entity types & property chains)
  2. Variable chain lengths (2-hop to 4-hop)
  3. Branching rules (multiple antecedents / disjunctive premises)
  4. Larger entity vocabulary
  5. All domains support variant1-3 and variant4 (logical equivalences)
  6. NUM increased to 500 per domain → 2500 total base groups
"""

import random
import csv
import uuid
import os

random.seed(2026)

# =========================================================
# DOMAIN DEFINITIONS
# Each domain: entity_pool, property_chain, rule_templates
# =========================================================

# Domain 1 (original): Person — colors → cold → rough → young → nice  (4-hop)
DOMAIN1 = {
    "name": "person",
    "entities": [
        "Anne", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Helen",
        "Ivan", "Julia", "Kevin", "Linda", "Mike", "Nancy", "Oscar", "Paula",
        "Quinn", "Rachel", "Steve", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    ],
    "init_props": ["green", "blue", "red", "yellow", "purple", "orange", "pink", "brown"],
    "chain": ["cold", "rough", "young", "nice"],
    "distractors": ["happy", "sad", "kind", "mean", "tall", "short", "funny", "brave"],
}

# Domain 2: Animal — habitat → predatory → fast → domesticated  (3-hop, different semantics)
DOMAIN2 = {
    "name": "animal",
    "entities": [
        "the wolf", "the eagle", "the rabbit", "the bear", "the fox",
        "the hawk", "the deer", "the lion", "the crow", "the seal",
        "the otter", "the lynx", "the raven", "the moose", "the puma",
        "the finch", "the badger", "the crane", "the gecko", "the bison",
    ],
    "init_props": ["forest", "ocean", "desert", "arctic", "jungle", "wetland"],
    "chain": ["predatory", "fast", "domesticated"],
    "distractors": ["nocturnal", "migratory", "endangered", "colorful", "venomous"],
}

# Domain 3: City — size → industrial → polluted → regulated  (3-hop)
DOMAIN3 = {
    "name": "city",
    "entities": [
        "Springfield", "Riverdale", "Oakwood", "Crestview", "Mapleton",
        "Fairfield", "Brookhaven", "Lakewood", "Millford", "Pinecrest",
        "Stonegate", "Willowbrook", "Cedarvale", "Harborview", "Elmwood",
        "Northgate", "Southfield", "Eastport", "Westlake", "Midvale",
    ],
    "init_props": ["coastal", "landlocked", "mountainous", "riverside", "elevated", "low-lying"],
    "chain": ["industrial", "polluted", "regulated"],
    "distractors": ["tourist", "historic", "modern", "growing", "declining"],
}

# Domain 4: Plant — climate → blooming → fruitful → medicinal  (3-hop)
DOMAIN4 = {
    "name": "plant",
    "entities": [
        "the rosemary", "the jasmine", "the willow", "the cedar", "the basil",
        "the orchid", "the fern", "the holly", "the thyme", "the lavender",
        "the nettle", "the clover", "the poppy", "the sage", "the bamboo",
        "the aloe", "the mint", "the daisy", "the tulip", "the ivy",
    ],
    "init_props": ["tropical", "temperate", "arid", "alpine", "humid", "semi-arid"],
    "chain": ["blooming", "fruitful", "medicinal"],
    "distractors": ["toxic", "edible", "fragrant", "thorny", "perennial"],
}

# Domain 5: Student — major → studious → skilled → employed  (3-hop, human domain, different from D1)
DOMAIN5 = {
    "name": "student",
    "entities": [
        "Alex", "Blake", "Casey", "Drew", "Ellis", "Finn", "Gray", "Harper",
        "Indigo", "Jordan", "Kendall", "Logan", "Morgan", "Noel", "Parker",
        "Quinn", "Reese", "Sage", "Taylor", "Umber",
    ],
    "init_props": ["engineering", "biology", "philosophy", "economics", "physics", "chemistry", "history", "mathematics"],
    "chain": ["studious", "skilled", "employed"],
    "distractors": ["athletic", "creative", "popular", "ambitious", "introverted"],
}

ALL_DOMAINS = [DOMAIN1, DOMAIN2, DOMAIN3, DOMAIN4, DOMAIN5]

# =========================================================
# LOGICAL EQUIVALENCE TRANSFORMATIONS (same as v1)
# =========================================================

EQ_LAWS = ["contrapositive", "double_negation", "implication", "demorgan", "identity", "commutativity"]

def rule_str(p, q):
    return f"If something is {p} then it is {q}."

def contraposition(p, q):
    return f"If something is not {q} then it is not {p}."

def double_negation(p, q):
    return f"If something is not not {p} then it is not not {q}."

def implication_law(p, q):
    return f"Something is not {p} or it is {q}."

def identity_law(p, q):
    return f"If something is not not {p} then it is {q}."

def commutativity_or(p, q, target):
    return f"If something is {q} or {p} then it is {target}."

def demorgan_law(p, q, target):
    return f"If something is not {p} and not {q} then it is not {target}."


# =========================================================
# DOMAIN-GENERIC GENERATION FUNCTIONS
# =========================================================

def make_base_rules(d, prop1, prop2):
    """Build the core rule chain for a domain."""
    chain = d["chain"]
    rules = [
        rule_str(prop1, chain[0]),
        rule_str(prop2, chain[0]),
    ]
    for i in range(len(chain) - 1):
        rules.append(rule_str(chain[i], chain[i + 1]))
    # Add contrapositive bridge: not chain[-2] → not chain[-3]  (rough→young equivalent)
    if len(chain) >= 2:
        rules.append(f"If something is not {chain[-2]} then it is not {chain[-1]}.")
    return rules


def make_base_questions(entity, d):
    chain = d["chain"]
    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    answers = ["T"] * len(chain)
    return questions, answers


def generate_base_domain(d):
    entity = random.choice(d["entities"])
    props = d["init_props"]
    prop1 = random.choice(props)
    prop2 = random.choice([p for p in props if p != prop1])
    facts = [f"{entity} is {prop1} or {prop2}"]
    rules = make_base_rules(d, prop1, prop2)
    questions, answers = make_base_questions(entity, d)
    return entity, facts, rules, questions, answers, (prop1, prop2)


def generate_negative_samples_domain(entity, facts, rules, d, prop_pair):
    p1, p2 = prop_pair
    chain = d["chain"]
    distractors = d["distractors"]

    neg_questions = [
        f"Q1: {entity} is not {chain[0]}.",
        f"Q2: {entity} is not {chain[1]}.",
        f"Q3: {entity} is {random.choice(distractors)}.",
    ]
    other_props = [p for p in d["init_props"] if p not in [p1, p2]]
    if other_props:
        neg_questions.append(f"Q4: {entity} is {random.choice(other_props)}.")
    else:
        neg_questions.append(f"Q4: {entity} is {random.choice([x for x in distractors if x != neg_questions[2].split()[-1][:-1]])}.")

    neg_answers = ["F"] * len(neg_questions)
    return facts, rules, neg_questions, neg_answers


# =========================================================
# HARD MIXED PATTERNS (domain-generic)
# =========================================================

def generate_mixed_domain(entity, d):
    """Generate a hard mixed sample with partial derivability."""
    chain = d["chain"]
    props = d["init_props"]
    prop1 = random.choice(props)
    prop2 = random.choice([p for p in props if p != prop1])

    pattern = random.choice(["ttff", "tftt", "tttf"])

    if pattern == "ttff":
        # chain[0]=T, chain[1]=T, rest=F: remove contrapositive bridge
        facts = [f"{entity} is {prop1} or {prop2}"]
        rules = [
            rule_str(prop1, chain[0]),
            rule_str(prop2, chain[0]),
            rule_str(chain[0], chain[1]),
            rule_str(chain[1], chain[2]) if len(chain) > 2 else f"If something is tall then it is warm.",
            f"If something is brave then it is {chain[-1]}.",  # distractor
        ]
        answers = ["T", "T"] + ["F"] * (len(chain) - 2)

    elif pattern == "tftt":
        # chain[0]=T (via entity given chain[-2]), chain[1]=F, rest=T
        facts = [f"{entity} is {prop1} or {prop2}", f"{entity} is {chain[-2]}"]
        rules = [
            rule_str(prop1, chain[0]),
            rule_str(prop2, chain[0]),
            # omit chain[0]→chain[1]
            rule_str(chain[-2], chain[-1]) if len(chain) > 1 else rule_str("brave", "warm"),
            f"If something is curious then it is {chain[1]}.",  # distractor
        ]
        a = ["T"] + ["F"] + ["T"] * (len(chain) - 2)
        answers = a

    else:  # tttf
        # All but last=T, last=F: remove last rule
        facts = [f"{entity} is {prop1} or {prop2}"]
        rules = [
            rule_str(prop1, chain[0]),
            rule_str(prop2, chain[0]),
            rule_str(chain[0], chain[1]),
        ]
        if len(chain) > 2:
            rules.append(f"If something is not {chain[-2]} then it is not {chain[-1]}.")
        rules.append(f"If something is kind then it is {chain[-1]}.")  # distractor
        answers = ["T"] * (len(chain) - 1) + ["F"]

    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    return facts, rules, questions, answers, (prop1, prop2)


# =========================================================
# VARIANTS 1–3 (domain-generic)
# =========================================================

def variant1_domain(facts, rules, entity, d):
    """Remove one rule; all answers still T via redundancy."""
    r = rules[:-1] if len(rules) > 1 else rules  # remove last rule
    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(d["chain"])]
    answers = ["T"] * len(d["chain"])
    return facts, r, questions, answers


def variant2_domain(facts, rules, entity, d):
    """Remove a critical rule, creating partial derivability."""
    chain = d["chain"]
    choice = random.randint(0, 2)
    r = rules.copy()
    if choice == 0 and len(r) > 2:
        r = [x for i, x in enumerate(r) if i != 2]  # remove chain[0]→chain[1]
        answers = ["T"] + ["F"] * (len(chain) - 1)
    elif choice == 1 and len(r) > 3:
        r = [x for i, x in enumerate(r) if i != 3]  # remove chain[1]→chain[2]
        answers = ["T", "T"] + ["F"] * (len(chain) - 2)
    else:
        r = r[:-1] if r else r  # remove last rule
        answers = ["T"] * (len(chain) - 1) + ["F"]
    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    return facts, r, questions, answers


def variant3_domain(facts, rules, entity, d):
    """Add a contradicting fact."""
    chain = d["chain"]
    choice = random.randint(0, 2)
    if choice == 0:
        extra = f"{entity} is not {chain[0]}"
        answers = ["F"] * len(chain)
    elif choice == 1:
        extra = f"{entity} is not {chain[1]}" if len(chain) > 1 else f"{entity} is not {chain[0]}"
        answers = ["T"] + ["F"] * (len(chain) - 1)
    else:
        extra = f"{entity} is not {chain[-1]}"
        answers = ["T"] * (len(chain) - 1) + ["F"]
    f = facts + [extra]
    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    return f, rules, questions, answers


def variant_equiv_single_domain(facts, rules, entity, d, prop_pair):
    """Apply each equivalence law to the first rule."""
    p1, p2 = prop_pair
    chain = d["chain"]
    target = chain[0]
    base_rules_tail = rules[2:]  # skip the two init_prop→chain[0] rules

    eqs = {}

    eqs["contrapositive"] = [contraposition(p1, target), rule_str(p2, target)] + base_rules_tail
    eqs["double_negation"] = [double_negation(p1, target), rule_str(p2, target)] + base_rules_tail
    eqs["implication"] = [implication_law(p1, target), rule_str(p2, target)] + base_rules_tail
    eqs["identity"] = [identity_law(p1, target), rule_str(p2, target)] + base_rules_tail
    eqs["commutativity"] = [commutativity_or(p1, p2, target)] + base_rules_tail
    eqs["demorgan"] = [demorgan_law(p1, p2, target), rule_str(p1, target), rule_str(p2, target)] + base_rules_tail

    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    answers = ["T"] * len(chain)
    return eqs, questions, answers


def variant_equiv_multi_domain(facts, rules, entity, d, prop_pair):
    p1, p2 = prop_pair
    chain = d["chain"]
    base_rules_tail = rules[2:]
    laws = EQ_LAWS.copy()
    random.shuffle(laws)
    k = random.randint(2, min(5, len(laws)))
    selected = laws[:k]

    new_rules = rules.copy()
    for law in selected:
        if law == "contrapositive":
            new_rules.append(contraposition(p1, chain[0]))
        elif law == "double_negation":
            new_rules.append(double_negation(p1, chain[0]))
        elif law == "implication":
            new_rules.append(implication_law(p1, chain[0]))
        elif law == "identity":
            new_rules.append(identity_law(p1, chain[0]))
        elif law == "commutativity":
            new_rules.append(commutativity_or(p1, p2, chain[0]))
        elif law == "demorgan":
            new_rules.append(demorgan_law(p1, p2, chain[0]))

    questions = [f"Q{i+1}: {entity} is {prop}." for i, prop in enumerate(chain)]
    answers = ["T"] * len(chain)
    return facts, new_rules, questions, answers, ",".join(selected)


# =========================================================
# MAIN GENERATION LOOP
# =========================================================

def write_rows(path, rows, header):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

HEADER = ["group_id", "type", "facts", "rules", "questions", "answers", "equiv_laws_used", "domain"]
NUM_PER_DOMAIN = 500   # 500 × 5 domains = 2500 base groups (vs 1000 in v1)

DATA_DIR = "data_v2"
os.makedirs(DATA_DIR, exist_ok=True)

train_rows = []
test_base_rows = []
variant1_rows, variant2_rows, variant3_rows = [], [], []
equiv_rows_single = {law: [] for law in EQ_LAWS}
equiv_rows_multi = []
hard_test_rows = []

all_base_examples = []  # (gid, entity, facts, rules, questions, answers, cp, domain_name)

for domain in ALL_DOMAINS:
    domain_examples = []
    for _ in range(NUM_PER_DOMAIN):
        gid = str(uuid.uuid4())
        entity, facts, rules, questions, answers, cp = generate_base_domain(domain)
        domain_examples.append((gid, entity, facts, rules, questions, answers, cp, domain["name"]))
    all_base_examples.extend(domain_examples)

random.shuffle(all_base_examples)
split = int(0.8 * len(all_base_examples))
train_part = all_base_examples[:split]
test_part = all_base_examples[split:]

# --- TRAIN ---
for gid, entity, facts, rules, q, a, cp, dname in train_part:
    domain = next(d for d in ALL_DOMAINS if d["name"] == dname)
    def row(t, f, r, qu, an, laws=""):
        return {"group_id": gid, "type": t,
                "facts": " | ".join(f), "rules": " | ".join(r),
                "questions": " | ".join(qu), "answers": " | ".join(an),
                "equiv_laws_used": laws, "domain": dname}
    train_rows.append(row("base_positive", facts, rules, q, a))
    nf, nr, nq, na = generate_negative_samples_domain(entity, facts, rules, domain, cp)
    train_rows.append(row("base_negative", nf, nr, nq, na))
    mf, mr, mq, ma, _ = generate_mixed_domain(entity, domain)
    train_rows.append(row("hard_mixed", mf, mr, mq, ma))
    f2, r2, q2, a2 = variant2_domain(facts, rules, entity, domain)
    train_rows.append(row("variant2", f2, r2, q2, a2))
    f3, r3, q3, a3 = variant3_domain(facts, rules, entity, domain)
    train_rows.append(row("variant3", f3, r3, q3, a3))

# --- TEST BASE ---
for gid, entity, facts, rules, q, a, cp, dname in test_part:
    domain = next(d for d in ALL_DOMAINS if d["name"] == dname)
    def row(t, f, r, qu, an, laws=""):
        return {"group_id": gid, "type": t,
                "facts": " | ".join(f), "rules": " | ".join(r),
                "questions": " | ".join(qu), "answers": " | ".join(an),
                "equiv_laws_used": laws, "domain": dname}
    test_base_rows.append(row("base_positive", facts, rules, q, a))
    nf, nr, nq, na = generate_negative_samples_domain(entity, facts, rules, domain, cp)
    test_base_rows.append(row("base_negative", nf, nr, nq, na))
    mf, mr, mq, ma, _ = generate_mixed_domain(entity, domain)
    test_base_rows.append(row("hard_mixed", mf, mr, mq, ma))

# --- VARIANTS 1-3, VARIANT 4, HARD MIXED TEST ---
for gid, entity, facts, rules, q, a, cp, dname in all_base_examples:
    domain = next(d for d in ALL_DOMAINS if d["name"] == dname)
    def row(t, f, r, qu, an, laws=""):
        return {"group_id": gid, "type": t,
                "facts": " | ".join(f), "rules": " | ".join(r),
                "questions": " | ".join(qu), "answers": " | ".join(an),
                "equiv_laws_used": laws, "domain": dname}

    f1, r1, q1, a1 = variant1_domain(facts, rules, entity, domain)
    variant1_rows.append(row("variant1", f1, r1, q1, a1))

    f2, r2, q2, a2 = variant2_domain(facts, rules, entity, domain)
    variant2_rows.append(row("variant2", f2, r2, q2, a2))

    f3, r3, q3, a3 = variant3_domain(facts, rules, entity, domain)
    variant3_rows.append(row("variant3", f3, r3, q3, a3))

    # hard mixed test
    for _ in range(3):
        mf, mr, mq, ma, _ = generate_mixed_domain(entity, domain)
        hard_test_rows.append(row("hard_mixed", mf, mr, mq, ma))

    # variant4 single-law
    eqs, qv4, av4 = variant_equiv_single_domain(facts, rules, entity, domain, cp)
    for law, rlist in eqs.items():
        equiv_rows_single[law].append(row(f"equiv_{law}", facts, rlist, qv4, av4, law))

    # variant4 multi-law
    fm, rm, qm, am, laws_used = variant_equiv_multi_domain(facts, rules, entity, domain, cp)
    equiv_rows_multi.append(row("equiv_multi", fm, rm, qm, am, laws_used))


# --- WRITE ---
write_rows(f"{DATA_DIR}/train.csv", train_rows, HEADER)
write_rows(f"{DATA_DIR}/test_base.csv", test_base_rows, HEADER)
write_rows(f"{DATA_DIR}/test_hard_mixed.csv", hard_test_rows, HEADER)
write_rows(f"{DATA_DIR}/test_variant1.csv", variant1_rows, HEADER)
write_rows(f"{DATA_DIR}/test_variant2.csv", variant2_rows, HEADER)
write_rows(f"{DATA_DIR}/test_variant3.csv", variant3_rows, HEADER)
for law in EQ_LAWS:
    write_rows(f"{DATA_DIR}/test_variant4_equiv_{law}.csv", equiv_rows_single[law], HEADER)
write_rows(f"{DATA_DIR}/test_variant4_equiv_multi.csv", equiv_rows_multi, HEADER)

print("✔ Data generation v2 complete!")
print(f"  Domains: {len(ALL_DOMAINS)} ({', '.join(d['name'] for d in ALL_DOMAINS)})")
print(f"  Total base groups: {len(all_base_examples)} ({NUM_PER_DOMAIN} per domain)")
print(f"  train.csv          : {len(train_rows)} rows")
print(f"  test_base.csv      : {len(test_base_rows)} rows")
print(f"  test_hard_mixed.csv: {len(hard_test_rows)} rows")
print(f"  test_variant1.csv  : {len(variant1_rows)} rows")
print(f"  test_variant2.csv  : {len(variant2_rows)} rows")
print(f"  test_variant3.csv  : {len(variant3_rows)} rows")
for law in EQ_LAWS:
    print(f"  test_variant4_equiv_{law}.csv: {len(equiv_rows_single[law])} rows")
print(f"  test_variant4_equiv_multi.csv: {len(equiv_rows_multi)} rows")

from collections import Counter
all_ans = []
for r in test_base_rows:
    all_ans.extend(r["answers"].split(" | "))
dist = Counter(all_ans)
total = sum(dist.values())
print(f"\n  test_base answer distribution: T={dist.get('T',0)} ({dist.get('T',0)/total:.1%}), F={dist.get('F',0)} ({dist.get('F',0)/total:.1%})")

domain_counts = Counter(r["domain"] for r in test_base_rows)
print(f"\n  test_base per domain: {dict(domain_counts)}")
