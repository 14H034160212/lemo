# data_gen.py
import random
import csv
import uuid
import os

random.seed(42)

NAMES = [
    "Anne", "Bob", "Claire", "David", "Emma",
    "Frank", "Grace", "Helen", "Ivan", "Julia",
    "Kevin", "Linda", "Mike", "Nancy", "Oscar",
]
COLORS = ["green", "blue", "red", "yellow", "purple", "orange"]

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
# NON-EQUIVALENT REWRITES  (the control class for Variant 4)
# =========================================================
# Variant 4 as originally built contained only logic-PRESERVING rewrites, so
# every question kept its answer and the whole split was labelled T. On an
# all-T split a model that simply answers T scores 1.000, which is
# indistinguishable from one that genuinely recognises logical equivalence --
# measured directly: LIRE lifts variant4-multi 0.712 -> 0.998 while its rate of
# answering T on the balanced base split goes 0.443 -> 0.833 and base accuracy
# falls 1.000 -> 0.694. The "invariance gain" was a shift toward T.
#
# These are the classical fallacies: near-identical surface form, different
# logic. A model that has learned equivalence must keep its answer under the
# transformations above and CHANGE it under these. Labels are not hand-written;
# they are computed by the forward-chaining oracle (see variant_noneq_single).

def converse(p, q):
    # P -> Q  does NOT entail  Q -> P
    return f"If someone is {q} then they are {p}."

def inverse(p, q):
    # P -> Q  does NOT entail  not P -> not Q
    return f"If someone is not {p} then they are not {q}."

def affirming_consequent(p, q):
    # treats Q as sufficient for P
    return f"If someone is {q} then they are {p}."

def demorgan_fallacy(p, q):
    # not (P or Q)  is NOT  not P or not Q
    return f"If someone is not {p} or not {q} then they are not cold."

def disjunction_to_conjunction(p, q):
    # P or Q  is NOT  P and Q
    return f"If someone is {p} and {q} then they are cold."


# =========================================================
# BASE GENERATION — answers: T, T, T, T
# =========================================================

def generate_base(name):
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])

    facts = [f"{name} is {color1} or {color2}"]

    rules = [
        rule(color1, "cold"),
        rule(color2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),   # contrapositive: rough → young
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
# NEGATIVE SAMPLES — answers: F, F, F, F
# =========================================================

def generate_negative_samples(name, facts, rules, color_pair):
    """
    Negative samples: ask about negated or unrelated properties.
    All answers should be F.
    """
    c1, c2 = color_pair
    unrelated_props = ["happy", "sad", "kind", "mean", "tall", "short", "funny", "brave"]

    neg_questions = []
    neg_answers = []

    # Type 1: negated known-true property
    neg_questions.append(f"Q1: {name} is not cold.")
    neg_answers.append("F")

    neg_questions.append(f"Q2: {name} is not rough.")
    neg_answers.append("F")

    # Type 2: unrelated property (not derivable)
    random_prop1 = random.choice(unrelated_props)
    neg_questions.append(f"Q3: {name} is {random_prop1}.")
    neg_answers.append("F")

    # Type 3: wrong color
    other_colors = [c for c in COLORS if c not in [c1, c2]]
    if other_colors:
        wrong_color = random.choice(other_colors)
        neg_questions.append(f"Q4: {name} is {wrong_color}.")
        neg_answers.append("F")
    else:
        random_prop2 = random.choice([p for p in unrelated_props if p != random_prop1])
        neg_questions.append(f"Q4: {name} is {random_prop2}.")
        neg_answers.append("F")

    return facts, rules, neg_questions, neg_answers


# =========================================================
# HARD MIXED SAMPLES — answers: T, T, F, F
# (cold and rough derivable; young and nice NOT derivable)
# =========================================================

def generate_mixed_ttff(name):
    """
    Pattern T, T, F, F.
    Remove the 'not young → not rough' rule so young cannot be inferred.
    cold=T (from color), rough=T (cold→rough), young=F (no derivation), nice=F (needs young).
    """
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])
    facts = [f"{name} is {color1} or {color2}"]
    rules = [
        rule(color1, "cold"),
        rule(color2, "cold"),
        rule("cold", "rough"),
        # deliberately OMIT: rule("not young", "not rough")
        rule("young", "cold"),
        rule("young", "nice"),
        # distractor: an irrelevant rule about a property no one has
        f"If someone is tall then they are warm.",
    ]
    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]
    answers = ["T", "T", "F", "F"]
    return facts, rules, questions, answers, (color1, color2)


# =========================================================
# HARD MIXED SAMPLES — answers: T, F, T, T
# (cold derivable; rough NOT derivable; young given; nice from young)
# =========================================================

def generate_mixed_tftt(name):
    """
    Pattern T, F, T, T.
    Young is given as a fact; cold→rough rule is absent.
    cold=T (young→cold), rough=F (no rule), young=T (given), nice=T (young→nice).
    """
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])
    facts = [f"{name} is {color1} or {color2}", f"{name} is young"]
    rules = [
        rule(color1, "cold"),
        rule(color2, "cold"),
        # deliberately OMIT: rule("cold", "rough")
        rule("young", "cold"),
        rule("young", "nice"),
        # distractor
        f"If someone is brave then they are rough.",
    ]
    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]
    answers = ["T", "F", "T", "T"]
    return facts, rules, questions, answers, (color1, color2)


# =========================================================
# HARD MIXED SAMPLES — answers: T, T, T, F
# (cold, rough, young all derivable; nice NOT derivable)
# =========================================================

def generate_mixed_tttf(name):
    """
    Pattern T, T, T, F.
    Remove young→nice so nice cannot be inferred.
    cold=T, rough=T, young=T (via rough→young), nice=F (no rule).
    """
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])
    facts = [f"{name} is {color1} or {color2}"]
    rules = [
        rule(color1, "cold"),
        rule(color2, "cold"),
        rule("cold", "rough"),
        rule("not young", "not rough"),   # rough→young via contrapositive
        rule("young", "cold"),
        # deliberately OMIT: rule("young", "nice")
        # distractor
        f"If someone is kind then they are nice.",
    ]
    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]
    answers = ["T", "T", "T", "F"]
    return facts, rules, questions, answers, (color1, color2)


# =========================================================
# HARD MIXED SAMPLES — answers: F, F, T, T
# (young given as fact; NO color→cold chain)
# =========================================================

def generate_mixed_fftt(name):
    """
    Pattern F, F, T, T.
    Only 'young' is a fact; no rules that derive cold or rough from young.
    cold=F (no derivation), rough=F (no derivation), young=T (given), nice=T (young→nice).
    """
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])
    # Facts: only young — no color facts so cold chain is broken
    facts = [f"{name} is young"]
    rules = [
        # No color→cold rules (colors not in facts anyway)
        # No cold→rough
        # No young→cold  (so cold stays F)
        rule("young", "nice"),
        # distractor: a plausible-looking rule
        f"If someone is {color1} then they are cold.",
        f"If someone is cold then they are rough.",
    ]
    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]
    answers = ["F", "F", "T", "T"]
    return facts, rules, questions, answers, (color1, color2)


# =========================================================
# HARD MIXED SAMPLES — answers: F, T, T, T  (via negation chain)
# =========================================================

def generate_mixed_fttt(name):
    """
    Pattern F, T, T, T.
    Name is explicitly not cold; rough derived from 'not cold → rough'; young from rough; nice from young.
    cold=F (given as not-cold fact), rough=T (not-cold→rough), young=T (rough→young via contrapositive),
    nice=T (young→nice).
    """
    color1 = random.choice(COLORS)
    color2 = random.choice([c for c in COLORS if c != color1])
    facts = [f"{name} is not cold"]  # cold is explicitly false
    rules = [
        rule("not cold", "rough"),         # not-cold → rough
        rule("not young", "not rough"),    # contrapositive: rough → young
        rule("young", "nice"),
        # distractors
        f"If someone is {color1} then they are cold.",
        f"If someone is {color2} then they are cold.",
    ]
    questions = [
        f"Q1: {name} is cold.",
        f"Q2: {name} is rough.",
        f"Q3: {name} is young.",
        f"Q4: {name} is nice.",
    ]
    answers = ["F", "T", "T", "T"]
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
    """
    Randomly remove one of three critical rules; each gives a DIFFERENT answer pattern:
      Sub-type A: remove cold→rough  (rules[2]) → [T, F, F, F]
      Sub-type B: remove not_young→not_rough (rules[3]) → [T, T, F, F]
                  (rough still T from cold→rough, but rough→young derivation gone)
      Sub-type C: remove young→nice  (rules[5]) → [T, T, T, F]
    This ensures the model cannot simply memorise a single answer pattern.
    """
    choice = random.randint(0, 2)
    r = rules.copy()
    if choice == 0:   # remove cold→rough
        r = [x for i, x in enumerate(r) if i != 2]
        answers = ["T", "F", "F", "F"]
    elif choice == 1:  # remove not_young→not_rough (contrapositive of rough→young)
        r = [x for i, x in enumerate(r) if i != 3]
        answers = ["T", "T", "F", "F"]
    else:              # remove young→nice
        r = [x for i, x in enumerate(r) if i != 5]
        answers = ["T", "T", "T", "F"]
    return facts, r, [f"Q1: {name} is cold.",
                      f"Q2: {name} is rough.",
                      f"Q3: {name} is young.",
                      f"Q4: {name} is nice."], answers


def variant3(facts, rules, name):
    """
    Inject one of three contradicting facts, at three different points in the
    deduction chain. All three yield [F, F, F, F].

    The all-False labelling is required by the conservative contradiction
    semantics the benchmark is defined under: once the premise set is
    inconsistent (Gamma |- bot), the deduction is invalid and *every* query on
    that instance is False, including the ones whose derivation would still go
    through if the contradiction were ignored.

    An earlier revision labelled the sub-types [F,F,F,F] / [T,F,F,F] /
    [T,T,T,F] to stop a model memorising one answer pattern. That silently
    switched Variant 3 to a "locally still derivable" semantics and left ~34%
    True labels, which is incompatible with the definition above and makes the
    contradiction-detection failure mode unmeasurable: a model that ignores the
    contradiction entirely scores well instead of scoring zero. The three
    injection points are kept — they still prevent pattern memorisation,
    because the model must detect a contradiction sited anywhere in the chain —
    but the correct response to all of them is to halt, so all answers are F.
    """
    choice = random.randint(0, 2)
    if choice == 0:
        extra = f"{name} is not cold"    # contradicts the head of the chain
    elif choice == 1:
        extra = f"{name} is not rough"   # contradicts mid-chain
    else:
        extra = f"{name} is not nice"    # contradicts the tail
    answers = ["F", "F", "F", "F"]
    f = facts + [extra]
    return f, rules, [f"Q1: {name} is cold.",
                      f"Q2: {name} is rough.",
                      f"Q3: {name} is young.",
                      f"Q4: {name} is nice."], answers


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

def variant_noneq_single(facts, rules, color_pair, name):
    """Variant 4's control class: rewrites that are NOT logic-preserving.

    Same shape as variant_equiv_single, but each rewrite is a classical
    fallacy rather than an equivalence, so the answers are expected to move.
    Labels come from forward_chain(), never from a hand-written pattern -- the
    whole point of this split is that the answer is not predictable from the
    split name, so writing the labels by hand would defeat it.
    """
    from scripts.utils.forward_chain import forward_chain, check_answer

    c1, c2 = color_pair
    tail = [
        rule("cold", "rough"),
        rule("not young", "not rough"),
        rule("young", "cold"),
        rule("young", "nice"),
    ]
    # Both colour rules must be rewritten. The fact is a disjunction
    # ("<name> is c1 or c2"), so corrupting only the c1 rule leaves the c2 rule
    # to derive "cold" anyway and the whole chain survives unchanged -- which is
    # exactly the all-T degeneracy this split exists to avoid.
    neq = {
        "converse":         [converse(c1, "cold"), converse(c2, "cold")] + tail,
        "inverse":          [inverse(c1, "cold"), inverse(c2, "cold")] + tail,
        "demorgan_fallacy": [demorgan_fallacy(c1, c2)] + tail,
        "disj_to_conj":     [disjunction_to_conjunction(c1, c2)] + tail,
    }

    questions = [f"Q1: {name} is cold.",
                 f"Q2: {name} is rough.",
                 f"Q3: {name} is young.",
                 f"Q4: {name} is nice."]

    # NOTE on how this split must be *used*: on its own it is all-F, just as the
    # equivalence split on its own is all-T, and either one alone can be
    # saturated by a constant predictor. They are only informative interleaved
    # into a single split whose labels then contain both classes -- see
    # variant4_mixed in the writer below. Keeping them as separate dicts here is
    # deliberate: the caller decides the mix.
    out = {}
    facts_str = " | ".join(facts)
    for law, rule_set in neq.items():
        rules_str = " | ".join(rule_set)
        closure = forward_chain(facts_str, rules_str)
        answers = []
        for q in questions:
            val = check_answer(q, closure)
            # An attribute that cannot be derived is False under the benchmark's
            # closed-world reading, matching how every other split is labelled.
            answers.append("T" if val is True else "F")
        out[law] = (rule_set, questions, answers)
    return out


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
equiv_rows_multi = []
# Variant 4's control split: equivalence rewrites (answers preserved) interleaved
# with non-equivalence rewrites (answers change). Neither half is usable alone --
# each is single-class and a constant predictor saturates it. Mixed, the labels
# contain both classes, so accuracy here separates a model that recognises
# logical equivalence from one that has merely drifted toward a single answer.
noneq_rows_mixed = []

base_examples = []
# NUM=1000: train 800, test 200; variant test sets have 1000 examples (4000 questions)
# Balances diversity, quantity and training time
NUM = 1000

MIXED_GENERATORS = [
    generate_mixed_ttff,
    generate_mixed_tftt,
    generate_mixed_tttf,
    generate_mixed_fftt,
    generate_mixed_fttt,
]

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# common header with extra column for equivalence info
header = ["group_id","type","facts","rules","questions","answers","equiv_laws_used"]

# ---- TRAIN SET ----
# For each training base example, add: positive + negative samples
for gid, name, facts, rules, q, a, cp in train_part:
    # Positive sample (base, all T)
    train_rows.append({
        "group_id": gid,
        "type": "base_positive",
        "facts": " | ".join(facts),
        "rules": " | ".join(rules),
        "questions": " | ".join(q),
        "answers": " | ".join(a),
        "equiv_laws_used": "",
    })
    # Negative sample
    neg_facts, neg_rules, neg_q, neg_a = generate_negative_samples(name, facts, rules, cp)
    train_rows.append({
        "group_id": gid,
        "type": "base_negative",
        "facts": " | ".join(neg_facts),
        "rules": " | ".join(neg_rules),
        "questions": " | ".join(neg_q),
        "answers": " | ".join(neg_a),
        "equiv_laws_used": "",
    })
    # Mixed hard sample (randomly choose one generator)
    gen = random.choice(MIXED_GENERATORS)
    mf, mr, mq, ma, _ = gen(name)
    train_rows.append({
        "group_id": gid,
        "type": f"hard_mixed_{gen.__name__}",
        "facts": " | ".join(mf),
        "rules": " | ".join(mr),
        "questions": " | ".join(mq),
        "answers": " | ".join(ma),
        "equiv_laws_used": "",
    })
    # Variant2 sample: cold→rough rule removed, answers T,F,F,F
    f2, r2, q2, a2 = variant2(facts, rules, name)
    train_rows.append({
        "group_id": gid,
        "type": "variant2",
        "facts": " | ".join(f2),
        "rules": " | ".join(r2),
        "questions": " | ".join(q2),
        "answers": " | ".join(a2),
        "equiv_laws_used": "",
    })
    # Variant3 sample: contradicting fact added, answers F,F,F,F
    f3, r3, q3, a3 = variant3(facts, rules, name)
    train_rows.append({
        "group_id": gid,
        "type": "variant3",
        "facts": " | ".join(f3),
        "rules": " | ".join(r3),
        "questions": " | ".join(q3),
        "answers": " | ".join(a3),
        "equiv_laws_used": "",
    })

# ---- TEST SET (base) ----
# Include: positive, negative, AND mixed samples so accuracy=1 is impossible by trivial guessing
for gid, name, facts, rules, q, a, cp in test_part:
    # Positive sample (all T)
    test_base_rows.append({
        "group_id": gid,
        "type": "base_positive",
        "facts": " | ".join(facts),
        "rules": " | ".join(rules),
        "questions": " | ".join(q),
        "answers": " | ".join(a),
        "equiv_laws_used": "",
    })
    # Negative sample (all F on negated/unrelated queries)
    neg_facts, neg_rules, neg_q, neg_a = generate_negative_samples(name, facts, rules, cp)
    test_base_rows.append({
        "group_id": gid,
        "type": "base_negative",
        "facts": " | ".join(neg_facts),
        "rules": " | ".join(neg_rules),
        "questions": " | ".join(neg_q),
        "answers": " | ".join(neg_a),
        "equiv_laws_used": "",
    })
    # One hard mixed sample per base example (cycle through generators)
    gen = MIXED_GENERATORS[int(gid[:8], 16) % len(MIXED_GENERATORS)]
    mf, mr, mq, ma, _ = gen(name)
    test_base_rows.append({
        "group_id": gid,
        "type": f"hard_mixed",
        "facts": " | ".join(mf),
        "rules": " | ".join(mr),
        "questions": " | ".join(mq),
        "answers": " | ".join(ma),
        "equiv_laws_used": "",
    })

# ---- HARD TEST SET ----
# A dedicated hard test file with all 5 mixed types across all 600 base examples
hard_test_rows = []
for gid, name, facts, rules, q, a, cp in base_examples:
    for gen in MIXED_GENERATORS:
        mf, mr, mq, ma, _ = gen(name)
        hard_test_rows.append({
            "group_id": gid,
            "type": gen.__name__,
            "facts": " | ".join(mf),
            "rules": " | ".join(mr),
            "questions": " | ".join(mq),
            "answers": " | ".join(ma),
            "equiv_laws_used": "",
        })

# ---- VARIANTS 1–3 ----
for gid, name, facts, rules, q, a, cp in base_examples:
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

# ---- VARIANT 4 SINGLE-LAW ----
for gid, name, facts, rules, q, a, cp in base_examples:
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

# ---- VARIANT 4 MIXED (equivalence + non-equivalence control) ----
for gid, name, facts, rules, q, a, cp in base_examples:
    # equivalence half: answers unchanged
    for law, rlist in variant_equiv_single(facts, rules, cp).items():
        noneq_rows_mixed.append({
            "group_id": gid,
            "type": f"equiv_{law}",
            "facts": " | ".join(facts),
            "rules": " | ".join(rlist),
            "questions": " | ".join(q),
            "answers": " | ".join(a),
            "equiv_laws_used": law,
        })
    # non-equivalence half: answers recomputed by the forward-chaining oracle
    for law, (rlist, qs, ans) in variant_noneq_single(facts, rules, cp, name).items():
        noneq_rows_mixed.append({
            "group_id": gid,
            "type": f"noneq_{law}",
            "facts": " | ".join(facts),
            "rules": " | ".join(rlist),
            "questions": " | ".join(qs),
            "answers": " | ".join(ans),
            "equiv_laws_used": f"NONEQ:{law}",
        })

# ---- VARIANT 4 MULTI-LAW ----
for gid, name, facts, rules, q, a, cp in base_examples:
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


# =========================================================
# WRITE FILES
# =========================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Two training corpora, because the baselines and the method need different ones.
#
#   train.csv          — NO Variant-3 rows. This is what the untreated baselines
#                        (train.py) are trained on, so Variant 3 stays genuinely
#                        held out for them and the contradiction failure mode is
#                        measurable. If a baseline is trained on Variant 3 it just
#                        learns "these inputs are all False" and scores 1.000,
#                        which measures exposure, not contradiction detection.
#   train_with_v3.csv  — the full corpus including Variant 3. The conflict-aware
#                        method is supposed to learn halting from contradiction
#                        examples, so its data generators read this file.
train_rows_no_v3 = [r for r in train_rows if r["type"] != "variant3"]
write_rows(f"{DATA_DIR}/train.csv", train_rows_no_v3, header)
write_rows(f"{DATA_DIR}/train_with_v3.csv", train_rows, header)
write_rows(f"{DATA_DIR}/test_base.csv", test_base_rows, header)
write_rows(f"{DATA_DIR}/test_hard_mixed.csv", hard_test_rows, header)
write_rows(f"{DATA_DIR}/test_variant1.csv", variant1_rows, header)
write_rows(f"{DATA_DIR}/test_variant2.csv", variant2_rows, header)
write_rows(f"{DATA_DIR}/test_variant3.csv", variant3_rows, header)

for law, rows in equiv_rows_single.items():
    write_rows(f"{DATA_DIR}/test_variant4_equiv_{law}.csv", rows, header)

write_rows(f"{DATA_DIR}/test_variant4_equiv_multi.csv", equiv_rows_multi, header)
write_rows(f"{DATA_DIR}/test_variant4_mixed.csv", noneq_rows_mixed, header)

# ---- STATS ----
print(f"✔ Data generation complete! Files saved to '{DATA_DIR}/'")
print(f"  train.csv          : {len(train_rows_no_v3)} rows  (base_pos + base_neg + hard_mixed + variant2; Variant 3 held out)")
print(f"  train_with_v3.csv  : {len(train_rows)} rows  (adds variant3, for the conflict-aware method)")
print(f"  test_base.csv      : {len(test_base_rows)} rows (positive + negative + mixed)")
print(f"  test_hard_mixed.csv: {len(hard_test_rows)} rows (5 mixed types × {NUM} examples)")
print(f"  test_variant1.csv  : {len(variant1_rows)} rows")
print(f"  test_variant2.csv  : {len(variant2_rows)} rows")
print(f"  test_variant3.csv  : {len(variant3_rows)} rows")
for law in EQ_LAWS:
    print(f"  test_variant4_equiv_{law}.csv: {len(equiv_rows_single[law])} rows")
print(f"  test_variant4_equiv_multi.csv: {len(equiv_rows_multi)} rows")
print(f"  test_variant4_mixed.csv: {len(noneq_rows_mixed)} rows  (equivalence + non-equivalence control)")

# Answer distribution in test_base
from collections import Counter
all_answers = []
for row in test_base_rows:
    all_answers.extend(row["answers"].split(" | "))
dist = Counter(all_answers)
total = sum(dist.values())
print(f"\n  test_base answer distribution: T={dist.get('T',0)} ({dist.get('T',0)/total:.1%}), F={dist.get('F',0)} ({dist.get('F',0)/total:.1%})")
print("  (A model predicting all-T or all-F cannot reach accuracy=1)")
