## Overview
This repository provides a fully reproducible pipeline for evaluating whether language models (BERT / Qwen2 / LLaMA-family) can perform systematic logical reasoning.

It includes:
1. Synthetic data generation (base reasoning tasks + multiple controlled variants)
2. LoRA-based model training (BERT / Qwen2 / TinyLlama)
3. Detailed evaluation with prediction logging
4. Logical equivalence stress tests (single-law & multi-law)
5. Built-in annotations describing exactly what was changed per variant

This framework allows you to measure how models behave when:
1. Rules are removed
2. Rules contradict each other
3. Rules are rewritten using logical equivalence
4. Multiple equivalent rules are added and stacked

## 1. Environment Setup
### 1.1 Create environment (recommended: conda)
```
conda create -n logic python=3.10 -y
conda activate logic
```

### 1.2 Install dependencies
```
pip install torch transformers datasets peft accelerate sentencepiece pandas numpy tqdm
```
(Install CUDA packages depending on GPU setup if needed.)

## 2. Repository Structure
```
.
├── data_gen.py                      # Data generator for all variants
├── train.py                         # LoRA model training (bert / qwen / llama)
├── evaluate.py                      # Full evaluation suite
├── train.csv                        # Auto-generated (base, 80%)
├── test_base.csv                    # Base test set (base, 20%)
├── test_variant1.csv                # Remove redundant rule
├── test_variant2.csv                # Remove key rule
├── test_variant3.csv                # Contradictory facts
├── test_variant4_equiv_contrapositive.csv
├── test_variant4_equiv_double_negation.csv
├── test_variant4_equiv_implication.csv
├── test_variant4_equiv_demorgan.csv
├── test_variant4_equiv_identity.csv
├── test_variant4_equiv_commutativity.csv
├── test_variant4_equiv_multi.csv    # 2–5 logical equivalence rules combined
└── trained_models/
    ├── bert/
    │   └── predictions/*.csv
    ├── qwen/
    │   └── predictions/*.csv
    └── llama/
        └── predictions/*.csv
```

## 3. Data Generation
```
python data_gen.py
```

This will generate:

### 3.1 Training Set
```
train.csv — base examples (80%)
```

### 3.2 Test set
```
test_base.csv — original reasoning chain

test_variant1.csv — redundant rule removed

test_variant2.csv — critical rule removed

test_variant3.csv — contradictory facts added

Variant 4 — logical equivalence tests:
1. test_variant4_equiv_contrapositive.csv
2. test_variant4_equiv_double_negation.csv
3. test_variant4_equiv_implication.csv
4. test_variant4_equiv_demorgan.csv
5. test_variant4_equiv_identity.csv
6. test_variant4_equiv_commutativity.csv

Variant 4 multi-law:
1. test_variant4_equiv_multi.csv — 2–5 different equivalence laws applied
```

Each CSV contains:

|Column|Meaning|
|:---|:---|
|group_id|Unique example ID|
|type|base / variantX / logical_equiv|
|facts|Natural-language facts|
|rules|Rules used for inference|
|facts|Natural-language facts|
|questions|All 4 questions separated by|
|answers|Corresponding "T" / "F" truth values|
|equiv_laws_used|For logical equivalence cases only|

## 4. Example Cases
### 4.1 Base Example
**Facts**
```
Anne is green or blue
```

**Rules**
```
If someone is green then they are cold.
If someone is blue then they are cold.
If someone is cold then they are rough.
If someone is not young then they are not rough.
If someone is young then they are cold.
If someone is young then they are nice.
```

**Questions & Answers**
```
Q1: Anne is cold.      → T
Q2: Anne is rough.     → T
Q3: Anne is young.     → T
Q4: Anne is nice.      → T
```

**Variant 1 — Remove Redundant Rule**
Removed:
```
If someone is young then they are cold.
```
All conclusions remain the same.

**Variant 2 — Remove Key Rule**
Removed:
```
If someone is cold then they are rough.
```

|Question|Base|Varient2|
|:---|:---|:--|
|Q1 cold|T|T|
|Q2 rough|T|F|
|Q3 young|T|F|
|Q4 nice|T|F|

**Variant 3 — Contradictory Facts**
Added:
```
Anne is not cold or not nice
```
This breaks the reasoning chain → all answers become False.

**Variant 4 — Logical Equivalence**
We rewrite:
```
If someone is green then they are cold.
```
using multiple logical-equivalent forms.

**Contrapositive**
```
If someone is not cold then they are not green.
```

**Double Negation**
```
If someone is not not green then they are not not cold.
```

**Implication Law**
```
Someone is not green or they are cold.
```

**De Morgan**
```
If someone is not green and not blue then they are not cold.
```

**Identity**
```
If someone is not not green then they are cold.
```

**Commutativity**
```
If someone is blue or green then they are cold.
```

**Multi-Law Example**
```
equiv_laws_used="contrapositive,implication,demorgan"
equiv_law_count=3
```

## 5. Model Training
Train BERT:
```
python train.py --model bert
```

Train Qwen2:
```
python train.py --model qwen
```

Train LLaMA-family (TinyLlama):
```
python train.py --model llama
```

All models use:
1. LoRA fine-tuning
2. Same preprocessing
3. Automatic pad token fix for decoder-only models

Trained models saved to:
```
trained_models/{model}/
```

## 6. Evaluation
Run evaluation:
```
python evaluate.py --model bert
python evaluate.py --model qwen
python evaluate.py --model llama
```

The evaluation script:

Evaluates all 11 test sets

Saves predictions under:
```
trained_models/{model}/predictions/{model}_{split}_predictions.csv
```

Produces an accuracy table:

<img width="304" height="157" alt="image" src="https://github.com/user-attachments/assets/d62c11a0-0c90-4962-89d2-280166def15e" />

**Prediction CSV includes:**
|Column|Description|
|:---|:---|
|facts|Facts used|
|rules|Rule list|
|question|Question text|
|ground_truth|True answer|
|prediction|Model prediction|
|equiv_laws_used|Which logical laws applied|
|equiv_law_count|Number of laws|
|changed_rule|Human-readable explanation|

## 7. Observed Model Behavior

Across BERT / Qwen2 / TinyLlama:
1. Base reasoning: perfect (1.00 accuracy)
2. Variant 1: unaffected (1.00)
3. Variant 2: fails logically (≈0.25)
4. Variant 3: fully broken (0.00)
5. Variant 4 single-law: robust (1.00)
6. Variant 4 multi-law: also robust (1.00)

This reveals:
1. Models rely on full rule chains
2. Contradictions confuse them
3. Logical equivalence does not break reasoning
4. Redundant rule clutter does not harm performance

