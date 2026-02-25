## Overview
This repository provides a fully reproducible pipeline for evaluating whether language models (BERT / Qwen2 / LLaMA-family) can perform systematic logical reasoning.

It includes:
1. Synthetic data generation (base reasoning tasks + multiple controlled variants)
2. LoRA-based model training (BERT / Qwen2 / TinyLlama) with multiple training strategies
3. Detailed evaluation with prediction logging
4. Logical equivalence stress tests (single-law & multi-law)
5. Built-in annotations describing exactly what was changed per variant
6. Advanced training pipelines: CoT, DPO, Fusion, RA-CoT

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
pip install -r requirements.txt
```

## 2. Repository Structure
```
.
├── train.py                         # Main LoRA training script (bert / qwen / llama)
├── evaluate.py                      # Main evaluation suite
├── data_gen.py                      # Data generator for all variants
├── requirements.txt
│
├── data/
│   ├── train.csv                    # Base training set (80%)
│   ├── test_base.csv                # Base test set (20%)
│   ├── test_variant1.csv            # Remove redundant rule
│   ├── test_variant2.csv            # Remove key rule
│   ├── test_variant3.csv            # Contradictory facts
│   ├── test_variant4_equiv_*.csv    # Logical equivalence variants
│   ├── train_cot.csv                # CoT training data
│   ├── train_dpo.jsonl              # DPO training pairs
│   ├── train_fusion.csv             # Fusion training data
│   ├── train_mixed.csv              # Mixed training data
│   ├── train_ra_cot.csv             # RA-CoT training data
│   └── real_world/                  # Real-world NLI evaluation data
│
├── scripts/
│   ├── data_generation/             # Data generation scripts
│   │   ├── stage1_data_gen.py
│   │   ├── stage1_data_gen_v2.py
│   │   ├── generate_cot_data.py
│   │   ├── generate_dpo_data.py
│   │   ├── generate_fusion_data.py
│   │   ├── generate_mixed_data.py
│   │   ├── generate_ra_cot_data.py
│   │   └── prepare_real_world_data.py
│   │
│   ├── training/                    # Advanced training scripts
│   │   ├── stage1_train.py          # Stage-1 SFT
│   │   ├── stage2_train.py          # Stage-2 SFT
│   │   ├── stage2_train_cot.py      # Chain-of-Thought fine-tuning
│   │   ├── stage2_train_dpo.py      # DPO fine-tuning
│   │   ├── stage2_train_fusion.py   # Fusion training
│   │   ├── stage2_train_ra_cot.py   # Retrieval-Augmented CoT
│   │   └── train_real_world.py      # Real-world NLI training
│   │
│   ├── evaluation/                  # Extended evaluation scripts
│   │   ├── evaluate_direct.py       # Direct answer evaluation
│   │   ├── evaluate_cot.py          # CoT evaluation
│   │   ├── evaluate_generative.py   # Generative model evaluation
│   │   ├── evaluate_optimized.py    # Optimized evaluation
│   │   └── evaluate_real_world.py   # Real-world NLI evaluation
│   │
│   └── utils/                       # Utility scripts
│       ├── summarize_results.py
│       ├── generate_final_report.py
│       ├── convert_to_evals.py
│       └── verify_template_adherence.py
│
├── evals_data/                      # OpenAI Evals format test data
├── evals_submission/                # OpenAI Evals submission files
├── results/                         # Evaluation summary CSVs
└── docs/                            # Documentation, paper, reports
    ├── QUICKSTART.md
    ├── TRAINING_GUIDE.md
    ├── TRAINING_GUIDE_V2.md
    └── paper.tex
```

## 3. Data Generation
```
python data_gen.py
```

This will generate:

### 3.1 Training Set
```
data/train.csv — base examples (80%)
```

### 3.2 Test Sets
```
data/test_base.csv — original reasoning chain

data/test_variant1.csv — redundant rule removed

data/test_variant2.csv — critical rule removed

data/test_variant3.csv — contradictory facts added

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
|questions|All 4 questions separated by \||
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

|Question|Base|Variant2|
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

### 5.1 Basic Training (LoRA fine-tuning)

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

### 5.2 Advanced Training Strategies

For extended training pipelines, see `scripts/training/`:

| Script | Strategy |
|:---|:---|
| `stage1_train.py` | Stage-1 SFT on base data |
| `stage2_train_cot.py` | Chain-of-Thought fine-tuning |
| `stage2_train_dpo.py` | Direct Preference Optimization |
| `stage2_train_fusion.py` | Fusion of SFT + CoT |
| `stage2_train_ra_cot.py` | Retrieval-Augmented CoT |

Generate training data for advanced strategies:
```
python scripts/data_generation/generate_cot_data.py
python scripts/data_generation/generate_dpo_data.py
python scripts/data_generation/generate_fusion_data.py
```

## 6. Evaluation

### 6.1 Standard Evaluation
```
python evaluate.py --model bert
python evaluate.py --model qwen
python evaluate.py --model llama
```

The evaluation script:
- Evaluates all 11 test sets
- Saves predictions under `trained_models/{model}/predictions/`

### 6.2 Extended Evaluation

| Script | Description |
|:---|:---|
| `scripts/evaluation/evaluate_direct.py` | Direct answer evaluation |
| `scripts/evaluation/evaluate_cot.py` | Chain-of-Thought evaluation |
| `scripts/evaluation/evaluate_generative.py` | Generative model evaluation |
| `scripts/evaluation/evaluate_real_world.py` | Real-world NLI (LogicNLI / MNLI) |

Produces an accuracy table:

<img width="580" height="300" alt="image" src="https://github.com/user-attachments/assets/d62c11a0-0c90-4962-89d2-280166def15e" />

We also submit the variant 3 test set to the Human Last Exam benchmark — all state-of-the-art models fail, including claude-sonnet-4-5, gpt-4.1, gpt-5.2, claude-opus-4-5, and gemini-3-pro-preview.
<img width="2333" height="1619" alt="image" src="https://github.com/user-attachments/assets/1dbb3195-cf47-46b1-9a75-b3525a6465fd" />


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
