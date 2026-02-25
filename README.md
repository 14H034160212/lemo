## Overview

This repository provides a fully reproducible pipeline for studying whether language models (BERT / Qwen2 / LLaMA-family) can perform systematic logical reasoning — and how robustly that reasoning survives rule perturbations.

It includes:
1. Synthetic data generation with multiple controlled perturbation variants
2. LoRA-based model training with a two-stage training pipeline
3. Multiple training strategies: SFT, Generative, Mixed, DPO, CoT, Fusion, RA-CoT
4. Detailed evaluation across 11 test splits with prediction logging
5. Logical equivalence stress tests (single-law & multi-law)
6. Real-world NLI evaluation (LogicNLI / MNLI)

The framework measures how models behave when:
1. Rules are removed
2. Rules contradict each other
3. Rules are rewritten using logical equivalence
4. Multiple equivalent rules are added and stacked

---

## 1. Environment Setup

### 1.1 Create environment (recommended: conda)
```bash
conda create -n logic python=3.10 -y
conda activate logic
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Repository Structure
```
.
├── train.py                         # Main LoRA training script
├── evaluate.py                      # Main evaluation suite
├── data_gen.py                      # Data generator for all variants
├── requirements.txt
│
├── data/
│   ├── train.csv                    # Base training set (80%)
│   ├── test_base.csv                # Base test set (20%)
│   ├── test_variant{1-3}.csv        # Perturbation variants
│   ├── test_variant4_equiv_*.csv    # Logical equivalence variants (×7)
│   ├── train_cot.csv                # Chain-of-Thought training data
│   ├── train_dpo.jsonl              # DPO training pairs
│   ├── train_fusion.csv             # Fusion (SFT+CoT) training data
│   ├── train_mixed.csv              # Mixed training data
│   ├── train_ra_cot.csv             # RA-CoT training data
│   └── real_world/                  # LogicNLI / MNLI evaluation data
│
├── scripts/
│   ├── data_generation/             # Data generation scripts
│   ├── training/                    # Advanced training scripts
│   ├── evaluation/                  # Extended evaluation scripts
│   └── utils/                       # Utilities, debug, reporting
│
├── evals_data/                      # OpenAI Evals format test data
├── evals_submission/                # OpenAI Evals submission
├── results/                         # Evaluation summary CSVs
└── docs/                            # Documentation, paper, reports
```

---

## 3. Data Generation

### 3.1 Generate all data
```bash
python data_gen.py
```

### 3.2 Test Splits Generated

| Split | Description |
|:---|:---|
| `test_base.csv` | Original reasoning chain |
| `test_variant1.csv` | Redundant rule removed (answers unchanged) |
| `test_variant2.csv` | Critical rule removed (answers change) |
| `test_variant3.csv` | Contradictory facts injected (all False) |
| `test_variant4_equiv_contrapositive.csv` | Rule rewritten via contrapositive |
| `test_variant4_equiv_double_negation.csv` | Rule rewritten via double negation |
| `test_variant4_equiv_implication.csv` | Rule rewritten via implication law |
| `test_variant4_equiv_demorgan.csv` | Rule rewritten via De Morgan |
| `test_variant4_equiv_identity.csv` | Rule rewritten via identity |
| `test_variant4_equiv_commutativity.csv` | Rule rewritten via commutativity |
| `test_variant4_equiv_multi.csv` | 2–5 equivalence laws combined |

Each CSV contains:

| Column | Meaning |
|:---|:---|
| `group_id` | Unique example ID |
| `type` | base / variantX / logical_equiv |
| `facts` | Natural-language facts |
| `rules` | Rules used for inference |
| `questions` | 4 questions separated by `\|` |
| `answers` | Corresponding T/F truth values |
| `equiv_laws_used` | Laws applied (logical equivalence cases only) |

---

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
Q1: Anne is cold.   → T
Q2: Anne is rough.  → T
Q3: Anne is young.  → T
Q4: Anne is nice.   → T
```

### 4.2 Variant 2 — Remove Key Rule

Removed: `If someone is cold then they are rough.`

| Question | Base | Variant 2 |
|:---|:---:|:---:|
| cold | T | T |
| rough | T | F |
| young | T | F |
| nice | T | F |

### 4.3 Variant 3 — Contradictory Facts

Added: `Anne is not cold or not nice` → breaks reasoning chain → all answers become **False**.

### 4.4 Variant 4 — Logical Equivalence

Rewriting `If someone is green then they are cold.` with different laws:

| Law | Rewritten Form |
|:---|:---|
| Contrapositive | `If someone is not cold then they are not green.` |
| Double Negation | `If someone is not not green then they are not not cold.` |
| Implication | `Someone is not green or they are cold.` |
| De Morgan | `If someone is not green and not blue then they are not cold.` |
| Identity | `If someone is not not green then they are cold.` |
| Commutativity | `If someone is blue or green then they are cold.` |
| Multi-law | `equiv_laws_used="contrapositive,implication,demorgan"` |

---

## 5. Training Pipeline

### 5.1 Basic Training (single-stage LoRA)

```bash
python train.py --model bert    # BERT
python train.py --model qwen    # Qwen2-1.5B
python train.py --model llama   # TinyLlama-1.1B
```

All models use LoRA fine-tuning with automatic pad token fix for decoder-only models.
Trained models saved to `trained_models/{model}/`.

### 5.2 Two-Stage Training Pipeline

The advanced pipeline separates training into two stages to improve robustness on hard variants (variant2/variant3).

**Stage 1 — Pre-training on incomplete/generative reasoning:**

| Script | Task |
|:---|:---|
| `scripts/training/stage1_train.py` | SFT on incomplete-rules data (variant2/3 style) |
| `scripts/training/stage1_train_bert.py` | Stage-1 for BERT |
| `scripts/training/stage1_train_generative.py` | Rule generation: input=masked rules → output=missing rule |

Generate Stage-1 data:
```bash
python scripts/data_generation/stage1_data_gen.py
```

**Stage 2 — Fine-tuning with multiple strategies (builds on Stage-1 checkpoint):**

| Script | Strategy | Key Idea |
|:---|:---|:---|
| `scripts/training/stage2_train.py` | SFT | Standard supervised fine-tuning |
| `scripts/training/stage2_train_cot.py` | Chain-of-Thought | Step-by-step reasoning traces |
| `scripts/training/stage2_train_dpo.py` | DPO | Direct Preference Optimization on correct/incorrect pairs |
| `scripts/training/stage2_train_fusion.py` | Fusion | Combined SFT + CoT loss |
| `scripts/training/stage2_train_generative.py` | Generative | Mixed rule-prediction + T/F tasks |
| `scripts/training/stage2_train_ra_cot.py` | RA-CoT | Retrieval-Augmented Chain-of-Thought |
| `scripts/training/train_real_world.py` | Real-World SFT | Trained on LogicNLI / MNLI data |

Generate Stage-2 data:
```bash
python scripts/data_generation/generate_cot_data.py
python scripts/data_generation/generate_dpo_data.py
python scripts/data_generation/generate_fusion_data.py
python scripts/data_generation/generate_mixed_data.py
python scripts/data_generation/generate_ra_cot_data.py
```

---

## 6. Evaluation

### 6.1 Standard Evaluation
```bash
python evaluate.py --model bert
python evaluate.py --model qwen
python evaluate.py --model llama
```

Evaluates all 11 test splits and saves predictions to `trained_models/{model}/predictions/`.

### 6.2 Extended Evaluation

| Script | Description |
|:---|:---|
| `scripts/evaluation/evaluate_direct.py` | Direct answer evaluation |
| `scripts/evaluation/evaluate_cot.py` | Chain-of-Thought evaluation |
| `scripts/evaluation/evaluate_generative.py` | Generative model evaluation |
| `scripts/evaluation/evaluate_optimized.py` | Optimized batched evaluation |
| `scripts/evaluation/evaluate_real_world.py` | Real-world NLI (LogicNLI / MNLI) |

---

## 7. Results

### 7.1 Accuracy Table

All experiments, accuracy per test split:

| Model | Stage | Base | V1 | V2 | V3 | V4-avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| BERT | Stage-1 SFT | **1.00** | **1.00** | 0.30 | 0.00 | **1.00** |
| BERT | Stage-2 SFT | **1.00** | **1.00** | 0.25 | 0.00 | **1.00** |
| LLaMA (TinyLlama) | Stage-1 SFT | **1.00** | **1.00** | 0.25 | 0.00 | ~**1.00** |
| LLaMA | Stage-2 Mixed | 0.54 | 0.69 | 0.53 | 0.15 | ~0.80 |
| Qwen2-1.5B | Stage-1 SFT | **1.00** | **1.00** | 0.25 | 0.00 | ~0.95 |
| Qwen2 | Stage-1 Generative | 0.18 | 0.19 | 0.56 | **0.91** | ~0.15 |
| Qwen2 | Stage-2 DPO | 0.00 | 0.00 | **0.75** | **1.00** | ~0.00 |
| Qwen2 | Stage-2 Mixed | 0.53 | **0.94** | 0.41 | **0.97** | ~0.45 |
| Qwen2 | Stage-2 Mixed+Aug | 0.49 | 0.91 | 0.45 | **0.99** | ~0.40 |

> V4-avg = average across all 7 logical equivalence splits.

<img width="580" height="300" alt="image" src="https://github.com/user-attachments/assets/d62c11a0-0c90-4962-89d2-280166def15e" />

### 7.2 Key Findings

**Standard LoRA (Stage-1 SFT):**
- Perfect accuracy on Base, Variant 1, and most Variant 4 (logical equivalence)
- Consistently fails on Variant 2 (≈0.25, near random) — relies on complete rule chains
- Always fails on Variant 3 (0.00) — contradictions fully break reasoning

**Generative / Mixed Training:**
- Stage-1 Generative (rule generation task) dramatically improves Variant 3 (0.91) but loses base accuracy
- Stage-2 Mixed training reaches 0.97–0.99 on Variant 3 while partially recovering base/V1
- Trade-off: generative/mixed training hurts Variant 4 logical equivalence performance

**DPO Training:**
- Best Variant 3 performance (1.00) and strong Variant 2 (0.75)
- Catastrophically fails on Base and Variant 4 — collapses to predicting "False"
- DPO alone is not a complete solution

**Overall:**
> Models that excel at logical equivalence robustness (Stage-1 SFT) are brittle to contradictions, while models that handle contradictions well (Mixed/DPO) lose logical equivalence robustness. This robustness trade-off is a core finding of this work.

### 7.3 Human Benchmark Comparison

We submitted the Variant 3 test set to the **Human Last Exam** benchmark. All state-of-the-art models fail, including claude-sonnet-4-5, gpt-4.1, gpt-5.2, claude-opus-4-5, and gemini-3-pro-preview.

<img width="2333" height="1619" alt="image" src="https://github.com/user-attachments/assets/1dbb3195-cf47-46b1-9a75-b3525a6465fd" />

---

## 8. Prediction Output Format

Saved to `trained_models/{model}/predictions/{model}_{split}_predictions.csv`:

| Column | Description |
|:---|:---|
| `facts` | Facts used |
| `rules` | Rule list |
| `question` | Question text |
| `ground_truth` | True answer |
| `prediction` | Model prediction |
| `equiv_laws_used` | Which logical laws applied |
| `equiv_law_count` | Number of laws |
| `changed_rule` | Human-readable description of the change |
