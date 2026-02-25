## Overview

This repository provides a fully reproducible pipeline for studying whether language models (BERT / Qwen2 / LLaMA) can perform systematic logical reasoning — and how robustly that reasoning survives rule perturbations.

It includes:
1. Synthetic data generation with multiple controlled perturbation variants
2. LoRA-based model training with a **two-stage training pipeline**
3. Multiple training strategies: SFT, Generative, Mixed, DPO, CoT, Fusion, RA-CoT
4. Detailed evaluation across 11 test splits with prediction logging
5. Logical equivalence stress tests (single-law & multi-law)
6. Real-world NLI generalization evaluation (LogicNLI / MNLI)

---

## 1. Environment Setup

```bash
conda create -n logic python=3.10 -y
conda activate logic
pip install -r requirements.txt
```

---

## 2. Repository Structure

```
.
├── train.py                   # Main LoRA training script
├── evaluate.py                # Main evaluation suite
├── data_gen.py                # Data generator for all variants
├── requirements.txt
│
├── data/
│   ├── train.csv              # Base training set (80%)
│   ├── test_base.csv          # Base test set (20%)
│   ├── test_variant{1-3}.csv  # Rule perturbation variants
│   ├── test_variant4_equiv_*.csv   # Logical equivalence variants (×7)
│   ├── train_cot.csv          # CoT training data
│   ├── train_dpo.jsonl        # DPO preference pairs
│   ├── train_fusion.csv       # Fusion (SFT+CoT) training data
│   ├── train_mixed.csv        # Mixed generative training data
│   ├── train_ra_cot.csv       # RA-CoT training data
│   └── real_world/            # LogicNLI / MNLI evaluation data
│
├── scripts/
│   ├── data_generation/       # Data generation scripts
│   ├── training/              # Advanced training scripts
│   ├── evaluation/            # Extended evaluation scripts
│   └── utils/                 # Utilities, debug, reporting
│
├── evals_data/                # OpenAI Evals format test data
├── evals_submission/          # OpenAI Evals submission
├── results/                   # Evaluation summary CSVs
└── docs/                      # Documentation, paper, reports
```

---

## 3. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA GENERATION                                │
│                                                                         │
│  data_gen.py ──► train.csv / test_base.csv / test_variant{1-4}.csv     │
│                                                                         │
│  scripts/data_generation/                                               │
│    stage1_data_gen.py   ──► data/stage1_train_{bert,generative}.csv    │
│    generate_cot_data.py ──► data/train_cot.csv                         │
│    generate_dpo_data.py ──► data/train_dpo.jsonl                       │
│    generate_fusion_data.py ──► data/train_fusion.csv                   │
│    generate_mixed_data.py  ──► data/train_mixed.csv                    │
│    generate_ra_cot_data.py ──► data/train_ra_cot.csv                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1 TRAINING                                 │
│                                                                         │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐   │
│  │   SFT on variant    │    │   Generative: rule generation task   │   │
│  │   2/3 style data    │    │   facts+masked_rules → missing_rule  │   │
│  │  stage1_train.py    │    │   stage1_train_generative.py         │   │
│  └──────────┬──────────┘    └──────────────────┬───────────────────┘   │
│             │                                  │                        │
│    bert_stage1 / qwen_stage1       qwen_stage1_gen checkpoint          │
└─────────────┼──────────────────────────────────┼────────────────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2 TRAINING                                 │
│           (fine-tune on Stage-1 checkpoint; Qwen2 / LLaMA)             │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   Mixed SFT  │  │     DPO      │  │     CoT      │  │  Fusion   │  │
│  │ (T/F + rule  │  │ (preference  │  │ (step-by-    │  │ (SFT+CoT) │  │
│  │  prediction) │  │   pairs)     │  │  step trace) │  │           │  │
│  │stage2_train_ │  │stage2_train_ │  │stage2_train_ │  │stage2_    │  │
│  │generative.py │  │   dpo.py     │  │   cot.py     │  │fusion.py  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                  │                │         │
│  qwen_stage2_mixed  qwen_stage2_dpo    (cot model)    (fusion model)   │
│  llama_stage2_mixed                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION                                    │
│                                                                         │
│  evaluate.py ──► 11 synthetic splits (base + variant1-3 + variant4×7) │
│                                                                         │
│  scripts/evaluation/                                                    │
│    evaluate_real_world.py ──► LogicNLI / MNLI generalization test      │
│    evaluate_cot.py        ──► CoT model evaluation                     │
│    evaluate_generative.py ──► Generative model evaluation              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Generation

```bash
python data_gen.py
```

### Test Splits

| Split | Description | Expected Behavior |
|:---|:---|:---|
| `test_base.csv` | Original reasoning chain | All correct |
| `test_variant1.csv` | Redundant rule removed | Unchanged answers |
| `test_variant2.csv` | **Critical rule removed** | Answers change |
| `test_variant3.csv` | **Contradictory facts injected** | All False |
| `test_variant4_equiv_contrapositive.csv` | Contrapositive rewrite | Unchanged answers |
| `test_variant4_equiv_double_negation.csv` | Double negation rewrite | Unchanged answers |
| `test_variant4_equiv_implication.csv` | Implication law rewrite | Unchanged answers |
| `test_variant4_equiv_demorgan.csv` | De Morgan rewrite | Unchanged answers |
| `test_variant4_equiv_identity.csv` | Identity rewrite | Unchanged answers |
| `test_variant4_equiv_commutativity.csv` | Commutativity rewrite | Unchanged answers |
| `test_variant4_equiv_multi.csv` | 2–5 laws combined | Unchanged answers |

---

## 5. Model Training

### 5.1 Basic Training

```bash
python train.py --model bert    # BERT-base-uncased
python train.py --model qwen    # Qwen2-1.5B
python train.py --model llama   # TinyLlama-1.1B
```

### 5.2 Advanced Training

**Stage 1:**
```bash
python scripts/data_generation/stage1_data_gen.py
python scripts/training/stage1_train.py --model qwen
python scripts/training/stage1_train_generative.py --model qwen
```

**Stage 2 (run after Stage 1):**
```bash
# Generate data first
python scripts/data_generation/generate_mixed_data.py
python scripts/data_generation/generate_dpo_data.py
python scripts/data_generation/generate_cot_data.py

# Train
python scripts/training/stage2_train_generative.py    # Mixed SFT
python scripts/training/stage2_train_dpo.py           # DPO
python scripts/training/stage2_train_cot.py           # CoT
python scripts/training/stage2_train_fusion.py        # Fusion
```

---

## 6. Evaluation

```bash
python evaluate.py --model bert
python evaluate.py --model qwen
python evaluate.py --model llama
```

Predictions saved to `trained_models/{model}/predictions/{model}_{split}_predictions.csv`.

---

## 7. Results

### 7.1 Full Accuracy Table

`—` = not evaluated on this split.  V4-avg = average over all evaluated logical equivalence splits.

| Model | Strategy | Base | V1 | V2 | V3 | V4-avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| BERT | Stage-1 SFT | **1.000** | **1.000** | 0.295 | 0.000 | 0.999 |
| BERT (stage2) | Stage-2 SFT | **1.000** | **1.000** | 0.250 | 0.000 | **1.000** |
| LLaMA (TinyLlama) | Stage-1 SFT | **1.000** | **1.000** | 0.250 | 0.000 | 0.999 |
| LLaMA | Stage-2 Mixed | 0.538 | 0.693 | 0.533 | 0.145 | 0.797 |
| Qwen2-1.5B | Stage-1 SFT | **1.000** | **1.000** | 0.250 | 0.000 | 0.943 |
| Qwen2 | Stage-1 Generative | 0.175 | 0.185 | 0.555 | **0.908** | 0.165 † |
| Qwen2 | Stage-2 DPO | 0.000 | 0.000 | **0.750** | **1.000** | 0.000 † |
| Qwen2 | Stage-2 Mixed | 0.525 | **0.938** | 0.405 | **0.973** | 0.444 |
| Qwen2 | Stage-2 Mixed+Aug | 0.488 | 0.908 | 0.450 | **0.988** | 0.400 |

> † Only 3 of 7 V4 splits evaluated.

### 7.2 Logical Equivalence Detail (V4 per law)

| Law | BERT | BERT-S2 | LLaMA | LLaMA-S2 | Qwen | Qwen-Mixed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Commutativity | 0.993 | **1.000** | **1.000** | 0.858 | **1.000** | 0.498 |
| Contrapositive | **1.000** | **1.000** | **1.000** | 0.705 | **1.000** | 0.318 |
| De Morgan | **1.000** | **1.000** | **1.000** | 0.910 | **1.000** | 0.163 |
| Double Negation | **1.000** | **1.000** | **1.000** | 0.803 | **1.000** | 0.545 |
| Identity | **1.000** | **1.000** | **1.000** | 0.815 | **1.000** | 0.570 |
| Implication | **1.000** | **1.000** | **1.000** | 0.745 | 0.953 | 0.590 |
| Multi-law | **1.000** | **1.000** | 0.993 | 0.745 | 0.645 | 0.428 |

### 7.3 Real-World Generalization (LogicNLI / MNLI)

Models trained on synthetic logic data were evaluated on real-world NLI benchmarks.

| Model | Dataset | Predictions | Accuracy |
|:---|:---|:---|:---:|
| Qwen2 Fusion-Conflict | LogicNLI (n=500) | All "Unknown" | 0.000 |
| Qwen2 Fusion-Conflict | MNLI (n=349) | All "Unknown" | 0.000 |
| Qwen2 RealWorld-SFT | LogicNLI (n=500) | All "Unknown" | 0.000 |

> All models predict "Unknown" on real-world NLI, indicating **zero generalization** from synthetic logic reasoning to natural language inference. The reasoning skills learned are tightly coupled to the synthetic template format.

### 7.4 Key Findings

**Standard LoRA SFT is robust to logical equivalence but brittle to contradictions:**
- Perfect on Base / Variant 1 / Variant 4 (equivalence rewriting)
- ~0.25 on Variant 2 (near-random, loses critical rule) — relies on complete rule chains
- 0.00 on Variant 3 (contradictions fully break reasoning)

**Mixed/Generative training recovers contradiction robustness at a cost:**
- Stage-2 Mixed reaches 0.97–0.99 on Variant 3 and 0.94 on Variant 1
- But Variant 4 (logical equivalence) accuracy drops to ~0.40–0.45
- Variant 2 also stays weak at 0.40–0.45

**DPO maximizes contradiction robustness but collapses elsewhere:**
- Best Variant 3 (1.00) and Variant 2 (0.75)
- Catastrophic failure on Base (0.00) and all Variant 4 — collapses to always predicting False

**Core trade-off:**
> Models robust to logical equivalence rewrites (SFT) are brittle to contradictions. Models that handle contradictions (Mixed/DPO) lose logical equivalence robustness. No single training strategy dominates across all perturbation types.

**No generalization to real-world NLI:**
> All models predict "Unknown" on LogicNLI and MNLI, showing the learned reasoning is format-specific and does not transfer to natural language.

---

### 7.5 Human Benchmark Comparison

We submitted the Variant 3 test set to the **Human Last Exam** benchmark. All state-of-the-art models fail, including claude-sonnet-4-5, gpt-4.1, gpt-5.2, claude-opus-4-5, and gemini-3-pro-preview.

<img width="580" height="300" alt="accuracy table" src="https://github.com/user-attachments/assets/d62c11a0-0c90-4962-89d2-280166def15e" />

<img width="2333" height="1619" alt="human benchmark" src="https://github.com/user-attachments/assets/1dbb3195-cf47-46b1-9a75-b3525a6465fd" />

---

## 8. Example Cases

### Base Example

**Facts:** `Anne is green or blue`

**Rules:**
```
If someone is green then they are cold.
If someone is blue then they are cold.
If someone is cold then they are rough.
If someone is not young then they are not rough.
If someone is young then they are cold.
If someone is young then they are nice.
```

| Q | Base | V1 (remove redundant) | V2 (remove key) | V3 (contradiction) |
|:---|:---:|:---:|:---:|:---:|
| Anne is cold | T | T | T | F |
| Anne is rough | T | T | F | F |
| Anne is young | T | T | F | F |
| Anne is nice | T | T | F | F |

### Variant 4 — Logical Equivalence Rewrites

Original: `If someone is green then they are cold.`

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

## 9. Output Format

Predictions saved to `trained_models/{model}/predictions/{model}_{split}_predictions.csv`:

| Column | Description |
|:---|:---|
| `facts` | Input facts |
| `rules` | Rule list |
| `question` | Question text |
| `ground_truth` | Correct answer |
| `prediction` | Model prediction |
| `equiv_laws_used` | Logical laws applied (V4 only) |
| `equiv_law_count` | Number of laws applied |
| `changed_rule` | Human-readable description of the change |
