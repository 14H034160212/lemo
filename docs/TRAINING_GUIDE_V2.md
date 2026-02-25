# Two-Stage Training Guide V2: Rule Prediction + T/F Classification

This guide explains the **correct** two-stage training approach with masked rule modeling.

## Overview

### Key Concept: Masked Rule Modeling

Similar to BERT's Masked Language Modeling (MLM), we train models to predict missing rules:

- **Stage 1**: Train to predict masked rules (similar to MLM)
  - Input: facts + **masked_rules** + question
  - Output: **missing_rule**

- **Stage 2**: Mixed training on both tasks
  - Task A: Predict missing rules (from Stage 1)
  - Task B: Answer T/F questions (original task)

### Model Types

| Model | Stage 1 Approach | Stage 2 Approach |
|-------|------------------|------------------|
| **Qwen/LLaMA (Generative)** | Generate missing rule text | Mixed: generate rules + generate T/F |
| **BERT (Discriminative)** | Multiple choice rule selection | Standard T/F classification |

---

## Prerequisites

1. Generate base training/test data:
```bash
python data_gen.py
```

This creates:
- `data/train.csv` - Original T/F training data
- `data/test_*.csv` - Test splits

---

## Path 1: Generative Models (Qwen/LLaMA) - **Recommended**

### Step 1.1: Generate Stage 1 Data (Rule Prediction)

```bash
python stage1_data_gen_v2.py \
    --train_csv data/train.csv \
    --output_prefix data/stage1_train \
    --num_samples 200 \
    --format generative
```

**Output**: `data/stage1_train_generative.csv`

**Sample format**:
```
Input:
Given the following information:

Facts: Bob is green or blue

Rules:
- If someone is blue then they are cold.
- If someone is not young then they are not rough.
- If someone is young then they are cold.
- If someone is young then they are nice.

Question: Q1: Bob is cold.

One critical rule is missing from the rules above. What is the missing rule?

Missing rule:

Target:
If someone is green then they are cold.
```

### Step 1.2: Train Stage 1 Model (Rule Prediction)

```bash
# Train with Qwen
python stage1_train_generative.py --model qwen --epochs 3

# Or train with LLaMA
python stage1_train_generative.py --model llama --epochs 3
```

**Output**: Model saved to `trained_models/{model}_stage1_gen/`

**What the model learns**: Given incomplete rules, generate the missing critical rule.

### Step 2: Train Stage 2 Model (Mixed Training)

```bash
# Continue from Stage 1 with mixed data (RECOMMENDED)
python stage2_train_generative.py \
    --model qwen \
    --from_stage1 \
    --mixed_data \
    --epochs 2

# Or just fine-tune on original T/F data
python stage2_train_generative.py \
    --model qwen \
    --from_stage1 \
    --epochs 2
```

**Output**: Model saved to `trained_models/{model}_stage2_mixed/`

**What the model learns**:
- Task A: Predict missing rules (from Stage 1)
- Task B: Answer T/F questions
- **Combined**: Better reasoning under both complete and incomplete information

### Step 3: Evaluate

```bash
# Evaluate Stage 1 model
python evaluate_generative.py --model qwen --stage stage1_gen

# Evaluate Stage 2 mixed model
python evaluate_generative.py --model qwen --stage stage2_mixed
```

---

## Path 2: BERT (Multiple Choice)

BERT cannot easily generate text, so we use **multiple choice** for Stage 1.

### Step 2.1: Generate Stage 1 Data (Multiple Choice)

```bash
python stage1_data_gen_v2.py \
    --train_csv data/train.csv \
    --output_prefix data/stage1_train \
    --num_samples 200 \
    --format bert
```

**Output**: `data/stage1_train_bert.csv`

**Sample format**:
```
Context: Facts: Bob is green or blue. Rules: [masked rules]. Question: Q1: Bob is cold.

Candidates:
0. If someone is green then they are cold. ← CORRECT
1. If someone is cold then they are nice.
2. If someone is rough then they are cold.
3. If someone is young then they are rough.

Correct Answer: 0
```

### Step 2.2: Train Stage 1 Model (Multiple Choice)

```bash
python stage1_train_bert.py --epochs 3
```

**Output**: Model saved to `trained_models/bert_stage1_mc/`

### Step 2.3: Train Stage 2 Model

For BERT, use the original `train.py` for Stage 2:

```bash
python train.py --model bert
```

**Note**: BERT's Stage 2 is separate from Stage 1 due to different task formats.

### Step 2.4: Evaluate

```bash
# Evaluate original BERT model
python evaluate.py --model bert
```

---

## Complete Workflow Examples

### Example 1: Qwen with Mixed Training (Full Pipeline)

```bash
# 1. Generate base data
python data_gen.py

# 2. Generate Stage 1 data (rule prediction)
python stage1_data_gen_v2.py --format generative --num_samples 200

# 3. Train Stage 1 (rule prediction)
python stage1_train_generative.py --model qwen --epochs 3 --batch_size 2

# 4. Train Stage 2 (mixed: rule prediction + T/F)
python stage2_train_generative.py \
    --model qwen \
    --from_stage1 \
    --mixed_data \
    --epochs 2 \
    --batch_size 2

# 5. Evaluate
python evaluate_generative.py --model qwen --stage stage2_mixed
```

### Example 2: LLaMA without Mixed Data

```bash
# Assumes base data already generated

# 1. Generate Stage 1 data
python stage1_data_gen_v2.py --format generative

# 2. Train Stage 1
python stage1_train_generative.py --model llama --epochs 3

# 3. Train Stage 2 (only T/F, no mixing)
python stage2_train_generative.py \
    --model llama \
    --from_stage1 \
    --epochs 2

# 4. Evaluate
python evaluate_generative.py --model llama --stage stage2
```

### Example 3: BERT Multiple Choice

```bash
# 1. Generate base data
python data_gen.py

# 2. Generate Stage 1 data (multiple choice format)
python stage1_data_gen_v2.py --format bert --num_samples 200

# 3. Train Stage 1 (multiple choice)
python stage1_train_bert.py --epochs 3

# 4. (Optional) Train Stage 2 with original script
python train.py --model bert

# 5. Evaluate original model
python evaluate.py --model bert
```

---

## Training Strategy Comparison

| Strategy | Description | Expected Performance |
|----------|-------------|----------------------|
| **Baseline** | Train only on complete T/F data | Good on base, struggles on variant2 |
| **Stage 1 only** | Train only on rule prediction | Good at finding missing rules, poor on T/F |
| **Stage 2 from Stage 1** | Stage1 → Continue on T/F | Better generalization |
| **Stage 2 mixed** | Stage1 → Continue on both tasks | **Best**: handles both incomplete rules and T/F |

---

## Data Formats

### Generative Format (Qwen/LLaMA)

**Stage 1 (Rule Prediction)**:
```csv
input_text,target_text
"Given facts... Rules (masked)... What is the missing rule?","If someone is cold then they are rough."
```

**Stage 2 (T/F Prediction - converted from original data)**:
```csv
input_text,target_text
"Given facts... Rules... Question: Bob is cold. True or false?","True"
```

### BERT Format (Multiple Choice)

```csv
context,candidate_0,candidate_1,candidate_2,candidate_3,correct_answer
"Facts... Rules (masked)... Question...","Rule A","Rule B","Rule C","Rule D",0
```

---

## Key Differences from Previous Version

| Aspect | Old Approach | New Approach (V2) |
|--------|--------------|-------------------|
| **Stage 1 Task** | T/F classification on incomplete data | **Predict missing rule text** |
| **Generative Models** | Used sequence classification | **Use text generation (causal LM)** |
| **BERT** | Used sequence classification | **Use multiple choice** |
| **Loss Computation** | Classification loss on T/F | **Generation loss on rule text** |
| **Evaluation** | Predict T/F | **Generate T/F or rule text** |

---

## File Structure

```
.
├── data_gen.py                      # Generate base data
├── stage1_data_gen_v2.py            # Generate Stage 1 data (V2)
│
├── # Generative models (Qwen/LLaMA)
├── stage1_train_generative.py       # Stage 1: Rule prediction
├── stage2_train_generative.py       # Stage 2: Mixed training
├── evaluate_generative.py           # Evaluation for generative models
│
├── # BERT
├── stage1_train_bert.py             # Stage 1: Multiple choice
├── train.py                         # Stage 2: Original T/F training
├── evaluate.py                      # Evaluation for BERT
│
├── data/
│   ├── train.csv                    # Original T/F data
│   ├── stage1_train_generative.csv  # Stage 1 data (generative)
│   ├── stage1_train_bert.csv        # Stage 1 data (BERT)
│   └── test_*.csv                   # Test splits
│
└── trained_models/
    ├── qwen_stage1_gen/             # Qwen Stage 1
    ├── qwen_stage2_mixed/           # Qwen Stage 2 (mixed)
    ├── llama_stage1_gen/            # LLaMA Stage 1
    ├── llama_stage2_mixed/          # LLaMA Stage 2 (mixed)
    └── bert_stage1_mc/              # BERT Stage 1 (multiple choice)
```

---

## Advanced Configuration

### Custom Learning Rates

```bash
# Stage 1: Higher LR for initial learning
python stage1_train_generative.py --model qwen --learning_rate 5e-5

# Stage 2: Lower LR for fine-tuning
python stage2_train_generative.py --model qwen --from_stage1 --learning_rate 1e-5
```

### GPU Memory Optimization

```bash
# Reduce batch size and use gradient accumulation
python stage1_train_generative.py \
    --model qwen \
    --batch_size 1 \
    --epochs 5
```

### Generate More Training Data

```bash
# Generate 500 samples instead of 200
python stage1_data_gen_v2.py --format generative --num_samples 500
```

---

## Troubleshooting

### Issue: Model generates gibberish
**Solution**: Increase training epochs or reduce learning rate:
```bash
python stage1_train_generative.py --model qwen --epochs 5 --learning_rate 1e-5
```

### Issue: CUDA OOM (Out of Memory)
**Solution**: Reduce batch size:
```bash
python stage1_train_generative.py --model qwen --batch_size 1
```

### Issue: Generated answers are always "False"
**Solution**: Check if training data is balanced. Regenerate with more samples:
```bash
python stage1_data_gen_v2.py --num_samples 500
```

### Issue: Low accuracy on variant2
**Solution**: This is expected if you skip Stage 1. Train with mixed data:
```bash
python stage2_train_generative.py --model qwen --from_stage1 --mixed_data
```

---

## Expected Results

### Stage 1 (Rule Prediction)
- Model should generate reasonable rules (even if not exactly matching)
- Accuracy on rule prediction: 40-70% (exact match)

### Stage 2 (Mixed Training)
- **Base accuracy**: >95% (should remain high)
- **Variant2 accuracy**: 20-40% → **50-70%** (improvement from mixed training)
- **Variant3 accuracy**: 0-10% → **20-40%** (better at handling contradictions)

---

## Summary

### For Qwen/LLaMA (Generative):
1. Generate rule prediction data (`stage1_data_gen_v2.py --format generative`)
2. Train Stage 1 on rule prediction (`stage1_train_generative.py`)
3. Train Stage 2 with mixed data (`stage2_train_generative.py --mixed_data`)
4. Evaluate (`evaluate_generative.py`)

### For BERT:
1. Generate multiple choice data (`stage1_data_gen_v2.py --format bert`)
2. Train Stage 1 on multiple choice (`stage1_train_bert.py`)
3. Train Stage 2 on original task (`train.py`)
4. Evaluate (`evaluate.py`)

---

## Contact

For questions or issues, please refer to the main README.
