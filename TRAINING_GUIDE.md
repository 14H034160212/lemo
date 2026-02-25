# Two-Stage Training Guide

This guide explains how to use the two-stage training pipeline for logical reasoning models.

## Overview

The two-stage training approach:
- **Stage 1**: Train the model on incomplete/contradictory data (variant2/variant3 style)
- **Stage 2**: Continue training on complete data, or train with mixed data

## Prerequisites

1. Generate base training data:
```bash
python data_gen.py
```

This creates:
- `data/train.csv` - Original training data
- `data/test_*.csv` - Test splits for evaluation

## Stage 1: Train on Incomplete Rules

### Step 1.1: Generate Stage 1 Training Data

```bash
python stage1_data_gen.py \
    --train_csv data/train.csv \
    --output_prefix data/stage1_train \
    --num_samples 200 \
    --seed 42
```

This creates:
- `data/stage1_train_variant2.csv` - Incomplete rules samples
- `data/stage1_train_variant3.csv` - Contradictory facts samples
- `data/stage1_train_combined.csv` - Mixed variant2 + variant3 samples

**Sample output:**
```
Facts: Bob is green or blue
Rules: If someone is green then they are cold. | If someone is blue then they are cold. | If someone is not young then they are not rough. | If someone is young then they are cold. | If someone is young then they are nice.
Questions: Q1: Bob is cold. | Q2: Bob is rough. | Q3: Bob is young. | Q4: Bob is nice.
Answers: T | F | F | F
```

Note: The critical rule "If someone is cold then they are rough" is removed.

### Step 1.2: Train Stage 1 Model

```bash
# Train with BERT
python stage1_train.py --model bert

# Train with Qwen
python stage1_train.py --model qwen

# Train with LLaMA (TinyLlama)
python stage1_train.py --model llama
```

**Optional parameters:**
```bash
python stage1_train.py \
    --model bert \
    --train_data data/stage1_train_combined.csv \
    --output_dir trained_models/bert_stage1 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

**Output:** Model saved to `trained_models/{model}_stage1/`

## Stage 2: Continue Training or Mixed Training

You have **two options** for Stage 2:

### Option A: Continue from Stage 1 (Recommended)

Train on original data, starting from Stage 1 checkpoint:

```bash
python stage2_train.py \
    --model bert \
    --from_stage1 \
    --original_data data/train.csv
```

This loads the Stage 1 model and continues training on complete reasoning data.

**Output:** Model saved to `trained_models/{model}_stage2/`

### Option B: Mixed Data Training

Train on combined Stage 1 + original data:

```bash
python stage2_train.py \
    --model bert \
    --from_stage1 \
    --mixed_data \
    --original_data data/train.csv \
    --stage1_data data/stage1_train_combined.csv
```

This combines both datasets for more diverse training.

**Output:** Model saved to `trained_models/{model}_stage2_mixed/`

### Option C: Mixed Data from Scratch

Train from base model (not recommended, for comparison only):

```bash
python stage2_train.py \
    --model bert \
    --mixed_data
```

## Evaluation

### Evaluate Stage 1 Model

```bash
python evaluate.py --model bert --stage stage1
```

Or with custom path:
```bash
python evaluate.py --model bert --model_dir trained_models/bert_stage1
```

### Evaluate Stage 2 Model

```bash
# Evaluate Stage 2 (continue from stage1)
python evaluate.py --model bert --stage stage2

# Evaluate Stage 2 mixed
python evaluate.py --model bert --stage stage2_mixed
```

### Evaluate Original Model (for comparison)

```bash
python evaluate.py --model bert
```

This loads from `trained_models/bert/` (trained with `train.py`).

## Complete Workflow Example

Here's a complete example using BERT:

```bash
# 1. Generate base data (if not done)
python data_gen.py

# 2. Generate Stage 1 training data
python stage1_data_gen.py --num_samples 200

# 3. Train Stage 1 model
python stage1_train.py --model bert --epochs 3

# 4. Evaluate Stage 1 model
python evaluate.py --model bert --stage stage1

# 5. Train Stage 2 model (continue from Stage 1)
python stage2_train.py --model bert --from_stage1 --epochs 2

# 6. Evaluate Stage 2 model
python evaluate.py --model bert --stage stage2

# 7. (Optional) Train Stage 2 with mixed data
python stage2_train.py --model bert --from_stage1 --mixed_data --epochs 2

# 8. Evaluate Stage 2 mixed model
python evaluate.py --model bert --stage stage2_mixed
```

## Training Strategy Comparison

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Original (train.py)** | Train on complete data only | Baseline model |
| **Stage 1 only** | Train on incomplete/contradictory data | Test robustness to missing rules |
| **Stage 2 (from stage1)** | Stage1 → Continue on complete data | **Recommended**: Best of both worlds |
| **Stage 2 (mixed)** | Stage1 → Continue on stage1+complete data | More training diversity |
| **Stage 2 (from scratch)** | Train on stage1+complete from base model | Comparison baseline |

## Expected Results

The two-stage training approach should:
1. **Stage 1**: Model learns to handle incomplete information
2. **Stage 2**: Model generalizes better to test variants

Evaluation metrics:
- **Base accuracy**: Should remain high (>95%)
- **Variant2 accuracy**: Should improve compared to single-stage training
- **Variant3 accuracy**: Should improve in detecting contradictions

## File Structure

```
.
├── data_gen.py              # Generate base training/test data
├── stage1_data_gen.py       # Generate Stage 1 training data
├── train.py                 # Original single-stage training
├── stage1_train.py          # Stage 1 training script
├── stage2_train.py          # Stage 2 training script
├── evaluate.py              # Evaluation script (supports all stages)
├── data/
│   ├── train.csv                      # Original training data
│   ├── stage1_train_combined.csv      # Stage 1 training data
│   └── test_*.csv                     # Test splits
└── trained_models/
    ├── bert/                # Original model
    ├── bert_stage1/         # Stage 1 model
    ├── bert_stage2/         # Stage 2 model (continue)
    └── bert_stage2_mixed/   # Stage 2 model (mixed)
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size:
```bash
python stage1_train.py --model bert --batch_size 2
```

### Issue: Stage 1 data not found
**Solution:** Make sure to run `stage1_data_gen.py` first:
```bash
python stage1_data_gen.py
```

### Issue: Stage 1 model not found for Stage 2
**Solution:** Train Stage 1 first or specify custom path:
```bash
python stage2_train.py --model bert --stage1_model_dir path/to/stage1/model
```

## Advanced Usage

### Custom Learning Rates

```bash
# Stage 1: Higher learning rate for faster adaptation
python stage1_train.py --model bert --learning_rate 5e-5

# Stage 2: Lower learning rate for fine-tuning
python stage2_train.py --model bert --from_stage1 --learning_rate 1e-5
```

### More Training Epochs

```bash
# Stage 1: More epochs to learn incomplete reasoning
python stage1_train.py --model bert --epochs 5

# Stage 2: Fewer epochs to avoid overfitting
python stage2_train.py --model bert --from_stage1 --epochs 2
```

## Contact

For issues or questions, please refer to the main README or open an issue.
