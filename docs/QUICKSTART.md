# Quick Start: Two-Stage Training

## TL;DR

**For Generative Models (Qwen/LLaMA) - Complete Pipeline:**

```bash
# 1. Generate data
python data_gen.py
python stage1_data_gen_v2.py --format generative

# 2. Train Stage 1 (rule prediction)
# → Saves to: trained_models/qwen_stage1_gen/
python stage1_train_generative.py --model qwen --epochs 3

# 3. Train Stage 2 (mixed training)
# → Saves to: trained_models/qwen_stage2_mixed/
python stage2_train_generative.py --model qwen --from_stage1 --mixed_data --epochs 2

# 4. Evaluate
python evaluate_generative.py --model qwen --stage stage2_mixed
```

**For BERT (Multiple Choice):**

```bash
# 1. Generate data
python data_gen.py
python stage1_data_gen_v2.py --format bert

# 2. Train Stage 1 (multiple choice)
# → Saves to: trained_models/bert_stage1_mc/
python stage1_train_bert.py --epochs 3

# 3. Train Stage 2 (original T/F task)
# → Saves to: trained_models/bert/
python train.py --model bert

# 4. Evaluate
python evaluate.py --model bert
```

---

## What's Happening?

### Stage 1: Learn to Predict Missing Rules

**Input**: Facts + Incomplete Rules + Question
**Output**: The Missing Rule

Example:
```
Input:
Facts: Bob is green or blue
Rules: [Some rules, but critical "cold → rough" is missing]
Question: Bob is rough?

Output:
"If someone is cold then they are rough."
```

### Stage 2: Learn Both Tasks Together

**Task A (from Stage 1)**: Predict missing rules
**Task B (new)**: Answer True/False questions

**Result**: Model learns robust reasoning even with incomplete information.

---

## Scripts Overview

### Data Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `data_gen.py` | Generate base T/F data | `data/train.csv`, `data/test_*.csv` |
| `stage1_data_gen_v2.py` | Generate rule prediction data | `data/stage1_train_generative.csv` or `data/stage1_train_bert.csv` |

### Training (Generative: Qwen/LLaMA)

| Script | Purpose | Input | Output Model Directory |
|--------|---------|-------|------------------------|
| `stage1_train_generative.py` | **Stage 1**: Rule prediction | `stage1_train_generative.csv` | `trained_models/{model}_stage1_gen/` |
| `stage2_train_generative.py` | **Stage 2**: Mixed training | `train.csv` + `stage1_train_generative.csv` | `trained_models/{model}_stage2_mixed/` (if --mixed_data)<br>`trained_models/{model}_stage2/` (otherwise) |

### Training (BERT)

| Script | Purpose | Input | Output Model Directory |
|--------|---------|-------|------------------------|
| `stage1_train_bert.py` | **Stage 1**: Multiple choice | `stage1_train_bert.csv` | `trained_models/bert_stage1_mc/` |
| `train.py` | **Stage 2**: T/F classification | `train.csv` | `trained_models/bert/` |

### Evaluation

| Script | Purpose | Models |
|--------|---------|--------|
| `evaluate_generative.py` | Evaluate generative models | Qwen, LLaMA |
| `evaluate.py` | Evaluate BERT | BERT |

---

## Key Parameters

### Data Generation

```bash
# Generate more/fewer samples
python stage1_data_gen_v2.py --num_samples 500

# Choose format
python stage1_data_gen_v2.py --format generative  # For Qwen/LLaMA
python stage1_data_gen_v2.py --format bert        # For BERT
```

### Training

```bash
# Adjust epochs
--epochs 5

# Adjust batch size (for GPU memory)
--batch_size 2

# Adjust learning rate
--learning_rate 1e-5

# For Stage 2: Use mixed data
--mixed_data
```

### Evaluation

```bash
# Evaluate specific stage
--stage stage1_gen
--stage stage2_mixed

# Or specify custom directory
--model_dir path/to/model
```

---

## Common Workflows

### 1. Quick Test (Small Data)

```bash
# Generate small datasets
python data_gen.py  # Uses default 100 samples
python stage1_data_gen_v2.py --format generative --num_samples 50

# Quick training
python stage1_train_generative.py --model qwen --epochs 1 --batch_size 2
python stage2_train_generative.py --model qwen --from_stage1 --mixed_data --epochs 1

# Evaluate
python evaluate_generative.py --model qwen --stage stage2_mixed
```

### 2. Full Training (Recommended)

```bash
# Generate full datasets
python data_gen.py
python stage1_data_gen_v2.py --format generative --num_samples 200

# Full training
python stage1_train_generative.py --model qwen --epochs 3 --batch_size 2
python stage2_train_generative.py --model qwen --from_stage1 --mixed_data --epochs 2 --batch_size 2

# Evaluate
python evaluate_generative.py --model qwen --stage stage2_mixed
```

### 3. Compare Different Strategies

```bash
# Strategy 1: Only Stage 1
python stage1_train_generative.py --model qwen --epochs 3
python evaluate_generative.py --model qwen --stage stage1_gen

# Strategy 2: Stage 2 without mixing
python stage2_train_generative.py --model qwen --from_stage1 --epochs 2
python evaluate_generative.py --model qwen --stage stage2

# Strategy 3: Stage 2 with mixing (BEST)
python stage2_train_generative.py --model qwen --from_stage1 --mixed_data --epochs 2
python evaluate_generative.py --model qwen --stage stage2_mixed
```

---

## Expected Training Time

**With GPU (NVIDIA T4):**
- Stage 1 (200 samples, 3 epochs): ~10-15 minutes
- Stage 2 (mixed data, 2 epochs): ~15-20 minutes

**With CPU:**
- Stage 1: ~1-2 hours
- Stage 2: ~2-3 hours

---

## Troubleshooting

**Q: Disk space error?**
A: ✅ **FIXED**: All scripts now use `/mnt/lemo/.cache/huggingface` instead of `/root/.cache`. See `DISK_SPACE_FIX.md` for details.

**Q: CUDA out of memory?**
A: Reduce batch size: `--batch_size 1`

**Q: Model generates nonsense?**
A: Train longer: `--epochs 5` or reduce learning rate: `--learning_rate 1e-5`

**Q: Accuracy is very low?**
A: Make sure you're using `--mixed_data` in Stage 2

**Q: Where are the predictions saved?**
A: In `trained_models/{model}/predictions/` directory

---

## Next Steps

- Read `TRAINING_GUIDE_V2.md` for detailed explanations
- Check `data/` directory for generated data samples
- Explore `trained_models/` for saved models and predictions

---

## File Structure

```
.
├── QUICKSTART.md                    ← You are here
├── TRAINING_GUIDE_V2.md            ← Detailed guide
│
├── # Data Generation
├── data_gen.py                      ← Generate base data
├── stage1_data_gen_v2.py            ← Generate Stage 1 data
│
├── # Training Scripts (Generative: Qwen/LLaMA)
├── stage1_train_generative.py       ← Stage 1 training
├── stage2_train_generative.py       ← Stage 2 training
├── evaluate_generative.py           ← Evaluation
│
├── # Training Scripts (BERT)
├── stage1_train_bert.py             ← Stage 1 training
├── train.py                         ← Stage 2 training
├── evaluate.py                      ← Evaluation
│
├── # Data Files
├── data/
│   ├── train.csv                    ← Original T/F training data
│   ├── stage1_train_generative.csv  ← Stage 1 data (generative)
│   ├── stage1_train_bert.csv        ← Stage 1 data (BERT)
│   └── test_*.csv                   ← Test splits
│
└── # Trained Models (auto-generated)
└── trained_models/
    ├── # Generative Models (Qwen)
    ├── qwen_stage1_gen/             ← Stage 1: Rule prediction
    ├── qwen_stage2/                 ← Stage 2: T/F only
    ├── qwen_stage2_mixed/           ← Stage 2: Mixed (rule + T/F)
    │
    ├── # Generative Models (LLaMA)
    ├── llama_stage1_gen/            ← Stage 1: Rule prediction
    ├── llama_stage2/                ← Stage 2: T/F only
    ├── llama_stage2_mixed/          ← Stage 2: Mixed (rule + T/F)
    │
    └── # BERT
        ├── bert_stage1_mc/          ← Stage 1: Multiple choice
        └── bert/                    ← Stage 2: T/F classification
```
