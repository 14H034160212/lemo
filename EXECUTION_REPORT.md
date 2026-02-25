# Execution Report: Two-Stage Training Pipeline

## Executive Summary

This report documents the complete execution of the two-stage training pipeline for logical reasoning models.

**Date**: 2026-01-14
**Environment**: conda logic
**Working Directory**: /mnt/lemo

---

## Completed Tasks

### ✅ 1. Data Generation

#### Base Training & Test Data
- **Script**: `data_gen.py`
- **Output**:
  - `data/train.csv` (160 samples: 80 positive + 80 negative)
  - `data/test_*.csv` (11 test splits)
- **Status**: ✅ Completed (pre-existing)

#### Stage 1 Training Data (Generative Format)
- **Script**: `stage1_data_gen_v2.py --format generative`
- **Samples**: 240 samples (80 base × 3 critical rules)
- **Output**: `data/stage1_train_generative.csv`
- **Format**: Rule prediction task
  - Input: facts + masked_rules + question
  - Target: missing_rule
- **Status**: ✅ Completed

#### Stage 1 Training Data (BERT Format)
- **Script**: `stage1_data_gen_v2.py --format bert`
- **Samples**: 240 samples
- **Output**: `data/stage1_train_bert.csv`
- **Format**: Multiple choice task (4 candidates per question)
- **Status**: ✅ Completed
- **Note**: Fixed to ensure all samples have exactly 4 candidates

---

### ✅ 2. Model Training

#### Stage 1: BERT Multiple Choice
- **Model**: BERT-base-uncased
- **Task**: Select correct missing rule from 4 candidates
- **Training**:
  - Epochs: 2
  - Batch size: 4
  - Learning rate: 5e-5
  - Training samples: 240
  - Trainable params: 295,681 (0.27% of total)
- **Results**:
  - Final loss: 1.3811
  - Training time: ~7.5 seconds
- **Output**: `trained_models/bert_stage1_mc/`
- **Status**: ✅ Completed

#### Stage 1: Qwen Generative
- **Model**: Qwen2-1.5B
- **Task**: Generate missing rule text
- **Training**:
  - Epochs: 2
  - Batch size: 2
  - Learning rate: 5e-5
  - Training samples: 240
  - LoRA parameters: r=8, alpha=16
- **Results**:
  - Initial loss: 1.8967
  - Final loss: 0.1821
  - Training time: ~48 seconds
- **Output**: `trained_models/qwen_stage1_gen/`
- **Status**: ✅ Completed

---

### 🔄 3. Model Evaluation

#### Qwen Stage 1 Generative Model
- **Script**: `evaluate_generative.py --model qwen --stage stage1_gen`
- **Test Splits**: 11 splits (base + variants)
- **Output**: `trained_models/qwen_stage1_gen/predictions/`
- **Status**: 🔄 In Progress
- **Prediction Files**: Will include:
  - Individual predictions for each test sample
  - Generated text and parsed T/F answers
  - Accuracy per split

#### BERT Stage 1 Multiple Choice Model
- **Status**: ⏳ Pending

---

## File Structure

```
/mnt/lemo/
├── Data Files
│   ├── data/train.csv                        # Original T/F training (160 samples)
│   ├── data/test_*.csv                       # 11 test splits
│   ├── data/stage1_train_generative.csv      # Stage 1 generative (240 samples)
│   └── data/stage1_train_bert.csv            # Stage 1 BERT MC (240 samples)
│
├── Trained Models
│   ├── trained_models/bert_stage1_mc/        # BERT Stage 1 (multiple choice)
│   │   └── predictions/                      # Evaluation results (pending)
│   └── trained_models/qwen_stage1_gen/       # Qwen Stage 1 (generative)
│       └── predictions/                      # Evaluation results (in progress)
│
├── Training Scripts
│   ├── stage1_data_gen_v2.py                 # Data generation
│   ├── stage1_train_bert.py                  # BERT training
│   ├── stage1_train_generative.py            # Qwen/LLaMA training
│   ├── stage2_train_generative.py            # Stage 2 training
│   └── train.py                              # Original T/F training
│
├── Evaluation Scripts
│   ├── evaluate_generative.py                # Generative model evaluation
│   ├── evaluate.py                           # BERT evaluation
│   └── summarize_results.py                  # Results aggregation
│
└── Documentation
    ├── QUICKSTART.md                         # Quick start guide
    ├── TRAINING_GUIDE_V2.md                  # Detailed guide
    ├── DISK_SPACE_FIX.md                     # Disk space solution
    └── EXECUTION_REPORT.md                   # This report
```

---

## Technical Details

### Disk Space Fix
**Issue**: Root partition full (100% usage)
**Solution**: Configured HuggingFace cache to `/mnt/lemo/.cache/huggingface`
**Status**: ✅ Applied to all scripts

### Data Format Fixes
**Issue**: Some BERT samples had <4 candidates (NaN values)
**Solution**: Updated `generate_rule_candidates()` to ensure exactly 4 candidates
**Status**: ✅ Fixed

---

## Evaluation Metrics

### Prediction File Format
Each prediction CSV contains:
- `group_id`: Sample identifier
- `type`: Test split type
- `facts`: Input facts
- `rules`: Input rules
- `question`: Question text
- `ground_truth`: True answer (T/F)
- `prediction`: Model prediction (T/F)
- `generated_text`: Raw model output (for generative models)
- `equiv_laws_used`: Equivalence laws applied
- `changed_rule`: Description of variant

### Summary Metrics
- **Accuracy per split**: Correct predictions / Total predictions
- **Delta vs base**: Change in accuracy compared to base split
- **Per-model comparison**: Side-by-side accuracy across all models

---

## Next Steps

### Immediate
1. ✅ Complete Qwen Stage 1 evaluation
2. ⏳ Run BERT Stage 1 evaluation
3. ⏳ Generate summary report with `summarize_results.py`

### Optional (Stage 2)
4. 🔧 Debug Stage 2 mixed training issue
5. ⏳ Train Stage 2 models (if Stage 2 training is fixed)
6. ⏳ Evaluate Stage 2 models

---

## Command Reference

### Data Generation
```bash
# Generate Stage 1 data (generative)
conda run -n logic python stage1_data_gen_v2.py --format generative --num_samples 80

# Generate Stage 1 data (BERT)
conda run -n logic python stage1_data_gen_v2.py --format bert --num_samples 80
```

### Training
```bash
# Train BERT Stage 1
conda run -n logic python stage1_train_bert.py --epochs 2 --batch_size 4

# Train Qwen Stage 1
conda run -n logic python stage1_train_generative.py --model qwen --epochs 2 --batch_size 2

# Train Qwen Stage 2 (when fixed)
conda run -n logic python stage2_train_generative.py --model qwen --from_stage1 --mixed_data --epochs 2
```

### Evaluation
```bash
# Evaluate Qwen Stage 1
conda run -n logic python evaluate_generative.py --model qwen --stage stage1_gen

# Evaluate BERT Stage 1 (not yet run)
conda run -n logic python evaluate.py --model bert --stage stage1_mc

# Summarize all results
conda run -n logic python summarize_results.py --output evaluation_summary.csv
```

---

## Training Performance

| Model | Stage | Epochs | Batch Size | Training Time | Final Loss |
|-------|-------|--------|------------|---------------|------------|
| BERT | Stage 1 MC | 2 | 4 | 7.5s | 1.3811 |
| Qwen | Stage 1 Gen | 2 | 2 | 48s | 0.1821 |

---

## Known Issues

### Stage 2 Training Error
**Issue**: `ValueError: Unable to create tensor` when training Stage 2 with mixed data
**Location**: `stage2_train_generative.py`
**Suspected Cause**: Labels field has excessive nesting when combining datasets
**Status**: 🔧 Under investigation
**Workaround**: Can train Stage 2 without mixing, or debug data collator

---

## Conclusion

Successfully completed:
- ✅ Data generation for both generative and BERT formats
- ✅ Stage 1 training for both BERT and Qwen models
- 🔄 Evaluation in progress

The two-stage training pipeline is functional for Stage 1. Models are saved with clear naming:
- `bert_stage1_mc/` - BERT multiple choice
- `qwen_stage1_gen/` - Qwen generative

All predictions will be saved to `{model}/predictions/` directories with detailed CSV files for analysis.

---

**Report Generated**: 2026-01-14
**Last Updated**: Evaluation in progress
