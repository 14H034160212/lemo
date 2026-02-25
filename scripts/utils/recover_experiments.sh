#!/bin/bash
set -e

# ensure cache is on large disk
export HF_HOME='/mnt/lemo/.cache/huggingface'
export HF_DATASETS_CACHE='/mnt/lemo/.cache/huggingface/datasets'

echo "Resuming experiments..."

if [ -f "trained_models/qwen_stage2_cot/adapter_config.json" ]; then
    echo "Evaluating Method 2: CoT..."
    python evaluate_cot.py --model_dir trained_models/qwen_stage2_cot
else
    echo "Method 2 model not found! Creating..."
    # Should not happen if training succeeded
    exit 1
fi

echo "Starting Method 3: DPO Training..."
python stage2_train_dpo.py

echo "Evaluating Method 3: DPO..."
python evaluate_generative.py --model qwen --model_dir trained_models/qwen_stage2_dpo

echo "All recovered experiments finished!"
