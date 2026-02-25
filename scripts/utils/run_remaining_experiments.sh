#!/bin/bash
set -e

echo "Starting Method 2: CoT Training..."
python stage2_train_cot.py --model qwen --from_stage1 --stage1_model_dir ./trained_models/qwen_stage1_gen --original_data data/train_cot.csv --output_dir trained_models/qwen_stage2_cot --epochs 3

echo "Evaluating Method 2: CoT..."
python evaluate_cot.py --model_dir trained_models/qwen_stage2_cot

echo "Starting Method 3: DPO Training..."
python stage2_train_dpo.py

echo "Evaluating Method 3: DPO..."
python evaluate_generative.py --model qwen --model_dir trained_models/qwen_stage2_dpo

echo "All remaining experiments finished!"
