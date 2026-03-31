#!/bin/bash
# Master experiment runner
# Usage: bash run_experiments.sh [bert|qwen|llama|all]

set -e
cd /data/qbao775/lemo

export HF_HOME=/data/qbao775/lemo/.cache/huggingface
export HF_DATASETS_CACHE=/data/qbao775/lemo/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data/qbao775/lemo/.cache/huggingface/transformers
export CUDA_VISIBLE_DEVICES=0

CONDA_RUN="conda run -n lemo --no-capture-output"
TARGET=${1:-all}

run_model() {
    local model=$1
    echo "============================================"
    echo "  TRAINING: $model"
    echo "============================================"
    $CONDA_RUN python train.py --model $model

    echo "============================================"
    echo "  EVALUATING: $model"
    echo "============================================"
    $CONDA_RUN python evaluate.py --model $model
}

if [[ "$TARGET" == "all" ]]; then
    run_model bert
    run_model qwen
    run_model llama
else
    run_model $TARGET
fi

echo ""
echo "========== ALL EXPERIMENTS DONE =========="
echo "Results are in trained_models/<model>/accuracy_summary.csv"
