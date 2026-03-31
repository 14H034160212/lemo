#!/bin/bash
export HF_HOME=/data/qbao775/lemo/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
PY=/data/qbao775/miniconda3/envs/lemo/bin/python
LOG_DIR=logs
mkdir -p $LOG_DIR

step() { echo; echo "============================================================"; echo "STEP $1: $2"; echo "============================================================"; }

step 1 "Retrain BERT with diverse variant2/3 data"
$PY train.py --model bert 2>&1 | tee $LOG_DIR/retrain_bert.log

step 2 "Retrain Qwen with diverse variant2/3 data"
$PY train.py --model qwen 2>&1 | tee $LOG_DIR/retrain_qwen.log

step 3 "Retrain TinyLlama with diverse variant2/3 data"
$PY train.py --model llama 2>&1 | tee $LOG_DIR/retrain_llama.log

step 4 "Evaluate BERT"
$PY evaluate.py --model bert 2>&1 | tee $LOG_DIR/eval_bert_final.log

step 5 "Evaluate Qwen"
$PY evaluate.py --model qwen 2>&1 | tee $LOG_DIR/eval_qwen_final.log

step 6 "Evaluate TinyLlama"
$PY evaluate.py --model llama 2>&1 | tee $LOG_DIR/eval_llama_final.log

step 7 "Stage2 CoT training -- Qwen (generative, with CoT reasoning)"
$PY scripts/training/stage2_train_cot.py \
    --model qwen \
    --original_data data/train_cot.csv \
    --output_dir trained_models/qwen_stage2_cot \
    --epochs 3 --batch_size 2 --learning_rate 1e-5 \
    2>&1 | tee $LOG_DIR/stage2_qwen.log

step 8 "Stage2 CoT training -- TinyLlama (generative, with CoT reasoning)"
$PY scripts/training/stage2_train_cot.py \
    --model llama \
    --original_data data/train_cot.csv \
    --output_dir trained_models/llama_stage2_cot \
    --epochs 3 --batch_size 2 --learning_rate 1e-5 \
    2>&1 | tee $LOG_DIR/stage2_llama.log

step 9 "Evaluate Qwen Stage2 CoT (generative)"
$PY scripts/evaluation/evaluate_generative.py \
    --model qwen --model_dir trained_models/qwen_stage2_cot \
    2>&1 | tee $LOG_DIR/eval_qwen_stage2.log

step 10 "Evaluate TinyLlama Stage2 CoT (generative)"
$PY scripts/evaluation/evaluate_generative.py \
    --model llama --model_dir trained_models/llama_stage2_cot \
    2>&1 | tee $LOG_DIR/eval_llama_stage2.log

step 11 "Collect all results"
$PY collect_results.py 2>&1 | tee $LOG_DIR/final_results.log

echo
echo "=== PIPELINE COMPLETE ==="
