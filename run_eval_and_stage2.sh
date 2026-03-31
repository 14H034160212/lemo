#!/bin/bash
export HF_HOME=/data/qbao775/lemo/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
PY=/data/qbao775/miniconda3/envs/lemo/bin/python
LOG_DIR=logs
mkdir -p $LOG_DIR

step() { echo; echo "============================================================"; echo "STEP $1: $2"; echo "============================================================"; }

step 1 "Re-evaluate BERT (classification, fresh predictions)"
$PY evaluate.py --model bert 2>&1 | tee $LOG_DIR/eval_bert_v2.log

step 2 "Re-evaluate Qwen (classification, fresh predictions)"
$PY evaluate.py --model qwen 2>&1 | tee $LOG_DIR/eval_qwen_v2.log

step 3 "Re-evaluate TinyLlama (classification, fresh predictions)"
$PY evaluate.py --model llama 2>&1 | tee $LOG_DIR/eval_llama_v2.log

step 4 "Stage2 CoT generative training -- Qwen"
$PY scripts/training/stage2_train_cot.py \
    --model qwen \
    --original_data data/train_cot.csv \
    --output_dir trained_models/qwen_stage2_cot \
    --epochs 3 --batch_size 2 --learning_rate 1e-5 \
    2>&1 | tee $LOG_DIR/train_qwen_stage2_cot_v2.log

step 5 "Stage2 CoT generative training -- TinyLlama"
$PY scripts/training/stage2_train_cot.py \
    --model llama \
    --original_data data/train_cot.csv \
    --output_dir trained_models/llama_stage2_cot \
    --epochs 3 --batch_size 2 --learning_rate 1e-5 \
    2>&1 | tee $LOG_DIR/train_llama_stage2_cot_v2.log

step 6 "Evaluate Qwen Stage2 CoT (generative)"
$PY scripts/evaluation/evaluate_generative.py \
    --model qwen --model_dir trained_models/qwen_stage2_cot \
    2>&1 | tee $LOG_DIR/eval_qwen_stage2_cot_v2.log

step 7 "Evaluate TinyLlama Stage2 CoT (generative)"
$PY scripts/evaluation/evaluate_generative.py \
    --model llama --model_dir trained_models/llama_stage2_cot \
    2>&1 | tee $LOG_DIR/eval_llama_stage2_cot_v2.log

step 8 "Collect and compare all results"
$PY collect_results.py 2>&1 | tee $LOG_DIR/collect_results_v2.log

echo
echo "All done!"
