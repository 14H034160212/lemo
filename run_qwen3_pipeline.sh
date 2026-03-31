#!/bin/bash
# Auto-evaluate Qwen3 RLVF when training finishes
QWEN3_PID=1337400
PYTHON="/data/qbao775/miniconda3/envs/lemo/bin/python"
LOG="logs/qwen3_pipeline.log"

mkdir -p logs results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Waiting for Qwen3 RLVF (PID $QWEN3_PID)..."
while kill -0 $QWEN3_PID 2>/dev/null; do
    PROGRESS=$(tail -3 logs/rlvf_qwen3.log 2>/dev/null | grep -oP '\d+/\d+' | tail -1)
    log "  still training... $PROGRESS"
    sleep 1200
done
log "Qwen3 training done!"

# Evaluate
log "Starting Qwen3 evaluation..."
CUDA_VISIBLE_DEVICES=7 $PYTHON -u evaluate.py \
    --model qwen3 \
    --model_dir trained_models/qwen3_rlvf \
    > logs/qwen3_eval.log 2>&1
log "Qwen3 evaluation done!"

# Print comparison
$PYTHON -c "
import pandas as pd
models = {
    'Qwen2-SFT':  'trained_models/qwen/accuracy_summary.csv',
    'Qwen2-RLVF': 'trained_models/qwen_rlvf/accuracy_summary.csv',
    'Qwen3-RLVF': 'trained_models/qwen3_rlvf/accuracy_summary.csv',
}
for name, path in models.items():
    try:
        df = pd.read_csv(path).set_index('split')['accuracy']
        v4 = [s for s in df.index if 'variant4' in s]
        print(f'{name}: base={df.get(\"base\",\"?\"):.4f} v2={df.get(\"variant2\",\"?\"):.4f} v4_avg={df[v4].mean():.4f} overall={df.mean():.4f}')
    except: print(f'{name}: not found')
" | tee -a results/qwen3_comparison.txt

log "All done! Results in results/qwen3_comparison.txt"
