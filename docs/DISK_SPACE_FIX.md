# Disk Space Issue - Fixed

## Problem

The root partition `/` was full (100% usage), causing HuggingFace cache errors:
```
OSError: Not enough disk space
```

## Solution Applied

All training and evaluation scripts have been updated to use `/mnt/lemo/.cache/huggingface` instead of the default `/root/.cache/huggingface`.

### What Changed

Added to all Python scripts:
```python
import os

# Set HuggingFace cache to avoid disk space issues
os.environ['HF_HOME'] = '/mnt/lemo/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/lemo/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/lemo/.cache/huggingface/transformers'
```

### Modified Scripts

✅ **Training Scripts:**
- `train.py`
- `stage1_train.py`
- `stage2_train.py`
- `stage1_train_bert.py`
- `stage1_train_generative.py`
- `stage2_train_generative.py`

✅ **Evaluation Scripts:**
- `evaluate.py`
- `evaluate_generative.py`

## Disk Space Status

```bash
# Root partition (was full)
/dev/vda3    97G   88G     0 100% /

# Data partition (has space)
/dev/vdb    393G  283G   90G  76% /mnt
```

**HuggingFace cache now uses:** `/mnt/lemo/.cache/huggingface` ✅

## How to Use

Simply run the scripts as normal - no additional configuration needed:

```bash
# This will now use /mnt/lemo/.cache instead of /root/.cache
python stage1_train_bert.py --epochs 3
python stage1_train_generative.py --model qwen --epochs 3
```

## Optional: Clean Up Old Cache (if needed)

If you want to reclaim space from the old cache:

```bash
# Check old cache size
du -sh /root/.cache/huggingface

# Remove old cache (CAUTION: only if you're sure)
# rm -rf /root/.cache/huggingface
```

**Note:** Only do this if you're certain you don't need the old cached models.

## Verify Cache Location

To verify the cache is being used correctly:

```bash
# Check new cache directory
ls -lh /mnt/lemo/.cache/huggingface/

# Check disk usage
du -sh /mnt/lemo/.cache/huggingface
```

## Manual Override (Optional)

If you want to manually set the cache directory for a single session:

```bash
export HF_HOME=/mnt/lemo/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/lemo/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/lemo/.cache/huggingface/transformers

python your_script.py
```

## Summary

✅ **Problem Fixed**: All scripts now use `/mnt/lemo/.cache/huggingface`
✅ **No Action Needed**: Just run scripts normally
✅ **Space Available**: 90GB free on `/mnt` partition

You can now proceed with training without disk space errors!
