# Reproducibility Record

Status of every table in the paper draft against what this repository can
actually reproduce, as established by a full audit in September 2026.

**Headline finding: the benchmark was regenerated on 2026-03-31, and the
paper's headline numbers predate it.** Anything measured before that date was
measured on a different dataset with different label semantics. This file
records which results survive that change and which do not.

---

## 1. The two benchmark versions

| | v1 (original) | v2 (current, in `data/`) |
|---|---|---|
| Introduced | commit `ea27058`, 2026-02-25 | commit `ba71ff0`, 2026-03-31 ("expand datasets 10x") |
| `test_variant3.csv` | 100 rows / 400 questions | 1000 rows / 4000 questions |
| Variant-3 labels | **F = 100%** | **F = 65.6%, T = 34.4%** |

The label change is the decisive one. Under the conservative contradiction
semantics stated in the paper (§3.1: "if `Γ ⊢ ⊥` then `Label(q) = False` for
every `q ∈ Q`"), *every* Variant-3 query must be False. The v1 file satisfies
that. The current file does not.

This single fact explains the paper's headline "collapse to 0.0000" on
Variant 3: on an all-False file, a model that answers True everywhere scores
exactly 0.0000 per question. On the current file the same model scores ≈0.344,
and a model that answers False everywhere scores 0.656.

There is a third dataset, `data_v2/`, which is a *different* axis of expansion
(five logical domains, variable chain length, branching rules — built to answer
the reviewer critique "benchmark too small and narrow"). It is not the same
thing as the v1→current change above. See §4.

## 2. Per-table reproducibility status

| Paper table | Status | Notes |
|---|---|---|
| **Table 1** (baseline Logic Inertia) | ❌ **does not reproduce** | Numbers come from `results/evaluation_summary.csv` (2026-03-23), i.e. the v1 benchmark. See §3. |
| **Table 2** (method comparison) | ⚠️ **partially reproduces** | 5 of 8 rows have no on-disk provenance and their checkpoints were deleted. See §5. |
| **Table 3** (stage-wise ablation) | ✅ **reproduces exactly** | All six rows match `trained_models/*/accuracy_summary.csv` to 3 decimal places on the current benchmark. |
| **Table 4** (OOD transfer) | ❌ **corrected** | Original LogicNLI/MNLI figures were produced against a mislabelled eval file. Corrected in commit `66edbf8`; see `README.md` §7.3. |
| **Table 5** (Lean verification) | not re-audited | |

## 3. Table 1 — measured on the current benchmark

The paper reports that untreated models hold at 1.000 on base/Variant 1, fall
to ~0.25 on Variant 2, and collapse to 0.0000 on Variant 3. Re-measuring the
same checkpoints on the current benchmark:

| Model | split | paper | current benchmark |
|---|---|---|---|
| BERT | variant2 / variant3 | 0.295 / **0.000** | **0.879 / 0.921** |
| Qwen2-1.5B | variant2 / variant3 | 0.250 / **0.000** | **1.000 / 1.000** |
| TinyLlama-1.1B | variant2 / variant3 | 0.250 / **0.000** | **0.977 / 0.505** |

The direction reverses. On the current benchmark, untreated Qwen2-1.5B is at
ceiling on contradiction injection — there is no collapse to repair.

**Consequence:** the phenomenon the paper names *Logic Inertia* is not
observable on the current benchmark as constructed. Before this work is
submitted anywhere, the benchmark's contradiction semantics needs to be
reconciled with §3.1 — either the generator should re-impose the all-False
override, or the paper's semantics and metric need to change to match the data.

## 4. `data_v2/` — the five-domain benchmark

`data_gen_v2.py` builds a broader benchmark (5 domains, 2–4 hop chains,
branching rules, 2500 base groups) explicitly to answer the "benchmark too
narrow" critique. Measured (classification path, unaffected by the generation
bugs in §6):

| Model | `data/` | `data_v2/` |
|---|---|---|
| Stage-1 SFT (1.5B) | 1.000 / 1.000 / 1.000 | **0.605 / 0.553 / 0.710** |
| + LIRE (1.5B) | 0.812 / 0.590 / 0.344 | 0.687 / 0.511 / 0.509 |
| Fusion-Conflict (1.5B) | 1.000 / 1.000 / 1.000 | **0.613 / 0.523 / 0.694** |
| Fusion-Conflict (8B) | 1.000 / 1.000 / 1.000 | **0.557 / 0.562 / 0.373** |

(base / variant2 / variant3.)

On `data_v2/`, Stage-1 SFT and the full pipeline are indistinguishable — the
DPO/LIRE/RLVF stages buy no measurable gain. This is a material limitation and
should be reported alongside any claim of saturation.

## 5. Table 2 — provenance of each row

Searched every result artefact on disk for each reported (base, V2, V3) triple:

| Row | Provenance | Checkpoint |
|---|---|---|
| Fusion-Conflict (1.5B / 8B) | ✅ matches `all_models_comparison.csv` exactly | `qwen_rlvf`, `qwen3_rlvf` |
| Mixed-Aug | ✅ matches `evaluation_summary.csv::qwen_stage2_mixed` | deleted |
| Stage-1 SFT, DPO, CoT, RA-CoT, Fusion-LRA | ❌ no matching artefact | deleted |

The five checkpoints were retrained from the repository scripts in September
2026 (`qwen_stage1_gen`, `qwen_stage2_dpo`, `qwen_stage2_mixed`,
`qwen_stage2_ra_cot`, `qwen_fusion_sft_conflict_aware`) and re-evaluated on the
current benchmark. Retraining `qwen_stage1_gen` first is required: it is the
initialisation for the DPO / Mixed-Aug / RA-CoT branches and had been deleted.

Result, with the fraction of generations that actually contain a parseable
`Answer:` line — without which `parse_answer()` falls through to its `"F"`
default and the reported accuracy is just the proportion of F labels:

| Model | base | V2 | V3 | parse rate | usable |
|---|---|---|---|---|---|
| Fusion-LRA | 0.858 | 0.581 | 0.654 | 0.99–1.00 | ✅ |
| CoT | 0.490 | 0.528 | — | 0.81–0.86 | ✅ (V3 run timed out) |
| RA-CoT | 0.522 | 0.484 | 0.583 | 0.59–0.74 | ⚠️ partial fallback |
| Stage-1 SFT | 0.472 | 0.497 | 0.635 | 0.24–0.30 | ❌ |
| DPO | 0.466 | 0.489 | 0.637 | 0.28–0.32 | ❌ |
| Mixed-Aug | 0.464 | 0.494 | 0.638 | 0.19–0.20 | ❌ |

The three unusable rows are a **task mismatch**, not a tuning problem: their
training targets never contain an `Answer:` line.
`stage1_train_generative.csv` targets are the *missing rule* (a rule-completion
task); `train_mixed.csv` has no `target_text` column at all; `train_dpo.jsonl`
pairs are bare `"True"`/`"False"` strings. Evaluating them with
`evaluate_generative.py`, which expects `Answer: True/False`, cannot work.
Confirmed by running `qwen_stage1_gen` at `--max_new_tokens 1024`: the parse
rate stays at 0.05.

Note also that Table 2 mixes two evaluation paths — Fusion-Conflict is a
sequence-classification head choosing between two logits, while the baselines
must generate and parse free text. These are not equally hard, and the table
should say which path each row used.

## 6. Bugs found and fixed during the audit

| Bug | Effect | Fix |
|---|---|---|
| `prepare_real_world_data.py` looked LogicNLI's *string* labels up in an int-keyed dict with a `"False"` default | all 500 rows silently labelled False; a constant-False predictor scores 1.000 | raises on an unmapped label; `logicnli_eval_fixed.csv` added (commit `66edbf8`) |
| `evaluate_generative.py` generated one prompt at a time | ~18 s per generation; a full split took hours | batched left-padded generation, `--batch_size` (default 32) — ~150× faster, bit-identical greedy output |
| `max_new_tokens` too low | models restate every rule before concluding, so 94–96% of traces were cut off before the `Answer:` line and fell through to the `"F"` default | raised to 384; **verified by parse rate, not by accuracy** — the truncated run looked self-consistent across batch sizes precisely because every setting was equally broken |
| HF cache pointed at `/mnt/lemo` (14 scripts) | permission error on any machine without that mount | repo-relative `.cache/huggingface` |
| `trained_models/qwen3_lire` missing `vocab.json` / `merges.txt` | tokenizer fell back to a slow path requiring protobuf, which is not installed | copied from `trained_models/qwen3_rlvf` (same base model) |
| no way to evaluate a subset of splits | forced a 15,600-row / 11-split run to see one column | `--splits`, `--max_rows` |

**Lesson worth keeping:** the truncation bug produced accuracies that were
stable across batch sizes and therefore looked verified. They were identical
because every configuration was truncated the same way. Any generative
evaluation in this repo should report the parse rate next to the accuracy;
`evaluate_generative.py` now makes this checkable from the prediction CSVs.

## 7. How to re-run

```bash
# Stage-1 generative model (prerequisite for the DPO / Mixed-Aug / RA-CoT branches)
python scripts/training/stage1_train_generative.py --model qwen \
  --train_data data/stage1_train_generative.csv \
  --output_dir trained_models/qwen_stage1_gen

# Generative evaluation (always check the parse rate in the prediction CSV)
python scripts/evaluation/evaluate_generative.py --model qwen \
  --model_dir trained_models/<model> \
  --splits base variant2 variant3 --batch_size 128 --max_new_tokens 384

# Classification evaluation, current benchmark
python evaluate.py --model qwen --model_dir trained_models/<model>

# Classification evaluation, five-domain benchmark
python evaluate.py --model qwen --model_dir trained_models/<model> \
  --data_dir data_v2 --output_suffix _v2
```

Raw per-row predictions live in `trained_models/*/predictions/` and
`results/`. The out-of-distribution transfer re-run is
`scripts/evaluation/evaluate_real_world_v2.py`; its output is
`results/real_world_rerun_summary.json`.
