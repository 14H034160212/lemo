# Reproducibility Record

What this repository can and cannot reproduce, as established by a continuous
audit from 2026-09-08 to 2026-09-21. Every figure below was measured in that
window on the current benchmark with re-trained checkpoints.

**Read this first: most of the paper draft's claims did not survive the audit.**
Three of its four contributions turned out to be artefacts of benchmark
construction, and two arguments the audit itself introduced were later refuted
by its own later measurements. Retractions are listed in §2 with what replaced
them.

---

## 1. Why the original numbers could not be reproduced

The benchmark was regenerated mid-project, and the paper's tables straddle the
change.

| | v1 | v2 (current, in `data/`) |
|---|---|---|
| Introduced | `ea27058`, 2026-02-25 | `ba71ff0`, 2026-03-31 ("expand datasets 10x") |
| `test_variant3.csv` | 100 rows / 400 questions | 1000 rows / 4000 questions |
| Variant-3 labels | F = 100% | F = 65.6%, T = 34.4% |

Table 1 and Table 2 were measured 2026-03-23 (v1); Table 3 was measured
2026-04-07 (v2); the LIRE and RLVF code was written 2026-03-31, eight days
*after* the numbers attributed to it. The original checkpoints were deleted and
the original `train.csv` overwritten, so 2026-03-23 cannot be recovered.

No result file recorded which data it was run against, which is why the mismatch
surfaced six months later by comparing timestamps. `scripts/utils/eval_provenance.py`
now records test-file SHA-256, git commit and checkpoint mtime beside every
summary.

The conservative semantics of §3.1 have been restored in the generator: under
`Γ ⊢ ⊥` every query is False. That makes `test_variant3.csv` single-class, which
is the subject of §3.

## 2. Retracted claims

| Claim | Status | What the data shows |
|---|---|---|
| Untreated models degrade to ~0.25 on Variant 2 | **retracted** | 0.754 / 1.000 / 1.000 on the current benchmark; V2 is not a hard split |
| LIRE improves logical invariance | **retracted** | The gain came from all-True labels. On `variant4_mixed` all three models score exactly 0.600 = the majority baseline, predicting True on 40000/40000 |
| RLVF repairs contradiction handling | **retracted** | 0.0000 on Variant 3 at both 1.5B and 8B, answering True on 4000/4000 |
| Fusion-Conflict = 1.000/1.000/1.000 | **retracted** | That row is a SEQ_CLS head; on the two-class control it scores 0.433, below a constant responder |
| The method scales to 8B | **retracted** | An 8B Fusion-LRA improves base (0.9075) and V2 (0.8023) but *falls* on the control (0.6184 vs 0.698), with the discrimination gap collapsing from 0.420 to 0.081 |
| Verification-first prompting degrades contradiction handling | **retracted** | Measured on the single-class V3, where "answering False less often" registers as a loss. On the two-class control the sign reverses: 0.7767 with the prompt against 0.7633 without |
| Lean 4 agrees with 99.0% of T-derivations | **scope corrected** | Reproduced at 0.9965 on `data_v2` (n=1660, 9x the original sample) but 0.6119 on `data/`, the benchmark every other table uses. The figure is a property of `data_v2` |
| An unbiased oracle reward needs no trust region | **retracted** | Without one the EMA baseline reached +0.695 and the policy degenerated into non-parsing output at step 250. A KL penalty was required |

Two arguments introduced *by the audit* were also refuted by it, and are recorded
here so they are not revived:

- **"A classification head structurally cannot represent contradiction."**
  Proposed to explain why LIRE and RLVF failed while Fusion-LRA worked. Refuted:
  both failures are explained by the reward-oracle defect in §4, which is
  independent of the head type.
- **"Outcome reward is degenerate for premise validation."** Proposed after the
  oracle was fixed and RLVF still collapsed. Refuted by §5: the training
  distribution contains a perfect surface cue, so the collapse is ordinary
  shortcut exploitation, not reward degeneracy.

## 3. Single-class splits, and the controls added

Nine of twelve splits were single-class: `variant1`, `variant3`, and the seven
`variant4_equiv_*`. A constant predictor saturates them. This is not incidental
— when a perturbation deterministically fixes the label, the perturbed split is
single-class by construction.

Two controls were added. Each interleaves the perturbed instances with
consistent ones of identical surface form, so answering one constant scores at
the baseline instead of perfectly.

| Split | Size | Baseline | Purpose |
|---|---|---|---|
| `test_variant4_mixed.csv` | 10000 / 40000 q | 0.600 | equivalence rewrites vs. converse, inverse, De Morgan fallacy, disjunction-to-conjunction |
| `test_variant3_mixed.csv` | 3000 / 12000 q | 0.5835 | contradictory instances vs. consistent ones carrying the same negated sentence |

Results on `variant3_mixed`, over training seeds:

| | accuracy | discrimination gap | n seeds |
|---|---|---|---|
| Fusion-LRA (conflict-aware SFT) | 0.698 ± 0.054 | 0.420 ± 0.200 | 6 |
| \+ verifier-reward RL | 0.583 ± 0.007 | 0.014 ± 0.018 | 4 |
| majority baseline | 0.5835 | 0 | — |

The gap is the consistent-instance True-rate minus the contradictory-instance
True-rate. The two sets of seeds do not overlap (Welch *p*=0.0020;
Mann-Whitney *p*=0.0070, reported because the variances differ by an order of
magnitude). Two of the four RL seeds answer False on all 12000 questions.

**RULEBREAKERS (ICML 2025) reached the same requirement first** and its paired
accuracy is stricter: a constant scores 0 there and 0.5835 here.

## 4. Defects found

Each produced a plausible-looking number. The metric that would have caught it
is named.

| Defect | Effect | Caught by |
|---|---|---|
| LogicNLI string labels looked up in an int-keyed dict with a `"False"` default | all 500 rows labelled False; a constant-False predictor scores 1.000 | majority-class baseline |
| `max_new_tokens` too low | 94–96% of traces cut before `Answer:`, falling to the parser's `"F"` default; accuracy identical across four batch sizes, which looked like verification | answer-parse rate |
| `v3_rows[:320]` hardcoded for a 160-row corpus | contradictions diluted 14.3% → 1.0%; Variant 3 fell 0.982 → 0.654 | class ratio in the training corpus |
| `--output_suffix` applied to summaries but not prediction CSVs | a second benchmark's run silently overwrote the first's predictions | file provenance |
| `disable_adapter()` returns the raw base model, not the SFT checkpoint | KL measured against an un-finetuned reference | logging KL |
| REINFORCE with no trust region | policy degenerated to non-parsing output at step 250 | parse rate during training |
| `oracle_answer()` omitted the contradiction override | oracle said True on 1200/1200 sampled Variant-3 questions where the benchmark says False — reward and evaluation exactly opposed | comparing oracle output to stored labels |
| `parse_rules` dropped `"If someone is not X then they are Y"` | 4042 instances; on rows containing it oracle/label agreement was 0.2500 against 1.0000 elsewhere | same comparison, run per split |
| `parse_rules` bound `\w+` to the bare word `"not"` | fabricated an implication over an attribute named `"not"` | same |
| Lean `DATA_DIR` hardcoded to `data_v2` | Table 5 measured a different dataset than Tables 1–4 | provenance |
| Lean translator matched only the impersonal rule phrasing | 0/150 rows of `data/` parsed | parse rate |
| `stage2_train_fusion.py` had no seed argument | every run used HF's default 42; multi-seed replication was impossible | attempting it |
| `stage4` seeded prompt order but not torch | rollouts varied yet were irreproducible | attempting to reproduce |

## 5. The training set contains the shortcut

The controls in §3 audit the test set. Auditing the training set changes the
interpretation of everything above.

In the training distribution, an instance whose facts contain a negation has
answer `False` with probability **1.000** (792/792 sampled); instances without
one are False 38.2% of the time. Contradiction injection is the only
perturbation that adds a negated fact, so "a negation is present" and "the
premises are inconsistent" are perfectly confounded.

Both objectives partly key on the cue. It explains the RL collapse without
appealing to reward degeneracy, and it explains why Fusion-LRA, which does
discriminate, is weakest where the cue misleads.

The standard repair — adding consistent-but-negated counterexamples — was
attempted and failed informatively: the hardest class rose 0.550 → 0.969 while
contradiction detection fell 0.861 → 0.444, and the retrained model emitted
"no conflict" on 100% of inputs, never once saying "conflict detected". The two
trace templates are separable by surface form, so the model exchanged one cue
for another.

**A two-class control on the test set is necessary but not sufficient when the
training distribution contains the shortcut.**

## 6. Out-of-distribution transfer

Measured on the generation path with two controls: a forced read-out (free
generation states an explicit answer on only 7–46% of out-of-distribution items,
and the parser's `"F"` default manufactures the constant-False signature being
tested for), and an untrained backbone.

| Dataset | untrained backbone | Fusion-LRA | Δ | paired *p* | baseline |
|---|---|---|---|---|---|
| LogicNLI (n=500) | 0.396 | 0.606 | +0.210 | 1.2e-9 | 0.750 |
| MNLI-contradiction (n=349) | 0.722 | 0.874 | +0.152 | 1.2e-7 | 0.501 |
| FOLIO (n=811, first-order) | 0.677 | 0.658 | −0.019 | 0.40 n.s. | 0.567 |

Training helps on premise-consistency tasks and does nothing on first-order
reasoning. LogicNLI's +0.210 still leaves 0.606 below its 0.750 majority
baseline.

## 7. Frontier models

On `variant3_mixed`, 600 questions per configuration, zero failed calls:

| Model | effort | accuracy | output tok/question |
|---|---|---|---|
| gpt-5-mini | medium | 0.900 | 168.3 |
| gpt-5.2 | high | 0.797 | 85.3 |
| gpt-5.2 | default | 0.742 | 2.5 |
| Fusion-LRA (1.5B) | — | 0.698 | ~51 |
| gpt-4o-mini | default | 0.607 | 2.0 |

GPT-5.2 with no reasoning tokens is more accurate than the trained 1.5B model at
one twentieth the output. **A token-efficiency argument for the structural prior
is not supported**, and a depth-scaled benchmark confirms the mechanism it would
rest on does not exist: cost is flat from 2-hop to 16-hop chains (2.5 tokens at
both ends).

### Matched inference forms

`test_mt_control3.csv` generates each instance in three logically equivalent
arms sharing gold answers, differing only in the third rule. 150 instances per
arm, zero failed calls.

| Model | effort | mp (forward) | mp_neg (negation) | mt (contraposition) |
|---|---|---|---|---|
| gpt-5.2 | medium | 0.940 | **1.000** | **0.000** |
| gpt-5-mini | default | 1.000 | 1.000 | 0.571 |
| gpt-5.2 | default | 0.747 | 0.500 | 0.000 |

GPT-5.2 at medium effort applies a rule with a negated premise perfectly and a
contrapositive not once in 150 attempts, with upstream steps at 0.940. The
failure is contraposition, not negation, and it is total rather than degraded.
Q4 is wrong given Q3 wrong in 96–100% of cases, so roughly half the error on the
`mt` arm is one capability gap counted twice.

Breadth is incomplete: API credits were exhausted, leaving two model families
with clean three-arm data.

## 8. Unresolved

- **RLVF-Lean does not exist.** No checkpoint, no training script; the training
  code never calls `lean_reward`. The paper presents it as an extension.
- **`SR_macro`**, defined in §3.1, is not implemented anywhere.
- **Lean's F-side** agrees on 0.2213 of non-derivable queries (`variant2`: 0/293).
  Proving non-derivability needs the closed-world assumption encoded; the
  translator does not do it.
- **ZebraLogic** was last evaluated 2026-07-25, before the audit, at 0.500 for
  all three models on 626 balanced probes — a constant predictor's score.
- **Single-seed results.** Only `variant3_mixed` has multi-seed data (n=6 SFT,
  n=4 RL). Everything else in this file is one run.
- **`variant3_mixed` under Lean** hung the runner for six hours with no output
  and is not diagnosed.

## 9. Re-running

```bash
# benchmark and controls
python data_gen.py
python scripts/data_generation/make_variant3_mixed.py
python scripts/data_generation/make_mt_control.py --pairs 300 --out data/test_mt_control3.csv

# conflict-aware SFT (--seed controls LoRA init and data order)
python scripts/training/stage2_train_fusion.py \
  --train_file data/train_fusion.csv --output_dir trained_models/<name> --seed 42

# verifier-reward RL, with the contradiction override and answer balancing
python scripts/training/stage4_train_rlvf_generative.py \
  --balance_answers --kl_beta 0.05 --seed 0 --output_dir trained_models/<name>

# generative evaluation -- always read the parse rate beside the accuracy
python scripts/evaluation/evaluate_generative.py --model qwen \
  --model_dir trained_models/<name> --splits variant3_mixed base variant2 \
  --batch_size 64 --max_new_tokens 384

# transfer, with both controls
python scripts/evaluation/evaluate_ood_generative.py \
  --model_dir trained_models/<name> --forced_answer
python scripts/evaluation/evaluate_ood_generative.py \
  --base_only Qwen/Qwen2-1.5B --forced_answer

# Lean; LEMO_DATA_DIR selects the dataset and names the output after it
LEMO_DATA_DIR=data python lean_demo/run_benchmark_lean_eval.py

# label balance across public benchmarks
python scripts/analysis/survey_label_balance.py
```

Per-row predictions are under `trained_models/*/predictions/`,
`results/ood_generative/` and `results/frontier_cost/`. Every summary has a
`.provenance.json` beside it.
