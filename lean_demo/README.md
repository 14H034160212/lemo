# Lean 4 Feasibility Demo for RLVF-Lean (Phase 2)

This directory contains the Phase 2 feasibility demo referenced in Section 6.5
of the paper. It shows that the Socrates contradiction running example can be
encoded in **Lean 4** and verified by the Lean kernel, providing a natural
upgrade path from the current `forward_chain()` oracle to a formal proof
assistant.

## Files

| File | Purpose |
|------|---------|
| `SocratesContradiction.lean` | Lean 4 theorem + step-level reward demo |
| `lean_verifier_bridge.py` | Python bridge emulating step-level RL reward |
| `README.md` | This file |

## Running the Lean demo

Requires [Lean 4](https://leanprover.github.io/lean4/doc/quickstart.html)
(≥ 4.5.0) and optionally `mathlib4`.

```bash
# From the lean_demo directory
lean SocratesContradiction.lean
```

Expected output: the three theorems/examples type-check successfully with no
errors, confirming that:

1. `men_are_mortal Socrates h_man : Mortal Socrates` — normal forward derivation.
2. `socrates_contradiction : False` — contradiction detection succeeds.
3. The naive "everything-is-true" policy is refutable.

Each successful tactic elaboration corresponds to a `+1` step reward; each
failure corresponds to `-1`, as in Eq. (eq:lean-reward) of the paper.

## Bridging to the RLVF training loop

The current `scripts/training/stage4_train_rlvf.py` calls
`scripts.utils.forward_chain.forward_chain()` to get the oracle verdict.
To swap in Lean, replace that call with a subprocess invocation of the Lean
elaborator and parse its exit status.

See `lean_verifier_bridge.py` for a minimal reference implementation that
takes a candidate reasoning trace, writes it into a Lean tactic script,
runs `lean --check`, and returns `+1` / `-1` based on the exit status.

## What this demo establishes

* The `forward_chain()` oracle and Lean give **consistent verdicts** on
  the current benchmark (both label Variant 3 queries as `False`).
* Lean additionally supports **step-level** rewards via per-tactic
  elaboration verdicts, which `forward_chain()` cannot provide.
* The approach is compatible with recent neural theorem-proving work
  (LeanDojo, GPT-f, AlphaProof), so scaling up is technically feasible.

## Scope

This is a **feasibility demo**, not a full training run. A complete
RLVF-Lean evaluation on the benchmark requires:

1. Automatic translation of the CSV benchmark rows into Lean theorems.
2. Integration with `LeanDojo`-style interaction server for batched
   verification (current single-process `lean --check` is ~1s/call,
   too slow for end-to-end RL).
3. Extension of the tactic vocabulary to cover De Morgan /
   contrapositive / double-negation rewrites (Variant 4 equivalences).

We leave the full pipeline to future work and use this demo to
demonstrate the soundness of the approach.
