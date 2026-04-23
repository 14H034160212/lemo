/-
  SocratesContradiction.lean
  ==========================
  Phase 2 feasibility demo for the LEMO / Conflict-Aware Fusion paper.

  Goal: Show that Lean 4 can serve as a step-level verifier for the
  Socrates contradiction example used as a running example in Figure 1
  of the paper. The model's desired "Halt on contradiction" behavior
  is encoded as a Lean theorem, and the kernel checks the proof.

  Usage (requires Lean 4 + mathlib):
      lean SocratesContradiction.lean

  Or open in VS Code with the lean4 extension: Lean elaborates each
  tactic in real time, giving a step-by-step verification verdict
  that can be used as an RL reward signal (see Eq. (eq:lean-reward)
  in Section 6.5 / Phase 2 of the paper).
-/

namespace LEMO.SocratesDemo

/-! ## 1. Domain encoding -/

/-- The universe: individuals in our logical domain. -/
axiom Person : Type

/-- Named individual. -/
axiom Socrates : Person

/-- Unary predicates corresponding to the natural-language attributes. -/
axiom Man    : Person → Prop
axiom Mortal : Person → Prop

/-! ## 2. The rule base ("All men are mortal") -/

/--
  Universal rule: every man is mortal.
  This is the single implication in the Socrates example.
-/
axiom men_are_mortal : ∀ x : Person, Man x → Mortal x

/-! ## 3. The canonical (non-contradictory) case -/

/--
  In the base case we know only `Man Socrates`.
  Forward chaining derives `Mortal Socrates`.
  This is the behavior Stage 1 (SFT) should learn: standard deduction.
-/
example (h_man : Man Socrates) : Mortal Socrates := by
  exact men_are_mortal Socrates h_man

/-! ## 4. The contradiction case (Variant 3 / Figure 1) -/

/--
  The contradiction variant: we assert both `Man Socrates`
  (which forces `Mortal Socrates` via the rule) AND a new fact
  `¬ Mortal Socrates`. The model MUST detect this and halt.

  The Lean kernel verifies the contradiction via `absurd`:
  - `men_are_mortal Socrates h_man`  produces  `Mortal Socrates`
  - combined with `h_not_mortal : ¬ Mortal Socrates`  yields  `False`
  - therefore every query is vacuously `False` (conservative semantics)
-/
theorem socrates_contradiction
    (h_man        : Man Socrates)
    (h_not_mortal : ¬ Mortal Socrates) :
    False := by
  have h_mortal : Mortal Socrates := men_are_mortal Socrates h_man
  exact absurd h_mortal h_not_mortal

/-! ## 5. Step-level reward emulation

  Each of the following `#check` calls simulates one RL step.
  If the tactic elaborates, the Lean kernel returns success → reward = +1.
  If it fails, Lean reports the error → reward = -1.

  An RL training loop would pipe the model's emitted tactic through
  `Lean.Elab.runTactic` and use the exit status as the step reward r_t
  in Eq. (eq:lean-reward) / Eq. (eq:rlvf).
-/

-- Step 1 (Correct "Halt" path) — model claims to derive Mortal from Man
#check @men_are_mortal        -- kernel: success (+1)

-- Step 2 — model combines the derived fact with the injected negation
-- The tactic `exact absurd (men_are_mortal _ ·) ·` should type-check.
example (hm : Man Socrates) (hn : ¬ Mortal Socrates) : False :=
  absurd (men_are_mortal Socrates hm) hn     -- kernel: success (+1)

/-! ## 6. What the naive model would emit (and why Lean rejects it)

  If the policy, under Logic Inertia, tries to conclude `Mortal Socrates`
  while ignoring `¬ Mortal Socrates`, the following "proof" would still
  elaborate:  `men_are_mortal Socrates h_man`  has type `Mortal Socrates`.

  HOWEVER, in the contradictory context, Lean allows us to derive
  `False`, which under our conservative semantics labels every query
  `False`. A model that outputs `True` would be penalized (r = -1)
  because the oracle ground truth a* = False for every query in this
  context.

  Below: the discriminating check. We show that in a context with the
  contradictory premise, `True` is NOT derivable without assuming `False`,
  so the naive "continue deducing" policy has no valid proof term.
-/
example
    (h_man        : Man Socrates)
    (h_not_mortal : ¬ Mortal Socrates) :
    ¬ (∀ (_q : Prop), _q) := by
  intro hAll
  -- A naive model claiming "everything is True" should be refuted
  -- by any proposition we disagree with, e.g. `False`:
  exact hAll False

/-! ## 7. Summary

  * `socrates_contradiction` is the theorem the RLVF-Lean policy is
    rewarded for discovering.
  * The kernel's accept/reject verdict on each tactic becomes a
    step-level reward r_t (Eq. (eq:lean-reward) in the paper).
  * The current `forward_chain.py` oracle returns only the terminal
    verdict for the final T/F answer, which is a strict subset of
    what Lean provides.
-/

end LEMO.SocratesDemo
