"""
lean_verifier_bridge.py
=======================
Minimal reference implementation of an RLVF-Lean verifier bridge.

Takes a candidate reasoning trace (a list of tactic strings) and runs
Lean 4 on a synthesized script containing the user's hypotheses and the
emitted tactics. Returns +1 if Lean accepts the proof, -1 if any tactic
fails, 0 if the proof is incomplete but no error was raised.

This is the function that would replace `forward_chain()` in
`scripts/training/stage4_train_rlvf.py` for the RLVF-Lean variant of
the pipeline (Section 6.5 / Phase 2 of the paper).

Currently used only for feasibility demonstration --- a production
implementation should use `LeanDojo` or a persistent Lean server to
avoid ~1-second startup latency per call.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


# --------------------------------------------------------------------- #
# Template for the synthesized Lean script                              #
# --------------------------------------------------------------------- #

_LEAN_TEMPLATE = r"""
namespace LEMO.Runtime

axiom Person : Type
axiom Socrates : Person
axiom Man    : Person → Prop
axiom Mortal : Person → Prop
axiom men_are_mortal : ∀ x : Person, Man x → Mortal x

/-- The candidate proof built from the policy's tactic trace. -/
theorem _policy_claim
    (h_man        : Man Socrates)
    (h_not_mortal : ¬ Mortal Socrates) :
    False := by
{tactics}

end LEMO.Runtime
"""


# --------------------------------------------------------------------- #
# Verifier                                                              #
# --------------------------------------------------------------------- #

def lean_reward(
    tactic_trace: List[str],
    lean_bin: str = "lean",
    timeout: float = 10.0,
) -> Tuple[int, str]:
    """
    Run Lean on a candidate proof and return (reward, stderr_text).

    Parameters
    ----------
    tactic_trace :
        List of Lean 4 tactic strings. Indentation is added automatically.
        Example: ["have h_mortal := men_are_mortal Socrates h_man",
                  "exact absurd h_mortal h_not_mortal"]
    lean_bin :
        Path to the `lean` executable.
    timeout :
        Wall-clock seconds before aborting.

    Returns
    -------
    (reward, stderr)
        reward = +1 if Lean accepts the proof,
                -1 if any tactic fails / kernel rejects,
                 0 if Lean times out or the trace is empty.
    """
    if not tactic_trace:
        return 0, "empty trace"

    # Indent tactics for inclusion in the `by` block
    body = "\n".join("  " + t.rstrip() for t in tactic_trace)
    script = _LEAN_TEMPLATE.format(tactics=body)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = Path(f.name)

    try:
        res = subprocess.run(
            [lean_bin, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0, "lean timeout"
    except FileNotFoundError:
        return 0, f"lean binary not found at '{lean_bin}'"
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    # Treat `sorry` / `admit` as failures even though Lean exits with 0
    # (they emit warnings but don't prove anything). This is essential for
    # RL soundness: the policy must not be rewarded for admitting goals.
    combined = (res.stderr or "") + "\n" + (res.stdout or "")
    if "sorry" in combined.lower() or "admit" in combined.lower():
        return -1, combined

    if res.returncode == 0:
        return 1, res.stderr
    else:
        return -1, res.stderr


# --------------------------------------------------------------------- #
# Smoke test                                                            #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    # Correct halt-on-contradiction trace
    good_trace = [
        "have h_mortal : Mortal Socrates := men_are_mortal Socrates h_man",
        "exact absurd h_mortal h_not_mortal",
    ]

    # Naive "continue deducing" trace (missing the contradiction step)
    bad_trace = [
        "have h_mortal : Mortal Socrates := men_are_mortal Socrates h_man",
        "-- model concludes Mortal Socrates and ignores h_not_mortal",
    ]

    for name, trace in [("good (halt)", good_trace), ("bad (naive)", bad_trace)]:
        r, err = lean_reward(trace)
        print(f"[{name}] reward = {r:+d}")
        if err and r != 1:
            # Only show first few lines of Lean error for brevity
            print("  lean stderr:\n  " + "\n  ".join(err.splitlines()[:5]))
