"""
Provenance and sanity metadata for evaluation runs.

Why this exists
---------------
A September 2026 audit could not tell which benchmark version the paper's
numbers had been measured on. `results/evaluation_summary.csv` (2026-03-23) and
`results/all_models_comparison.csv` (2026-04-07) were presented side by side in
one paper, but the benchmark was regenerated in between (commit ba71ff0,
"expand datasets 10x") with different Variant-3 label semantics. Nothing in
either file recorded which data it had been run against, so the mismatch was
only found half a year later by comparing file timestamps.

Two things would have caught that immediately, and four other defects besides:

  write_provenance()  - records the exact test files (size + SHA-256), the git
                        commit, and the adapter mtime alongside every summary.

  sanity_columns()    - records, per split, the majority-class baseline, the
                        prediction distribution, and (for generative runs) the
                        answer-parse rate.

An accuracy figure on its own always looks plausible. These are the numbers
that make it falsifiable. Each of the following defects produced a
reasonable-looking accuracy and would have been caught by one of them:

  * LogicNLI labels collapsed to all-False -> accuracy 1.000, but the
    majority-class baseline was also 1.000.
  * Generation truncated before the answer -> accuracy stable across four batch
    sizes (so it looked verified), but the parse rate was 0.003.
  * Classifier degenerated to a constant -> accuracy 0.750, but the prediction
    distribution was a single class.
  * Contradiction samples diluted 14x by a hardcoded cap -> accuracy 0.654,
    with contradiction mentions at 1% of generations.
"""

import hashlib
import json
import os
import subprocess


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def collect_provenance(test_files, model_dir=None, extra=None):
    """Describe what a run was actually measured against.

    test_files: {split_name: path} of the evaluation data actually read.
    model_dir:  checkpoint directory, so a stale checkpoint is visible.
    """
    data = {}
    for split, path in sorted(test_files.items()):
        if not path or not os.path.exists(path):
            data[split] = {"path": path, "missing": True}
            continue
        data[split] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path)[:16],   # 16 hex chars is plenty to spot a change
        }

    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "test_files": data,
    }
    if model_dir and os.path.isdir(model_dir):
        adapter = os.path.join(model_dir, "adapter_model.safetensors")
        prov["model_dir"] = model_dir
        if os.path.exists(adapter):
            prov["adapter_mtime"] = int(os.path.getmtime(adapter))
    if extra:
        prov.update(extra)
    return prov


def write_provenance(summary_csv_path, provenance):
    """Write <summary>.provenance.json next to the summary it describes."""
    out = os.path.splitext(summary_csv_path)[0] + ".provenance.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    return out


def sanity_columns(gold, pred, generated=None):
    """Per-split numbers that make an accuracy figure falsifiable.

    gold / pred: sequences of labels (any hashable; str or int both fine).
    generated:   optional sequence of generated strings, for generative runs.

    Returns a dict that callers can merge into their summary row.
    """
    gold = list(gold)
    pred = list(pred)
    n = len(gold)
    if n == 0:
        return {}

    from collections import Counter
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)

    majority = max(gold_counts.values()) / n
    # Fraction of predictions falling in the single most-predicted class. At
    # 1.0 the classifier has collapsed and its accuracy says nothing about
    # reasoning -- it is just reporting the label distribution back.
    pred_concentration = max(pred_counts.values()) / n

    row = {
        "n": n,
        "majority_class_baseline": round(majority, 4),
        "pred_distribution": json.dumps(
            {str(k): v for k, v in sorted(pred_counts.items(), key=lambda kv: str(kv[0]))}
        ),
        "pred_concentration": round(pred_concentration, 4),
    }

    if generated is not None:
        g = [str(x) for x in generated]
        if g:
            import re
            parsed = sum(bool(re.search(r"[Aa]nswer\s*[:=]", x)) for x in g)
            row["parse_rate"] = round(parsed / len(g), 4)
    return row


def warn_if_uninformative(split, accuracy, sanity):
    """Print a warning when an accuracy figure cannot mean what it appears to.

    Deliberately loud: every defect in the audit produced a number that looked
    fine in isolation.
    """
    msgs = []
    mb = sanity.get("majority_class_baseline")
    if mb is not None and accuracy <= mb + 1e-9:
        msgs.append(f"accuracy {accuracy:.4f} <= majority-class baseline {mb:.4f}")
    pc = sanity.get("pred_concentration")
    if pc is not None and pc >= 0.99:
        msgs.append(f"predictions {pc:.1%} in one class (collapsed)")
    pr = sanity.get("parse_rate")
    if pr is not None and pr < 0.80:
        msgs.append(f"only {pr:.1%} of generations contain a parseable answer")
    if msgs:
        print(f"  !! [{split}] accuracy may be uninformative: " + "; ".join(msgs), flush=True)
    return msgs
