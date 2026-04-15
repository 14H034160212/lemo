"""
compute_bootstrap_ci.py
Compute 95% bootstrap confidence intervals for all model accuracy results.
Reads prediction CSVs from trained_models/*/predictions/
Outputs a table with accuracy ± CI for each model and split.
"""
import os
import csv
import random
import math
from collections import defaultdict

random.seed(42)
N_BOOTSTRAP = 5000

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "trained_models")

SPLITS_OF_INTEREST = [
    "base", "hard_mixed", "variant1", "variant2", "variant3",
    "variant4_equiv_contrapositive", "variant4_equiv_double_negation",
    "variant4_equiv_implication", "variant4_equiv_demorgan",
    "variant4_equiv_identity", "variant4_equiv_commutativity",
    "variant4_equiv_multi",
]
SPLIT_SHORT = {
    "base": "base", "hard_mixed": "hard", "variant1": "v1",
    "variant2": "v2", "variant3": "v3",
    "variant4_equiv_contrapositive": "v4_contra",
    "variant4_equiv_double_negation": "v4_dbl_neg",
    "variant4_equiv_implication": "v4_impl",
    "variant4_equiv_demorgan": "v4_morgan",
    "variant4_equiv_identity": "v4_ident",
    "variant4_equiv_commutativity": "v4_comm",
    "variant4_equiv_multi": "v4_multi",
}


def bootstrap_ci(correct_list, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Return (mean, lower_ci, upper_ci) via bootstrap."""
    n = len(correct_list)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs_mean = sum(correct_list) / n
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(correct_list) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot)]
    return obs_mean, lo, hi


def load_predictions(model_dir, model_tag):
    """Load per-sample correct/incorrect from prediction CSVs."""
    pred_dir = os.path.join(model_dir, "predictions")
    if not os.path.exists(pred_dir):
        return {}
    results = defaultdict(list)
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith("_predictions.csv"):
            continue
        # Infer split name: strip model_tag prefix and _predictions.csv suffix
        split = fname
        for prefix in [f"{model_tag}_", "qwen3_", "qwen_", "llama_", "bert_"]:
            if split.startswith(prefix):
                split = split[len(prefix):]
                break
        split = split.replace("_predictions.csv", "")
        if split not in SPLITS_OF_INTEREST:
            continue
        fpath = os.path.join(pred_dir, fname)
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            correct_col = next((c for c in cols if "correct" in c.lower()), None)
            pred_col = next((c for c in cols if c.lower() in ["prediction", "predicted", "pred"]), None)
            label_col = next((c for c in cols if c.lower() in ["ground_truth", "label", "answer", "target", "true_label"]), None)
            for row in reader:
                if correct_col:
                    val = row[correct_col].strip().lower()
                    results[split].append(1 if val in ["1", "true", "yes"] else 0)
                elif pred_col and label_col:
                    results[split].append(1 if row[pred_col].strip() == row[label_col].strip() else 0)
    return results


def main():
    model_dirs = {}
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path):
            model_dirs[name] = path

    # Collect CIs
    all_ci = {}
    for model_tag, model_dir in model_dirs.items():
        preds = load_predictions(model_dir, model_tag)
        if not preds:
            continue
        ci_data = {}
        for split in SPLITS_OF_INTEREST:
            if split in preds and preds[split]:
                mean, lo, hi = bootstrap_ci(preds[split])
                ci_data[split] = (mean, lo, hi)
        if ci_data:
            all_ci[model_tag] = ci_data

    if not all_ci:
        print("No prediction files found. Run evaluate.py first to generate predictions.")
        return

    # Print table
    col_w = 22
    header_cols = [SPLIT_SHORT.get(s, s) for s in SPLITS_OF_INTEREST]
    print("\n" + "=" * 120)
    print("  ACCURACY WITH 95% BOOTSTRAP CI")
    print("=" * 120)
    print(f"{'Model':<25}" + "".join(f"{h:>{col_w}}" for h in header_cols))
    print("-" * 120)
    for model_tag, ci_data in sorted(all_ci.items()):
        row_str = f"{model_tag:<25}"
        for split in SPLITS_OF_INTEREST:
            if split in ci_data:
                mean, lo, hi = ci_data[split]
                cell = f"{mean:.3f}±{(hi-lo)/2:.3f}"
            else:
                cell = "—"
            row_str += f"{cell:>{col_w}}"
        print(row_str)
    print("=" * 120)

    # Save to CSV
    out_path = os.path.join(BASE, "results", "bootstrap_ci.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [SPLIT_SHORT.get(s, s) for s in SPLITS_OF_INTEREST] +
                        [f"{SPLIT_SHORT.get(s,s)}_lo" for s in SPLITS_OF_INTEREST] +
                        [f"{SPLIT_SHORT.get(s,s)}_hi" for s in SPLITS_OF_INTEREST])
        for model_tag, ci_data in sorted(all_ci.items()):
            means = [ci_data[s][0] if s in ci_data else "" for s in SPLITS_OF_INTEREST]
            los = [ci_data[s][1] if s in ci_data else "" for s in SPLITS_OF_INTEREST]
            his = [ci_data[s][2] if s in ci_data else "" for s in SPLITS_OF_INTEREST]
            writer.writerow([model_tag] + means + los + his)
    print(f"\n  Saved to: {out_path}\n")


if __name__ == "__main__":
    main()
