"""
How many published reasoning-benchmark splits can a constant predictor saturate?

Motivation, from this project's own retractions. Three of the four claims in an
earlier draft turned out to be artefacts of the same structural property:

    contradiction injection + conservative semantics -> every query is False
    equivalence rewriting                            -> every answer unchanged
    essential-rule deletion                          -> every dependent query fails

The perturbation determines the label, so the perturbed split is single-class,
so answering one constant scores perfectly on it. Measured on our own
benchmark, 9 of 12 splits had this property, and the reported 1.000 on the
contradiction split came from a model that answered False on 4000/4000
questions without checking anything.

That mechanism is not specific to us -- it follows from how perturbation-style
benchmarks are built. This script measures the prevalence directly: for every
split of every benchmark, the label distribution, the majority-class baseline,
and therefore the score a constant predictor achieves.

Benchmarks whose questions are generated in balanced true/false pairs
(ProofWriter, RuleTaker) act as the control group: if the method is sound they
should come out near 0.5 and the perturbation-style benchmarks should not.
"""

import argparse
import json
import os
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", ".cache/huggingface")


def stats(labels):
    c = Counter(labels)
    n = sum(c.values())
    if not n:
        return None
    top, topn = c.most_common(1)[0]
    return {
        "n": n,
        "n_classes": len(c),
        "majority_class": str(top),
        "majority_baseline": round(topn / n, 4),
        "distribution": {str(k): v for k, v in c.most_common(6)},
    }


def survey(name, config, label_field, group_fields, max_rows, split_filter=None):
    from datasets import load_dataset
    ds = load_dataset(name, config) if config else load_dataset(name)
    out = {}
    for split in ds:
        if split_filter and split not in split_filter:
            continue
        d = ds[split]
        if label_field not in d.column_names:
            return {"error": f"no label field {label_field}; has {d.column_names}"}
        if max_rows and len(d) > max_rows:
            d = d.shuffle(seed=0).select(range(max_rows))
        rows = d.to_dict()
        # the whole split
        out[split] = stats(rows[label_field])
        # and each grouping the benchmark itself defines -- the "condition"
        # column is what a perturbation split is
        for gf in group_fields:
            if gf not in d.column_names:
                continue
            by = defaultdict(list)
            for g, lab in zip(rows[gf], rows[label_field]):
                by[str(g)].append(lab)
            for g, labs in sorted(by.items()):
                if len(labs) >= 30:
                    out[f"{split}::{gf}={g}"] = stats(labs)
    return out


BENCHMARKS = [
    # (key, hf name, config, label field, grouping fields, row cap, splits)
    ("ProofWriter", "tasksource/proofwriter", None, "answer",
     ["config", "QDep", "maxD"], 60000, ["test"]),
    ("RuleTaker", "tasksource/ruletaker", None, "label",
     ["config"], 60000, ["test"]),
    ("ProofWriter-MCQ", "renma/ProofWriter", None, "answer", [], None, None),
    ("FOLIO", "tasksource/folio", None, "label", [], None, None),
    ("LogicalEntailment", "tasksource/logical-entailment", None, "label",
     [], 20000, ["test"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark_survey.json")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    allout = {}
    for key, name, cfg, lab, grp, cap, spl in BENCHMARKS:
        if args.only and key not in args.only:
            continue
        print(f"=== {key} ({name})", flush=True)
        try:
            allout[key] = {"hf": name, "splits": survey(name, cfg, lab, grp, cap, spl)}
        except Exception as e:
            print(f"    FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
            allout[key] = {"hf": name, "error": f"{type(e).__name__}: {e}"}
            continue
        sp = allout[key]["splits"]
        if "error" in sp:
            print("   ", sp["error"], flush=True); continue
        bad = [(k, v) for k, v in sp.items() if v and v["majority_baseline"] >= 0.90]
        print(f"    {len(sp)} groups;  {len(bad)} with majority baseline >= 0.90", flush=True)
        for k, v in sorted(sp.items(), key=lambda kv: -kv[1]["majority_baseline"])[:6]:
            print(f"      {k:44s} n={v['n']:6d}  baseline={v['majority_baseline']:.4f}"
                  f"  classes={v['n_classes']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(allout, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
