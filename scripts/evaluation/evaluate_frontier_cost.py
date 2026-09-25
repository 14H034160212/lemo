"""
Accuracy against generated-token cost, on the two-class Variant-3 control.

The claim this measures: a model that has internalised "check the premises
before deducing" should reach the right answer in fewer generated tokens than
one that has to reason its way there at inference time, or one that calls an
external solver. That is the practical argument for a training-time structural
prior, and it has never been measured -- the existing frontier script records
answers but not usage.

Three things make this comparison honest rather than flattering:

  * It runs on test_variant3_mixed.csv, not the single-class Variant 3 split.
    On the latter a model that answers False on sight scores 1.000, so any
    tokens-per-correct-answer figure computed there is meaningless.

  * It counts *generated* tokens, which for the frontier models includes
    reasoning/thinking tokens. Claude Opus 5 has thinking on by default and
    those tokens are billed and waited for, so excluding them would understate
    the baseline's cost by most of it.

  * It reports tokens per question alongside accuracy rather than a single
    ratio, because a model can look cheap by being wrong quickly.

Our own models are measured separately from their prediction files, counting
tokens up to and including the answer -- the quantity comparable to what the
API bills -- rather than total output length, which for RLVF runs to the
generation cap long after the answer is settled.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.evaluation.evaluate_frontier_api import (
    SYSTEM_PROMPT, build_user_prompt, parse_answers,
)

VERIFY_FIRST = (
    "Before answering, normalise the rules, check the premise set for "
    "contradictions, and identify any missing dependency. If the premises are "
    "inconsistent, every query on them is False. Then answer."
)


def load_stratified(path, per_class, seed=0):
    """Equal numbers from each instance class, so the cost figure is not
    dominated by whichever class happens to be cheapest."""
    import csv
    from collections import defaultdict
    byc = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            byc[r["type"]].append(r)
    rng = random.Random(seed)
    out = []
    for k in sorted(byc):
        rng.shuffle(byc[k])
        out.extend(byc[k][:per_class])
    rng.shuffle(out)
    return out


def call(provider, client, model, row, verify_first, reasoning_effort=None):
    sys_prompt = SYSTEM_PROMPT + ("\n\n" + VERIFY_FIRST if verify_first else "")
    user = build_user_prompt(row["facts"], row["rules"], row["questions"])
    t0 = time.time()
    if provider == "anthropic":
        m = client.messages.create(
            model=model, max_tokens=2048, system=sys_prompt,
            messages=[{"role": "user", "content": user}])
        text = next((b.text for b in m.content if b.type == "text"), "")
        # output_tokens covers thinking as well as the visible reply
        out_tok, in_tok = m.usage.output_tokens, m.usage.input_tokens
    else:
        kw = ({"max_completion_tokens": 4096} if model.startswith("gpt-5")
              else {"max_tokens": 2048})
        # gpt-5.x defaults to NO reasoning: 10 completion tokens, 0 of them
        # reasoning, for a four-question instance. Any token comparison that
        # leaves the default in place is comparing our full verification trace
        # against a bare "True/False/True/False", so the effort level is swept
        # explicitly rather than inherited.
        if reasoning_effort and model.startswith("gpt-5"):
            kw["reasoning_effort"] = reasoning_effort
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}], **kw)
        text = r.choices[0].message.content or ""
        out_tok, in_tok = r.usage.completion_tokens, r.usage.prompt_tokens
        d = getattr(r.usage, "completion_tokens_details", None)
        reason_tok = getattr(d, "reasoning_tokens", 0) or 0
        return text, out_tok, in_tok, time.time() - t0, reason_tok
    return text, out_tok, in_tok, time.time() - t0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_file", default="data/test_variant3_mixed.csv")
    ap.add_argument("--per_class", type=int, default=100)
    ap.add_argument("--verify_first", action="store_true",
                    help="prepend the verification-first instruction, to price "
                         "the inference-time mitigation as well as measure it")
    ap.add_argument("--reasoning_effort", default=None,
                    choices=[None, "minimal", "low", "medium", "high"],
                    help="gpt-5.x only; the default is no reasoning at all")
    ap.add_argument("--sample_seed", type=int, default=0,
                    help="which stratified draw to evaluate; vary it across "
                         "repeats so the error bar covers instance sampling "
                         "as well as decoding")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_stratified(args.test_file, args.per_class, seed=args.sample_seed)
    print(f"{args.model}: {len(rows)} instances "
          f"({len(rows)*4} questions), verify_first={args.verify_first}", flush=True)

    if args.provider == "anthropic":
        import anthropic; client = anthropic.Anthropic()
    else:
        import openai; client = openai.OpenAI()

    recs, correct, total, tok_out, tok_in, secs, failed = [], 0, 0, 0, 0, 0.0, 0
    tok_reason = 0
    for i, row in enumerate(rows):
        gold = [a.strip() for a in row["answers"].split(" | ")]
        try:
            text, ot, it, dt, rt = call(args.provider, client, args.model, row,
                                        args.verify_first, args.reasoning_effort)
        except Exception as e:
            failed += 1
            print(f"  [{i}] call failed: {type(e).__name__}: {e}", flush=True)
            continue
        preds = parse_answers(text, len(gold))
        ok = sum(p == g for p, g in zip(preds, gold))
        correct += ok; total += len(gold)
        tok_out += ot; tok_in += it; secs += dt; tok_reason += rt
        recs.append({"type": row["type"], "depth": row.get("depth"),
                     "arm": row.get("arm"),
                     # kept so an arm and its foil can be scored jointly:
                     # per-arm rates cannot tell "both halves right" from
                     # "one right in each half"
                     "group_id": row.get("group_id"),
                     "gold": gold, "pred": preds,
                     "correct": ok, "n": len(gold),
                     "out_tokens": ot, "in_tokens": it,
                     "reasoning_tokens": rt, "seconds": round(dt, 2)})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] acc={correct/total:.4f}  "
                  f"out_tok/question={tok_out/total:.1f}", flush=True)
        time.sleep(args.delay)

    if not total:
        print("no successful calls"); return
    tag = (args.model.replace("/", "_")
           + (f"_{args.reasoning_effort}" if args.reasoning_effort else "_default")
           + ("_verifyfirst" if args.verify_first else ""))
    out = args.out or f"results/frontier_cost/{tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary = {
        "model": args.model, "verify_first": args.verify_first,
        "test_file": args.test_file, "instances": len(recs), "questions": total,
        "failed_calls": failed,
        "accuracy": round(correct / total, 4),
        "out_tokens_per_question": round(tok_out / total, 1),
        "reasoning_tokens_per_question": round(tok_reason / total, 1),
        "reasoning_effort": args.reasoning_effort,
        "sample_seed": args.sample_seed,
        "in_tokens_per_question": round(tok_in / total, 1),
        "seconds_per_question": round(secs / total, 3),
        # computed from the rows actually evaluated: the hardcoded 0.5835
        # belonged to variant3_mixed and was wrong for every other file
        "majority_class_baseline": round(max(
            sum(g.count("T") for g in (r["gold"] for r in recs)),
            sum(g.count("F") for g in (r["gold"] for r in recs)),
        ) / total, 4) if total else None,
    }
    # Per-depth breakdown: the whole point of the depth benchmark is whether
    # cost grows with chain length, which a single mean hides.
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0, 0])      # key -> [correct, n, out_tokens]
    for r in recs:
        if r.get("depth") is None:
            continue
        cls = "contra" if "contra" in r["type"] else "consistent"
        for k in (f'depth{r["depth"]}_{cls}', f'depth{r["depth"]}_all'):
            by[k][0] += r["correct"]; by[k][1] += r["n"]; by[k][2] += r["out_tokens"]
    # Per question position. On the matched MP/MT control the chain position of
    # each query is fixed, so position 3 is the manipulated inference and
    # position 4 is downstream of it: comparing them separates a primary
    # failure from one that merely propagated.
    bypos = defaultdict(lambda: [0, 0])
    for r in recs:
        if not r.get("arm"):
            continue
        for i, (g, pr) in enumerate(zip(r["gold"], r["pred"])):
            k = f'{r["arm"]}_Q{i+1}'
            bypos[k][0] += int(pr == g); bypos[k][1] += 1
    if bypos:
        summary["by_position"] = {k: {"accuracy": round(v[0]/v[1], 4), "n": v[1]}
                                  for k, v in sorted(bypos.items())}
        # propagation: among pairs where Q3 is wrong, how often is Q4 also wrong
        prop = defaultdict(lambda: [0, 0])
        for r in recs:
            if not r.get("arm") or len(r["pred"]) < 4:
                continue
            q3ok = r["pred"][2] == r["gold"][2]
            q4ok = r["pred"][3] == r["gold"][3]
            prop[r["arm"]][1 if q3ok else 0] += 0   # touch
            if not q3ok:
                prop[r["arm"]][0] += int(not q4ok); prop[r["arm"]][1] += 1
        summary["q4_wrong_given_q3_wrong"] = {
            k: (round(v[0]/v[1], 4) if v[1] else None) for k, v in prop.items()}

    if by:
        summary["by_depth"] = {
            k: {"accuracy": round(v[0] / v[1], 4),
                "out_tokens_per_question": round(v[2] / v[1], 1), "questions": v[1]}
            for k, v in sorted(by.items(), key=lambda kv: (int(kv[0].split("_")[0][5:]), kv[0]))}
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
