"""
Generative RLVF -- the method as the paper actually describes it.

The paper (Eq. 4) specifies:

    "The policy emits a full reasoning trace tau ~ pi_theta(.|x); a deterministic
     parser Pi(tau) in {T,F} extracts the boolean answer from the final
     'Step N: Answer = ...' line (regex-matched; unparseable traces get r = -1).
     The oracle returns a* = ForwardChain(x) and reward compares parsed answer
     to oracle:
        L_RLVF = -E_tau[(r - b_t) log pi_theta(tau|x)]
        r = 1[Pi(tau) == a*] - 1[Pi(tau) != a*]
        b_t = mu b_{t-1} + (1-mu) rbar_t"

`stage4_train_rlvf.py` does not implement that. It builds an
AutoModelForSequenceClassification and optimises a 2-way logit: there is no
trace, no parser, and no sequence log-probability. On the corrected benchmark it
scores 0.0000 on Variant 3 -- it has no output through which "halt on
contradiction" could be expressed.

The one component whose implementation does match its description --
stage2_train_fusion.py, which is CAUSAL_LM and emits an explicit
"Step 1: Verify facts. Conflict detected! ... Step 2: Stop." trace -- is also
the one that works: 0.982 on Variant 3, with 97.8% of its generations naming
the conflict. This script tests whether RLVF recovers the paper's result once
it is run on the generation path the paper describes, starting from that
checkpoint.

Reward is the symbolic oracle, so nothing here depends on the benchmark's
stored labels:  a* = forward_chain(facts, rules) evaluated at the question.
"""

import argparse
import os
import random
import re

os.environ.setdefault('HF_HOME', '.cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', '.cache/huggingface/datasets')
os.environ.setdefault('TRANSFORMERS_CACHE', '.cache/huggingface/transformers')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from scripts.utils.forward_chain import (
    forward_chain, check_answer, detect_contradiction,
)


# ----------------------------------------------------------------- prompt/parse
def format_question_prompt(facts, rules, question):
    """Identical to evaluate_generative.py, so training and evaluation agree."""
    return f"Facts: {facts}\nRules: {rules}\nQuestion: {question}\nThink step by step."


_ANS_RE = re.compile(r"[Aa]nswer\s*[:=]\s*(True|False|T|F)\b")


def parse_answer(text):
    """Pi(tau). Returns 'T', 'F', or None when the trace has no parseable answer.

    None is distinct from a wrong answer: the paper assigns r = -1 to
    unparseable traces, and keeping them separate lets us report the parse rate,
    which is what reveals a truncated or degenerate policy.
    """
    m = None
    for m in _ANS_RE.finditer(text):
        pass                      # keep the LAST match: the trace concludes at the end
    if m is None:
        return None
    tok = m.group(1).lower()
    return "T" if tok in ("true", "t") else "F"


def oracle_answer(facts, rules, question):
    """a* = ForwardChain(x) under the paper's conservative contradiction
    semantics (Section 3.1): if the premise set is inconsistent, every query is
    False, whatever the chain would otherwise derive.

    The contradiction override is not optional. Without it this oracle returned
    True on 1200/1200 sampled Variant-3 training questions while the benchmark
    labels the same questions False on 1200/1200 -- reward and evaluation were
    exactly opposed on the one split the method exists to fix. RLVF then did
    what it was asked: both the classifier run and the generative run scored
    ~0.000 on Variant 3, and the generative run, initialised from a checkpoint
    that scored 0.982 there, was driven down to 0.0003.

    forward_chain() cannot supply this by itself -- it lets a later derivation
    silently overwrite an earlier conflicting value -- which is why
    detect_contradiction() exists as a separate additive check.
    """
    if detect_contradiction(facts, rules):
        return "F"
    closure = forward_chain(facts, rules)
    val = check_answer(question, closure)
    return "T" if val is True else "F"


# ----------------------------------------------------------------- data
def build_prompts(train_csv, pairs_csv, v4_ratio, seed=0):
    """Prompts only -- rewards come from the oracle at rollout time."""
    rng = random.Random(seed)
    items = []

    df = pd.read_csv(train_csv)
    for _, row in df.iterrows():
        facts, rules = str(row["facts"]), str(row["rules"])
        for q in str(row["questions"]).split(" | "):
            items.append({"facts": facts, "rules": rules, "question": q,
                          "source": str(row.get("type", "train"))})

    v4 = []
    if pairs_csv and os.path.exists(pairs_csv):
        dp = pd.read_csv(pairs_csv)
        for _, row in dp.iterrows():
            for fk, rk in (("base_facts", "base_rules"), ("equiv_facts", "equiv_rules")):
                if fk not in dp.columns:
                    continue
                facts, rules = str(row[fk]), str(row[rk])
                for q in str(row["questions"]).split(" | "):
                    v4.append({"facts": facts, "rules": rules, "question": q,
                               "source": "variant4"})

    # Same mixing rule as the classifier version: v4_ratio of the final mix.
    if v4 and 0 < v4_ratio < 1:
        target = int(len(items) * v4_ratio / (1 - v4_ratio))
        v4 = rng.sample(v4, min(target, len(v4)))
    items += v4
    rng.shuffle(items)
    return items


def balance_by_oracle(items, rng):
    """Equalise the oracle-correct answer distribution before RL.

    The default mix is 79.5% True-answer, because the Variant-4 equivalence
    rewrites -- half the prompts at v4_ratio=0.5 -- are all-True by
    construction. REINFORCE on that distribution rewards a True-bias, and the
    first run wiped out the contradiction-halting behaviour the SFT
    initialisation already had (Variant 3: 0.982 -> 0.0003).

    The reward oracle is unbiased per item; the *prompt distribution* is not.
    This downsamples the majority answer so the policy cannot profit from a
    constant response.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for it in items:
        try:
            buckets[oracle_answer(it["facts"], it["rules"], it["question"])].append(it)
        except Exception:
            continue
    if len(buckets) < 2:
        return items
    k = min(len(v) for v in buckets.values())
    out = []
    for ans, group in buckets.items():
        out.extend(rng.sample(group, k))
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(model, tokenizer, prompts, device, max_new_tokens, temperature):
    """Sample one trace per prompt. Sampling (not greedy) is required: REINFORCE
    needs the policy's own distribution, and a greedy rollout has no variance to
    learn from."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        tokenizer.padding_side = side
    plen = enc["input_ids"].shape[1]
    gen_ids = out[:, plen:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return enc, gen_ids, [t.strip() for t in texts]


def _token_logprobs(model, enc, gen_ids, tokenizer):
    """Per-token log pi(tau_t | x, tau_<t) plus the padding mask."""
    full = torch.cat([enc["input_ids"], gen_ids], dim=1)
    attn = torch.cat([enc["attention_mask"], (gen_ids != tokenizer.pad_token_id).long()], dim=1)
    logits = model(input_ids=full, attention_mask=attn).logits
    plen = enc["input_ids"].shape[1]
    gen_logits = logits[:, plen - 1:-1, :]
    # gather the target token's logit, then normalise -- equivalent to
    # log_softmax(...).gather(...) but without materialising a
    # [batch, seq, 151k] float32 buffer, which is what ran the card out of
    # memory at batch 16.
    gen_logits = gen_logits.float()
    tok_logit = gen_logits.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_logit - torch.logsumexp(gen_logits, dim=-1)
    mask = (gen_ids != tokenizer.pad_token_id).float()
    return tok_lp, mask


def sequence_logprob(model, enc, gen_ids, tokenizer):
    """log pi_theta(tau|x), summed over generated tokens, padding excluded."""
    full = torch.cat([enc["input_ids"], gen_ids], dim=1)
    attn = torch.cat([enc["attention_mask"], (gen_ids != tokenizer.pad_token_id).long()], dim=1)
    logits = model(input_ids=full, attention_mask=attn).logits
    plen = enc["input_ids"].shape[1]
    # predict token t from position t-1
    gen_logits = logits[:, plen - 1:-1, :]
    logprobs = F.log_softmax(gen_logits.float(), dim=-1)
    tok_lp = logprobs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    mask = (gen_ids != tokenizer.pad_token_id).float()
    # Length-normalise. The paper writes log pi(tau|x) as a plain sum, but a
    # trace is 40-160 tokens, so the summed term swamps the (r - b) factor and
    # the gradient tracks trace length more than reward -- the smoke run showed
    # the loss swinging between -10 and +3 while rewards stayed in [0, 0.75].
    # Dividing by the token count keeps the REINFORCE direction identical and
    # only rescales the step.
    return (tok_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ----------------------------------------------------------------- train
def train(args):
    print("=" * 70)
    print("Generative RLVF  (paper Eq. 4: trace -> parser -> oracle reward)")
    print("=" * 70)
    print(f"  init from   : {args.init_dir}")
    print(f"  train data  : {args.train_csv}")
    print(f"  v4 pairs    : {args.pairs_csv}  (ratio {args.v4_ratio})")
    print(f"  steps       : {args.max_steps}  batch {args.batch_size}")
    print(f"  output      : {args.output_dir}")
    print("=" * 70, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.init_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.init_dir, torch_dtype=torch.float32, is_trainable=True).to(device)
    model.train()

    # rollout() samples with do_sample=True, and nothing seeded torch, so runs
    # varied but could not be reproduced and --seed controlled only the prompt
    # order. Seed every RNG the rollout touches.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    items = build_prompts(args.train_csv, args.pairs_csv, args.v4_ratio, args.seed)
    if args.balance_answers:
        from collections import Counter
        before = Counter(it["source"].split("_")[0] for it in items)
        items = balance_by_oracle(items, random.Random(args.seed))
        after = Counter(it["source"].split("_")[0] for it in items)
        print(f"  balanced    : {sum(before.values())} -> {len(items)} prompts")
        print(f"    by source : {dict(after)}")
        chk = Counter(oracle_answer(i["facts"], i["rules"], i["question"])
                      for i in random.Random(1).sample(items, min(2000, len(items))))
        print(f"    oracle answers (sampled): {dict(chk)}", flush=True)
    print(f"  rollout pool: {len(items)} prompts", flush=True)

    # Frozen copy of the initial policy. Plain REINFORCE with an EMA baseline --
    # exactly what the paper's Eq. 4 specifies -- collapsed at step 250 of the
    # first run: once the baseline climbed to +0.695, a correct trace was worth
    # (1 - 0.695) = +0.305 while a wrong one was worth (-1 - 0.695) = -1.695, a
    # 5.6x asymmetry with nothing holding the policy near its starting
    # distribution. One unlucky batch pushed it off-distribution, generations
    # degenerated to gibberish, every trace then failed to parse (r = -1), and
    # the run locked into that state. The paper argues PPO's clipping is
    # unnecessary because the oracle reward is unbiased and deterministic; that
    # argument addresses reward noise, but clipping/KL also exists to stop the
    # policy from leaving the region where it can still produce valid text.
    # Reference policy = this model's weights as they are right now, before any
    # RL step. Two rejected alternatives:
    #   - a second AutoPeftModelForCausalLM: ~35 GB on top of the policy's own
    #     35 GB, which OOM'd at batch 16 on an 80 GB card;
    #   - disable_adapter(): that yields the RAW base model, not the
    #     conflict-aware SFT checkpoint we start from. Measured KL against it
    #     sat at +2.5 to +3.2 instead of ~0, i.e. the penalty would have pulled
    #     the policy back toward an un-finetuned model and undone the SFT.
    # Snapshotting just the trainable LoRA tensors costs a few MB.
    ref_state = {k: v.detach().clone()
                 for k, v in model.state_dict().items() if "lora" in k.lower()}

    def _swap_in_reference():
        cur = {k: model.state_dict()[k].detach().clone() for k in ref_state}
        model.load_state_dict(ref_state, strict=False)
        return cur

    def _restore(cur):
        model.load_state_dict(cur, strict=False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    baseline = 0.0          # b_t, EMA over batch mean reward
    bad_streak = 0
    last_good_state = None
    last_good_step = 0
    mu = args.baseline_momentum
    step = 0
    ptr = 0
    while step < args.max_steps:
        batch = items[ptr:ptr + args.batch_size]
        ptr += args.batch_size
        if ptr >= len(items):
            random.Random(args.seed + step).shuffle(items)
            ptr = 0
        if not batch:
            continue

        prompts = [format_question_prompt(b["facts"], b["rules"], b["question"]) for b in batch]
        enc, gen_ids, texts = rollout(model, tokenizer, prompts, device,
                                      args.max_new_tokens, args.temperature)

        rewards, n_parsed = [], 0
        for b, txt in zip(batch, texts):
            pred = parse_answer(txt)
            if pred is None:
                rewards.append(-1.0)          # unparseable trace, per the paper
                continue
            n_parsed += 1
            gold = oracle_answer(b["facts"], b["rules"], b["question"])
            rewards.append(1.0 if pred == gold else -1.0)
        r = torch.tensor(rewards, device=device, dtype=torch.float32)

        rbar = r.mean().item()

        # Reference pass FIRST, entirely under no_grad and with the weights
        # swapped, then restore before the policy pass. Doing it the other way
        # round mutates tensors the policy's autograd graph still holds and
        # backward fails with an in-place-modification error.
        with torch.no_grad():
            _saved = _swap_in_reference()
            ref_lp, ref_mask = _token_logprobs(model, enc, gen_ids, tokenizer)
            _restore(_saved)
            ref_denom = ref_mask.sum(dim=1).clamp(min=1)
            ref_logp = (ref_lp * ref_mask).sum(dim=1) / ref_denom

        tok_lp, mask = _token_logprobs(model, enc, gen_ids, tokenizer)
        denom = mask.sum(dim=1).clamp(min=1)
        logp = (tok_lp * mask).sum(dim=1) / denom
        # Per-sequence KL(pi_theta || pi_ref), estimated on the sampled trace.
        kl = (logp - ref_logp).detach()
        r_eff = r - args.kl_beta * kl

        rbar_eff = r_eff.mean().item()
        loss = -((r_eff - baseline) * logp).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        baseline = mu * baseline + (1 - mu) * rbar_eff

        step += 1
        parse_rate = n_parsed / len(batch)
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:5d}/{args.max_steps}  reward={rbar:+.3f}  "
                  f"baseline={baseline:+.3f}  KL={kl.mean().item():+.3f}  "
                  f"parse_rate={parse_rate:.3f}  loss={loss.item():+.4f}", flush=True)

        # Collapse guard. The first run spent five hours after the policy had
        # already degenerated: the parse rate hit 0.000 at step 300 and every
        # subsequent step trained on r = -1 for all 16 rollouts. A dead policy
        # produces no usable gradient, so there is nothing to recover by
        # continuing -- stop and keep the last healthy checkpoint instead.
        if parse_rate < args.min_parse_rate:
            bad_streak += 1
            if bad_streak >= args.patience:
                print(f"\n  !! parse rate below {args.min_parse_rate} for "
                      f"{bad_streak} consecutive steps -- policy has degenerated. "
                      f"Stopping at step {step} and keeping the checkpoint from "
                      f"step {last_good_step}.", flush=True)
                if last_good_state is not None:
                    model.load_state_dict(last_good_state)
                break
        else:
            bad_streak = 0
            if step % args.checkpoint_every == 0:
                last_good_state = {k: v.detach().clone()
                                   for k, v in model.state_dict().items()}
                last_good_step = step

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generative RLVF (paper Eq. 4)")
    # Start from the conflict-aware SFT model: it is the checkpoint that already
    # emits a verification trace, which is what the reward can then shape.
    ap.add_argument("--init_dir", default="trained_models/qwen_fusion_sft_conflict_aware")
    ap.add_argument("--train_csv", default="data/train_with_v3.csv")
    ap.add_argument("--pairs_csv", default="data/train_lire_pairs.csv")
    ap.add_argument("--output_dir", default="trained_models/qwen_rlvf_gen")
    ap.add_argument("--v4_ratio", type=float, default=0.5)
    ap.add_argument("--balance_answers", action="store_true",
                    help="equalise the oracle-correct answer distribution, so "
                         "reward maximisation cannot be served by a constant answer")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--baseline_momentum", type=float, default=0.99)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--kl_beta", type=float, default=0.05,
                    help="Weight on KL(pi_theta || pi_ref), folded into the reward.")
    ap.add_argument("--min_parse_rate", type=float, default=0.5,
                    help="Below this the policy is treated as degenerating.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Consecutive sub-threshold steps before stopping.")
    ap.add_argument("--checkpoint_every", type=int, default=25,
                    help="Snapshot cadence for the rollback checkpoint.")
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
