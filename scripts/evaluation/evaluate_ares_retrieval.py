"""
ARES-style Retrieval-Augmented Baseline for LEMO benchmark.

Implements a retrieval-augmented verification approach that:
  1. Indexes all training examples using TF-IDF over rules text
  2. For each test example, retrieves k=5 nearest training examples
  3. Combines two strategies:
     (a) Majority vote of retrieved labels (pure retrieval)
     (b) Hybrid: retrieval confidence gate + symbolic VeriCoT fallback

This addresses the reviewer's request for comparison with
"premise verification via RAG" style methods (cf. ARES framework,
Saad-Falcon et al. 2023).

The key question: can retrieval-based augmentation compensate for
V4 rule rewriting or V3 rule removal without any model training?

Usage:
  python scripts/evaluation/evaluate_ares_retrieval.py \
      --train_pred_dir trained_models/qwen3_rlvf/predictions \
      --test_pred_files trained_models/qwen3_rlvf/predictions/qwen3_base_predictions.csv ... \
      --output_dir results/baselines/ares_retrieval
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict

# Reuse symbolic predictor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_vericot_symbolic import symbolic_predict, parse_facts, parse_rules, forward_chain


# ---------------------------------------------------------------------------
# TF-IDF helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())


def build_tfidf_index(docs):
    """
    docs: list of strings
    Returns (tfidf_vectors, idf_map) where each vector is {term: tfidf_weight}.
    """
    N = len(docs)
    df = Counter()
    tok_docs = []
    for doc in docs:
        toks = set(tokenize(doc))
        tok_docs.append(toks)
        for t in toks:
            df[t] += 1

    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    vectors = []
    for tok_set in tok_docs:
        vec = {}
        toks_list = list(tok_set)
        for t in toks_list:
            vec[t] = idf.get(t, 1.0)
        # L2-normalise
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
        vec = {t: v / norm for t, v in vec.items()}
        vectors.append(vec)

    return vectors, idf


def tfidf_vec(text, idf):
    toks = set(tokenize(text))
    vec = {t: idf.get(t, 1.0) for t in toks if t in idf}
    norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def cosine(v1, v2):
    common = set(v1) & set(v2)
    return sum(v1[t] * v2[t] for t in common)


def retrieve_topk(query_vec, index_vecs, labels, k=5):
    """Return top-k (similarity, label) pairs."""
    scores = [(cosine(query_vec, iv), lbl) for iv, lbl in zip(index_vecs, labels)]
    scores.sort(key=lambda x: -x[0])
    return scores[:k]


# ---------------------------------------------------------------------------
# Build training index
# ---------------------------------------------------------------------------

def load_predictions_dir(pred_dir):
    """Load all *_predictions.csv files from a directory. Return list of rows."""
    rows = []
    if not os.path.isdir(pred_dir):
        return rows
    for fname in os.listdir(pred_dir):
        if not fname.endswith('_predictions.csv'):
            continue
        with open(os.path.join(pred_dir, fname), newline='') as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# ARES prediction strategies
# ---------------------------------------------------------------------------

def ares_majority_vote(retrieved):
    """Pure retrieval: majority vote over top-k retrieved labels."""
    labels = [lbl for _, lbl in retrieved]
    cnt = Counter(labels)
    return cnt.most_common(1)[0][0]


def ares_hybrid(retrieved, facts, rules, question, threshold=0.5):
    """
    Hybrid: if top-1 similarity > threshold use retrieval majority vote;
    otherwise fall back to symbolic VeriCoT.
    """
    if retrieved and retrieved[0][0] >= threshold:
        return ares_majority_vote(retrieved)
    return symbolic_predict(facts, rules, question)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_file(csv_path, index_vecs, index_labels, idf, output_dir, split_name, k=5):
    rows_in = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows_in.append(row)

    correct_mv = correct_hyb = correct_sym = 0
    rows_out = []
    for row in rows_in:
        facts    = row.get('facts', '')
        rules    = row.get('rules', '')
        question = row.get('question', '')
        gt       = row.get('ground_truth', '').strip()

        query_text = rules + ' ' + facts + ' ' + question
        qvec = tfidf_vec(query_text, idf)
        retrieved = retrieve_topk(qvec, index_vecs, index_labels, k=k)

        pred_mv  = ares_majority_vote(retrieved)
        pred_hyb = ares_hybrid(retrieved, facts, rules, question)
        pred_sym = symbolic_predict(facts, rules, question)

        if pred_mv  == gt: correct_mv  += 1
        if pred_hyb == gt: correct_hyb += 1
        if pred_sym == gt: correct_sym += 1

        rows_out.append({**row,
                         'ares_majority_pred': pred_mv,
                         'ares_hybrid_pred': pred_hyb,
                         'symbolic_pred': pred_sym,
                         'top1_sim': f'{retrieved[0][0]:.4f}' if retrieved else '0',
                         'correct_mv': int(pred_mv == gt),
                         'correct_hyb': int(pred_hyb == gt)})

    n = len(rows_in)
    acc_mv  = correct_mv  / n if n else 0.0
    acc_hyb = correct_hyb / n if n else 0.0
    acc_sym = correct_sym / n if n else 0.0
    print(f'  [{split_name}]  MajVote={acc_mv:.4f}  Hybrid={acc_hyb:.4f}  Symbolic={acc_sym:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'ares_{split_name}_predictions.csv')
    if rows_out:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    return split_name, acc_mv, acc_hyb, acc_sym, n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pred_dir', required=True,
                        help='Directory of prediction CSVs used as retrieval index')
    parser.add_argument('--test_pred_files', nargs='+', required=True,
                        help='Per-question prediction CSVs to evaluate')
    parser.add_argument('--output_dir', default='results/baselines/ares_retrieval')
    parser.add_argument('--k', type=int, default=5, help='Retrieval top-k')
    parser.add_argument('--hybrid_threshold', type=float, default=0.5)
    args = parser.parse_args()

    # --- Build index ---
    print('Building retrieval index from:', args.train_pred_dir)
    train_rows = load_predictions_dir(args.train_pred_dir)
    print(f'  Index size: {len(train_rows)} examples')

    index_texts  = [r.get('rules', '') + ' ' + r.get('facts', '') + ' ' + r.get('question', '')
                    for r in train_rows]
    index_labels = [r.get('ground_truth', 'F').strip() for r in train_rows]

    index_vecs, idf = build_tfidf_index(index_texts)

    # --- Evaluate ---
    print(f'\n=== ARES Retrieval Baseline (k={args.k}, hybrid_threshold={args.hybrid_threshold}) ===\n')
    print(f'  {"Split":<40} {"MajVote":>10} {"Hybrid":>10} {"Symbolic":>10}')
    print('  ' + '-' * 74)

    summary = []
    for path in args.test_pred_files:
        split = os.path.splitext(os.path.basename(path))[0].replace('_predictions', '')
        for pfx in ['qwen3_rlvf_', 'qwen3_lire_', 'qwen3_', 'qwen_rlvf_', 'qwen_lire_',
                    'qwen_', 'llama_', 'bert_']:
            if split.startswith(pfx):
                split = split[len(pfx):]
                break
        name, acc_mv, acc_hyb, acc_sym, n = evaluate_file(
            path, index_vecs, index_labels, idf, args.output_dir, split, k=args.k)
        summary.append((name, acc_mv, acc_hyb, acc_sym, n))

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'ares_accuracy_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'ares_majority_vote', 'ares_hybrid', 'symbolic_vericot', 'total'])
        for name, acc_mv, acc_hyb, acc_sym, n in summary:
            writer.writerow([name, f'{acc_mv:.6f}', f'{acc_hyb:.6f}', f'{acc_sym:.6f}', n])
    print(f'\n  Saved: {summary_path}')
    print('  ✅ ARES evaluation FINISHED.\n')


if __name__ == '__main__':
    main()
