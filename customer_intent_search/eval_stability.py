"""
eval_stability.py — measure how stable the evaluation metric is across
different corpus/query splits.

Current eval always picks query[0] per intent as the corpus representative.
That choice is arbitrary. This script randomly rotates which query becomes
the corpus entry and re-runs retrieval N times, reporting mean ± std.

If variance is low (e.g. ±0.5%), the metric is robust to corpus choice.
If variance is high (e.g. ±3%), the single-split result is noisy and we
should report a mean over multiple splits instead.

Usage:
    python3 eval_stability.py                          # best model, 5 splits
    python3 eval_stability.py --model-dir ../models/hn_experiments/v2/dynamic/hn_k_5
    python3 eval_stability.py --n-splits 10 --seed 0
"""

import os
import sys
import json
import argparse
import random
import numpy as np

from evaluate import (
    load_finetuned_model,
    embed_with_model,
    embed_with_tfidf,
    retrieval_metrics,
)
from data_preparation import load_dataset


def build_corpus_and_queries_split(test_data: dict, corpus_seed: int):
    """
    Like build_corpus_and_queries() but randomly picks which query per intent
    goes into the corpus (instead of always using index 0).

    Args:
        test_data   : dict of {intent_id: [list of query strings]}
        corpus_seed : random seed for reproducible selection

    Returns:
        corpus_texts, corpus_labels, query_texts, query_labels
    """
    rng = random.Random(corpus_seed)

    corpus_texts  = []
    corpus_labels = []
    query_texts   = []
    query_labels  = []

    for intent_id, queries in test_data.items():
        queries = list(queries)
        # pick a random query as corpus representative
        idx = rng.randrange(len(queries))
        corpus_texts.append(queries[idx])
        corpus_labels.append(intent_id)
        # rest → eval queries
        for i, q in enumerate(queries):
            if i != idx:
                query_texts.append(q)
                query_labels.append(intent_id)

    return corpus_texts, corpus_labels, query_texts, query_labels


def _fmt(v: float) -> str:
    return f"{v * 100:5.1f}%"


def run_eval_stability(model_dir: str, n_splits: int = 5, base_seed: int = 42,
                       also_tfidf: bool = True):
    """
    Re-evaluate model across N random corpus/query splits and report mean ± std.

    Args:
        model_dir  : path to model checkpoint
        n_splits   : number of random corpus splits to evaluate (default 5)
        base_seed  : first split seed; subsequent splits use base_seed+1, +2, ...
        also_tfidf : also run TF-IDF for each split (reference baseline)
    """

    print("=" * 65)
    print("  EVALUATION STABILITY — corpus-split randomisation")
    print("=" * 65)
    print(f"  Model     : {model_dir}")
    print(f"  Splits    : {n_splits}  (seeds {base_seed}–{base_seed + n_splits - 1})")
    print(f"  TF-IDF    : {'yes' if also_tfidf else 'no'}")

    # ── load data once ─────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    _, _, test_data = load_dataset()

    # ── load model once ────────────────────────────────────────────────────────
    print("\nLoading model...")
    model, tokenizer, config = load_finetuned_model(model_dir)

    # ── run N splits ───────────────────────────────────────────────────────────
    ft_runs    = []   # list of (r1, r5, mrr) for fine-tuned model
    tfidf_runs = []   # same for TF-IDF

    splits = [base_seed + i for i in range(n_splits)]

    print(f"\n  {'Split':>5} | {'seed':>5} | {'R@1':>6} | {'R@5':>6} | {'MRR':>6}"
          + (f" || {'TF-IDF R@1':>10} | {'R@5':>6} | {'MRR':>6}" if also_tfidf else ""))
    print("  " + "─" * (55 if not also_tfidf else 85))

    for i, seed in enumerate(splits, 1):
        corpus_texts, corpus_labels, query_texts, query_labels = \
            build_corpus_and_queries_split(test_data, corpus_seed=seed)

        # fine-tuned model
        ft_corpus  = embed_with_model(corpus_texts, model, tokenizer)
        ft_queries = embed_with_model(query_texts,  model, tokenizer)
        ft_m = retrieval_metrics(ft_queries, ft_corpus, query_labels, corpus_labels)
        r1  = ft_m["micro"]["recall_at_1"]
        r5  = ft_m["micro"]["recall_at_5"]
        mrr = ft_m["micro"]["mrr"]
        ft_runs.append((r1, r5, mrr))

        if also_tfidf:
            tfidf_corpus, tfidf_queries = embed_with_tfidf(corpus_texts, query_texts)
            tf_m  = retrieval_metrics(tfidf_queries, tfidf_corpus, query_labels, corpus_labels)
            tr1   = tf_m["micro"]["recall_at_1"]
            tr5   = tf_m["micro"]["recall_at_5"]
            tmrr  = tf_m["micro"]["mrr"]
            tfidf_runs.append((tr1, tr5, tmrr))
            print(f"  {i:>5} | {seed:>5} | {_fmt(r1)} | {_fmt(r5)} | {_fmt(mrr)}"
                  f" || {_fmt(tr1):>10} | {_fmt(tr5)} | {_fmt(tmrr)}")
        else:
            print(f"  {i:>5} | {seed:>5} | {_fmt(r1)} | {_fmt(r5)} | {_fmt(mrr)}")

    # ── summary stats ──────────────────────────────────────────────────────────
    arr = np.array(ft_runs)

    print("\n" + "=" * 65)
    print("  FINE-TUNED MODEL — summary across splits")
    print("=" * 65)
    print(f"  {'Metric':<12} | {'Mean':>7} | {'± Std':>7} | {'Min':>7} | {'Max':>7} | {'Range':>7}")
    print("  " + "─" * 55)
    for j, name in enumerate(["Recall@1", "Recall@5", "MRR"]):
        vals = arr[:, j] * 100
        print(f"  {name:<12} | {vals.mean():>6.2f}% | {vals.std():>6.2f}% | "
              f"{vals.min():>6.2f}% | {vals.max():>6.2f}% | {vals.max()-vals.min():>6.2f}%")

    if also_tfidf and tfidf_runs:
        tarr = np.array(tfidf_runs)
        print("\n  TF-IDF BASELINE — summary across splits")
        print("  " + "─" * 55)
        for j, name in enumerate(["Recall@1", "Recall@5", "MRR"]):
            vals = tarr[:, j] * 100
            print(f"  {name:<12} | {vals.mean():>6.2f}% | {vals.std():>6.2f}% | "
                  f"{vals.min():>6.2f}% | {vals.max():>6.2f}% | {vals.max()-vals.min():>6.2f}%")

    # ── save JSON ──────────────────────────────────────────────────────────────
    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(model_dir)),
                               "..", "..", "results")
    # fallback to ../results if above resolution looks wrong
    if not os.path.isdir(results_dir):
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    results_dir = os.path.normpath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "model_dir": model_dir,
        "n_splits": n_splits,
        "seeds": splits,
        "finetuned": {
            "runs": [{"seed": s, "recall_at_1": float(r1), "recall_at_5": float(r5),
                      "mrr": float(mrr)}
                     for s, (r1, r5, mrr) in zip(splits, ft_runs)],
            "mean_r1": float(arr[:, 0].mean()), "std_r1": float(arr[:, 0].std()),
            "mean_r5": float(arr[:, 1].mean()), "std_r5": float(arr[:, 1].std()),
            "mean_mrr": float(arr[:, 2].mean()), "std_mrr": float(arr[:, 2].std()),
        },
    }
    if tfidf_runs:
        tarr = np.array(tfidf_runs)
        output["tfidf"] = {
            "runs": [{"seed": s, "recall_at_1": float(r1), "recall_at_5": float(r5),
                      "mrr": float(mrr)}
                     for s, (r1, r5, mrr) in zip(splits, tfidf_runs)],
            "mean_r1": float(tarr[:, 0].mean()), "std_r1": float(tarr[:, 0].std()),
            "mean_r5": float(tarr[:, 1].mean()), "std_r5": float(tarr[:, 1].std()),
            "mean_mrr": float(tarr[:, 2].mean()), "std_mrr": float(tarr[:, 2].std()),
        }

    json_path = os.path.join(results_dir, "eval_stability_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results → {json_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metric stability across corpus splits")
    parser.add_argument("--model-dir",   type=str,
                        default="../models/stability/seed_44",
                        help="Model checkpoint to evaluate (default: best seed=44)")
    parser.add_argument("--n-splits",    type=int, default=5,
                        help="Number of random corpus splits (default: 5)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Base seed for corpus splits (default: 42)")
    parser.add_argument("--no-tfidf",    action="store_true",
                        help="Skip TF-IDF baseline (faster)")
    args = parser.parse_args()

    run_eval_stability(
        model_dir=args.model_dir,
        n_splits=args.n_splits,
        base_seed=args.seed,
        also_tfidf=not args.no_tfidf,
    )
