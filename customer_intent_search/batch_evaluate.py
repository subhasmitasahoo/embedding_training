"""
batch_evaluate.py — evaluate every model checkpoint and print one comparison table.

Loads test data, TF-IDF baseline, and random-init model ONCE, then runs
retrieval_metrics() on every checkpoint.  Much faster than calling evaluate.py
N times (no repeated dataset loading or TF-IDF fitting).

Usage:
    python3 batch_evaluate.py                  # all checkpoints
    python3 batch_evaluate.py --skip-stability # skip the 10 stability seeds
"""

import os
import sys
import argparse
import numpy as np
import torch

# reuse everything from evaluate.py
from evaluate import (
    load_finetuned_model,
    build_corpus_and_queries,
    embed_with_model,
    embed_with_tfidf,
    retrieval_metrics,
)
from data_preparation import load_dataset
from model import build_model


# ── Checkpoint registry ────────────────────────────────────────────────────────
# Each entry: (label shown in table, relative path from repo root, group)
# v0 is skipped — synthetic 20-intent data, not comparable

CHECKPOINTS = [
    # ── baselines ──────────────────────────────────────────────────────────────
    ("finetuned (default)",          "models/finetuned",                          "baseline"),

    # ── temperature experiments ────────────────────────────────────────────────
    ("temp τ=0.01",                  "models/experiments/temp_0.01",              "temperature"),
    ("temp τ=0.05",                  "models/experiments/temp_0.05",              "temperature"),
    ("temp τ=0.20",                  "models/experiments/temp_0.20",              "temperature"),

    # ── v1 (dynamic only, first CLINC run) ────────────────────────────────────
    ("v1 dynamic no_hn",             "models/hn_experiments/v1/no_hn",            "v1"),
    ("v1 dynamic hn_k=1",            "models/hn_experiments/v1/hn_k_1",           "v1"),
    ("v1 dynamic hn_k=3",            "models/hn_experiments/v1/hn_k_3",           "v1"),
    ("v1 dynamic hn_k=5",            "models/hn_experiments/v1/hn_k_5",           "v1"),

    # ── v2 static ─────────────────────────────────────────────────────────────
    ("v2 static  no_hn",             "models/hn_experiments/v2/static/no_hn",     "v2-static"),
    ("v2 static  hn_k=1",            "models/hn_experiments/v2/static/hn_k_1",    "v2-static"),
    ("v2 static  hn_k=3",            "models/hn_experiments/v2/static/hn_k_3",    "v2-static"),
    ("v2 static  hn_k=5",            "models/hn_experiments/v2/static/hn_k_5",    "v2-static"),

    # ── v2 dynamic ────────────────────────────────────────────────────────────
    ("v2 dynamic no_hn",             "models/hn_experiments/v2/dynamic/no_hn",    "v2-dynamic"),
    ("v2 dynamic hn_k=1",            "models/hn_experiments/v2/dynamic/hn_k_1",   "v2-dynamic"),
    ("v2 dynamic hn_k=3",            "models/hn_experiments/v2/dynamic/hn_k_3",   "v2-dynamic"),
    ("v2 dynamic hn_k=5",            "models/hn_experiments/v2/dynamic/hn_k_5",   "v2-dynamic"),

    # ── stability seeds ────────────────────────────────────────────────────────
    ("stability seed=7",             "models/stability/seed_7",                   "stability"),
    ("stability seed=23",            "models/stability/seed_23",                  "stability"),
    ("stability seed=42",            "models/stability/seed_42",                  "stability"),
    ("stability seed=43",            "models/stability/seed_43",                  "stability"),
    ("stability seed=44",            "models/stability/seed_44",                  "stability"),
    ("stability seed=45",            "models/stability/seed_45",                  "stability"),
    ("stability seed=46",            "models/stability/seed_46",                  "stability"),
    ("stability seed=99",            "models/stability/seed_99",                  "stability"),
    ("stability seed=137",           "models/stability/seed_137",                 "stability"),
    ("stability seed=256",           "models/stability/seed_256",                 "stability"),
]


def _fmt(v: float) -> str:
    return f"{v*100:5.1f}%"


def run_batch_evaluation(repo_root: str, skip_stability: bool = False):

    # ── Step 1: load test data once ────────────────────────────────────────────
    print("\nLoading dataset...")
    _, _, test_data = load_dataset()
    corpus_texts, corpus_labels, query_texts, query_labels = \
        build_corpus_and_queries(test_data)
    print(f"  Corpus : {len(corpus_texts)} intents")
    print(f"  Queries: {len(query_texts)} eval queries")

    # ── Step 2: compute baselines once ────────────────────────────────────────
    print("\nComputing baselines (TF-IDF + random-init) — done once for all checkpoints...")

    print("  [1/2] TF-IDF...")
    tfidf_corpus, tfidf_queries = embed_with_tfidf(corpus_texts, query_texts)
    tfidf_metrics = retrieval_metrics(tfidf_queries, tfidf_corpus, query_labels, corpus_labels)

    # random-init: load any tokenizer (they're all the same vocab) to build architecture
    any_tok_dir = os.path.join(repo_root, "models/finetuned")
    from tokenizer import SimpleTokenizer
    tok = SimpleTokenizer.load(os.path.join(any_tok_dir, "tokenizer.json"))
    rand_model = build_model(tok)
    rand_model.eval()
    print("  [2/2] Random-init model...")
    rand_corpus  = embed_with_model(corpus_texts, rand_model, tok)
    rand_queries = embed_with_model(query_texts,  rand_model, tok)
    rand_metrics = retrieval_metrics(rand_queries, rand_corpus, query_labels, corpus_labels)

    # ── Step 3: evaluate each checkpoint ──────────────────────────────────────
    rows = []  # list of (label, group, r1, r5, mrr)

    checkpoints = [c for c in CHECKPOINTS
                   if not (skip_stability and c[2] == "stability")]

    total = len(checkpoints)
    for idx, (label, rel_path, group) in enumerate(checkpoints, 1):
        model_dir = os.path.join(repo_root, rel_path)
        if not os.path.exists(os.path.join(model_dir, "model.pt")):
            print(f"  [{idx:02d}/{total}] SKIP (not found): {label}")
            continue

        print(f"  [{idx:02d}/{total}] {label}")
        try:
            model, tokenizer, _ = load_finetuned_model(model_dir)
            ft_corpus  = embed_with_model(corpus_texts, model, tokenizer)
            ft_queries = embed_with_model(query_texts,  model, tokenizer)
            m = retrieval_metrics(ft_queries, ft_corpus, query_labels, corpus_labels)
            rows.append((label, group,
                         m["micro"]["recall_at_1"],
                         m["micro"]["recall_at_5"],
                         m["micro"]["mrr"]))
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Step 4: print table ────────────────────────────────────────────────────
    W = 26
    print(f"\n{'='*70}")
    print(f"  BATCH EVALUATION RESULTS — Recall@1 / Recall@5 / MRR (micro)")
    print(f"{'='*70}")
    print(f"  {'Model':<{W}} | {'R@1':>6} | {'R@5':>6} | {'MRR':>6} | Group")
    print("  " + "─" * 62)

    # baselines first
    r  = rand_metrics["micro"]
    tf = tfidf_metrics["micro"]
    print(f"  {'random-init':<{W}} | {_fmt(r['recall_at_1'])} | {_fmt(r['recall_at_5'])} | {_fmt(r['mrr'])} | baseline")
    print(f"  {'TF-IDF':<{W}} | {_fmt(tf['recall_at_1'])} | {_fmt(tf['recall_at_5'])} | {_fmt(tf['mrr'])} | baseline")
    print("  " + "─" * 62)

    last_group = None
    for label, group, r1, r5, mrr in rows:
        if group != last_group and last_group is not None:
            print("  " + "─" * 62)
        last_group = group
        print(f"  {label:<{W}} | {_fmt(r1)} | {_fmt(r5)} | {_fmt(mrr)} | {group}")

    # stability summary row
    stab_rows = [(r1, r5, mrr) for lbl, grp, r1, r5, mrr in rows if grp == "stability"]
    if stab_rows:
        print("  " + "─" * 62)
        arr = np.array(stab_rows)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        print(f"  {'stability MEAN ± std':<{W}} | "
              f"{means[0]*100:4.1f}±{stds[0]*100:.1f}% | "
              f"{means[1]*100:4.1f}±{stds[1]*100:.1f}% | "
              f"{means[2]*100:4.1f}±{stds[2]*100:.1f}% | stability")

    print(f"{'='*70}")

    # ── Step 5: print best model summary ──────────────────────────────────────
    if rows:
        best = max(rows, key=lambda x: x[2])  # sort by R@1
        print(f"\n  Best checkpoint : {best[0]}")
        print(f"  R@1={_fmt(best[2])}  R@5={_fmt(best[3])}  MRR={_fmt(best[4])}")

    # ── Step 6: save JSON ──────────────────────────────────────────────────────
    import json

    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    stab_rows = [(r1, r5, mrr) for _, grp, r1, r5, mrr in rows if grp == "stability"]
    stab_summary = None
    if stab_rows:
        arr = np.array(stab_rows)
        stab_summary = {
            "mean_r1": float(arr[:, 0].mean()), "std_r1": float(arr[:, 0].std()),
            "mean_r5": float(arr[:, 1].mean()), "std_r5": float(arr[:, 1].std()),
            "mean_mrr": float(arr[:, 2].mean()), "std_mrr": float(arr[:, 2].std()),
        }

    output = {
        "baselines": {
            "random_init": {
                "recall_at_1": float(rand_metrics["micro"]["recall_at_1"]),
                "recall_at_5": float(rand_metrics["micro"]["recall_at_5"]),
                "mrr":         float(rand_metrics["micro"]["mrr"]),
            },
            "tfidf": {
                "recall_at_1": float(tfidf_metrics["micro"]["recall_at_1"]),
                "recall_at_5": float(tfidf_metrics["micro"]["recall_at_5"]),
                "mrr":         float(tfidf_metrics["micro"]["mrr"]),
            },
        },
        "checkpoints": [
            {
                "label":       label,
                "group":       group,
                "recall_at_1": float(r1),
                "recall_at_5": float(r5),
                "mrr":         float(mrr),
            }
            for label, group, r1, r5, mrr in rows
        ],
        "stability_summary": stab_summary,
        "best": {
            "label":       best[0],
            "recall_at_1": float(best[2]),
            "recall_at_5": float(best[3]),
            "mrr":         float(best[4]),
        } if rows else None,
    }

    json_path = os.path.join(results_dir, "batch_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results → {json_path}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints")
    parser.add_argument("--repo-root",       type=str, default="..",
                        help="Path to repo root (default: parent of this script)")
    parser.add_argument("--skip-stability",  action="store_true",
                        help="Skip the 10 stability-seed checkpoints")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    run_batch_evaluation(repo_root, skip_stability=args.skip_stability)
