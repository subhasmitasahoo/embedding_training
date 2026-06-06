import os
import json
import numpy as np
import torch

from tokenizer import SimpleTokenizer
from model import build_model


# ── Load fine-tuned model ──────────────────────────────────────────────────────

def load_finetuned_model(model_dir: str):
    """
    Rebuild the model architecture and load saved weights from model_dir.

    Why we need this:
        torch.save(model.state_dict()) only saves the weight tensors —
        not the architecture itself. To reload, we must:
          1. Rebuild the tokenizer (so we know vocab_size)
          2. Rebuild the model architecture (build_model)
          3. Load the saved weights into it (load_state_dict)

    Some more details:
    SimpleTokenizer.load(tok_path) — reads back the word2idx dictionary we saved after fitting on training data. The model's embedding table has exactly vocab_size rows — if we rebuild with a different vocabulary, the weights won't match.
    build_model(tokenizer) — creates the full architecture (word_emb + pos_emb + 2 TransformerBlocks) with random weights. At this point it's a blank slate.
    torch.load(model_path, map_location="cpu") — reads the .pt file. Returns a dictionary like {"word_emb.weight": tensor(...), "layers.0.attn.q_proj.weight": tensor(...), ...}.
    model.load_state_dict(state_dict) — copies each saved tensor into the matching layer. After this line, the model has the exact weights from the end of training.
    model.eval() — flips a flag that disables dropout. During training, dropout randomly zeros ~10% of values to prevent overfitting. During evaluation, you want the full signal every time — deterministic outputs.

    Args:
        model_dir: path to directory containing model.pt and tokenizer.json

    Returns:
        model     : MiniIntentEmbedder in eval mode, ready for inference
        tokenizer : SimpleTokenizer with saved vocabulary
        config    : dict of hyperparameters used during training
    """

    # ── Step 1: load tokenizer ─────────────────────────────────────
    # tokenizer.json holds the vocabulary (word → id mappings)
    # we need it before building the model so vocab_size is correct
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer = SimpleTokenizer.load(tok_path)
    print(f"  Tokenizer  : {tokenizer.vocab_size} tokens  ← loaded from {tok_path}")

    # ── Step 2: rebuild model architecture ────────────────────────
    # build_model(tokenizer) creates a fresh MiniIntentEmbedder
    # with random weights — same architecture as during training
    model = build_model(tokenizer)

    # ── Step 3: load saved weights ─────────────────────────────────
    # state_dict = all learned weight tensors (embedding table, Q/K/V, FFN ...)
    # map_location="cpu" ensures it loads even if saved on GPU
    model_path = os.path.join(model_dir, "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"  Weights    : {model.n_parameters():,} params  ← loaded from {model_path}")

    # ── Step 4: switch to eval mode ────────────────────────────────
    # model.eval() disables dropout so inference is deterministic
    # (dropout randomly zeros values during training but must be off at test time)
    model.eval()

    # ── Step 5: load training config ──────────────────────────────
    # config.json holds hyperparameters + per-epoch history
    # useful for understanding what settings produced this model
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"  Config     : temp={config['temperature']}  "
          f"epochs={config['epochs']}  lr={config['lr']}")

    return model, tokenizer, config

# quick test — run from customer_intent_search/
# python3 -c "
# from evaluate import load_finetuned_model
# model, tok, cfg = load_finetuned_model('../models/finetuned')
# print('OK')
# "

from data_preparation import load_dataset


# ── Build retrieval corpus ─────────────────────────────────────────────────────
# What each part does:

# queries[0] → corpus — the first query becomes the "anchor" in the corpus. At query time, a new customer message is embedded and compared against all 150 corpus entries. Whichever corpus entry is closest determines the predicted intent.

# queries[1:] → eval set — every other test query becomes something to retrieve. The model sees "track #ORD-1234" and must rank the corpus entry for track_order at position 1.

# Visually what the corpus + query split looks like:

# intent: track_order
#   corpus[0]  → "where is my order"          ← the reference
#   query[0]   → "has my package shipped"     ← must retrieve corpus[0]
#   query[1]   → "can you track #ORD-1234"    ← must retrieve corpus[0]
#   query[2]   → "my delivery hasn't arrived" ← must retrieve corpus[0]
#   ...
# intent: cancel_order
#   corpus[1]  → "I want to cancel my order"  ← the reference
#   query[30]  → "please cancel #ORD-5678"    ← must retrieve corpus[1]
#   ...
# The harder the queries are (different phrasing, slang, typos), the more the model has to rely on semantic understanding rather than keyword overlap — which is exactly what fine-tuning is supposed to provide.

def build_corpus_and_queries(test_data: dict):
    """
    Split the test set into a corpus and evaluation queries.

    How retrieval evaluation works:
        corpus  = one representative query per intent
                  this is what we search against — like a catalogue of known intents
        queries = all remaining test queries
                  these are the "incoming customer messages" we need to classify

        For each eval query:
            embed it → find nearest corpus entry → check if intent matches

    Why one representative per intent?
        In production, you would have a small set of canonical examples per intent.
        The model maps a new query to the nearest canonical example.
        This mirrors the real use case exactly.

    Args:
        test_data : dict of {intent_id: [list of query strings]}

    Returns:
        corpus_texts   : list of strings — one per intent (150 total)
        corpus_labels  : list of intent_ids matching corpus_texts
        query_texts    : list of strings — all remaining test queries (~4,200 total)
        query_labels   : list of intent_ids matching query_texts
    """

    corpus_texts  = []   # representative query for each intent
    corpus_labels = []   # intent id for each corpus entry
    query_texts   = []   # all remaining queries to evaluate on
    query_labels  = []   # intent id for each eval query

    for intent_id, queries in test_data.items():

        # first query → corpus representative
        # e.g. "where is my order" represents track_order in the corpus
        corpus_texts.append(queries[0])
        corpus_labels.append(intent_id)

        # all remaining queries → eval set
        # e.g. "has my package shipped", "track #ORD-1234", ...
        # the model must retrieve the correct corpus entry for each of these
        for q in queries[1:]:
            query_texts.append(q)
            query_labels.append(intent_id)

    print(f"\nRetrieval corpus:")
    print(f"  Corpus size  : {len(corpus_texts):,} entries  (1 per intent)")
    print(f"  Eval queries : {len(query_texts):,} queries  ({len(query_texts) // len(corpus_texts)} per intent on avg)")

    return corpus_texts, corpus_labels, query_texts, query_labels


def build_corpus_centroid(test_data: dict, model, tokenizer):
    """
    Select corpus representative per intent as the most-central query —
    the one whose embedding is closest to the mean of all intent embeddings.

    Why this is better than always picking query[0]:
        query[0] is arbitrary — it might use unusual phrasing that makes it a
        poor representative (e.g. "what are my coffers at" for `balance`).
        The centroid query is the most *typical* phrasing for the intent, so
        every incoming query has the best chance of matching it.

    Algorithm:
        1. Embed all test queries for intent i  →  E_i  shape (n_i, D)
        2. Compute mean embedding               →  mu_i shape (D,)  (re-normalised)
        3. Pick query j = argmax cosine(E_i[j], mu_i)
        4. Use query j as corpus entry; all others become eval queries

    Args:
        test_data  : dict {intent_id: [query strings]}
        model      : trained MiniIntentEmbedder in eval mode
        tokenizer  : fitted SimpleTokenizer

    Returns:
        corpus_texts, corpus_labels, query_texts, query_labels
    """
    import numpy as np

    corpus_texts  = []
    corpus_labels = []
    query_texts   = []
    query_labels  = []

    for intent_id, queries in test_data.items():
        queries = list(queries)

        # embed all queries for this intent
        embs = embed_with_model(queries, model, tokenizer)          # (n_i, D)

        # centroid = mean embedding, re-normalised to unit length
        mu   = embs.mean(axis=0)
        norm = np.linalg.norm(mu)
        if norm > 0:
            mu = mu / norm

        # pick the query closest to the centroid
        sims    = embs @ mu                                         # (n_i,)
        best_i  = int(np.argmax(sims))

        corpus_texts.append(queries[best_i])
        corpus_labels.append(intent_id)

        for i, q in enumerate(queries):
            if i != best_i:
                query_texts.append(q)
                query_labels.append(intent_id)

    print(f"\nRetrieval corpus (centroid selection):")
    print(f"  Corpus size  : {len(corpus_texts):,} entries  (1 per intent — most central query)")
    print(f"  Eval queries : {len(query_texts):,} queries")

    return corpus_texts, corpus_labels, query_texts, query_labels


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ── Embedding functions ────────────────────────────────────────────────────────

def embed_with_model(texts: list, model, tokenizer) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised vectors using MiniIntentEmbedder.

    Args:
        texts     : list of raw query strings
        model     : MiniIntentEmbedder in eval mode
        tokenizer : fitted SimpleTokenizer

    Returns:
        numpy array of shape (len(texts), 128)
        each row is a unit vector (length = 1.0)
    """

    # model.encode() handles batching internally (default batch_size=64)
    # returns numpy array [N, 128] — already L2 normalised inside forward()
    # torch.no_grad() is applied inside encode() so no gradient tracking
    embeddings = model.encode(texts, tokenizer)

    # shape check so we catch silent bugs early
    assert embeddings.shape == (len(texts), 128), \
        f"Expected ({len(texts)}, 128), got {embeddings.shape}"

    return embeddings


def embed_with_tfidf(
    corpus_texts: list,
    query_texts:  list,
) -> tuple:
    """
    Encode texts using TF-IDF vectors (classic keyword-overlap baseline).

    Why TF-IDF:
        TF-IDF = Term Frequency × Inverse Document Frequency
        It represents text as a sparse vector of word importance scores.
        No training needed — purely based on word overlap.
        A new query and a corpus entry are similar if they share rare words.

        This is the baseline every semantic model must beat.
        If fine-tuned model ≈ TF-IDF, training added nothing.

    How it works here:
        1. Fit TfidfVectorizer on corpus_texts  ← learns which words are rare/common
        2. Transform corpus_texts → sparse matrix [n_corpus, vocab]
        3. Transform query_texts  → sparse matrix [n_queries, vocab]
        4. L2-normalise both so cosine similarity = dot product (same as model)

    Args:
        corpus_texts : list of corpus strings  ← fit vocabulary on these
        query_texts  : list of eval query strings

    Returns:
        corpus_vecs : numpy array (n_corpus, tfidf_vocab_size)
        query_vecs  : numpy array (n_queries, tfidf_vocab_size)
    """

    # fit on corpus only — in production you only know the corpus in advance
    # sublinear_tf=True: uses log(tf) instead of raw tf — reduces impact of
    #   very frequent words like "my", "I", "the"
    # min_df=1: keep every word that appears at least once
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
    corpus_vecs = vectorizer.fit_transform(corpus_texts)   # sparse [n_corpus, V]

    # transform queries using the same vocabulary learned from corpus
    # words in queries but not in corpus → ignored (OOV)
    query_vecs = vectorizer.transform(query_texts)         # sparse [n_queries, V]

    # L2 normalise: make every vector unit length
    # now cosine_similarity(a, b) == dot(a, b) — same formula as model embeddings
    corpus_vecs = normalize(corpus_vecs, norm="l2")
    query_vecs  = normalize(query_vecs,  norm="l2")

    # convert sparse → dense numpy for consistent downstream handling
    corpus_vecs = corpus_vecs.toarray()
    query_vecs  = query_vecs.toarray()

    print(f"  TF-IDF vocab size : {len(vectorizer.vocabulary_):,} words")

    return corpus_vecs, query_vecs


# ── Retrieval metrics ──────────────────────────────────────────────────────────

# ── Retrieval metrics ──────────────────────────────────────────────────────────

def retrieval_metrics(
    query_vecs:    np.ndarray,   # (n_queries, dim)
    corpus_vecs:   np.ndarray,   # (n_corpus, dim)
    query_labels:  list,         # intent id for each query
    corpus_labels: list,         # intent id for each corpus entry
) -> dict:
    """
    Compute Recall@1, Recall@5 and MRR at two levels:
        - micro  : every query counts equally (overall system performance)
        - macro  : every intent counts equally (catches weak intents)
        - per_intent : full breakdown per intent (diagnostic detail)

    Micro vs Macro example (3 intents, unequal query counts):
        intent_0: 50 queries, Recall@1 = 0.90  → contributes 45 correct
        intent_1:  5 queries, Recall@1 = 0.20  → contributes  1 correct
        intent_2: 45 queries, Recall@1 = 0.80  → contributes 36 correct

        micro  = (45+1+36) / (50+5+45) = 82/100 = 0.82  ← dominated by big intents
        macro  = (0.90+0.20+0.80) / 3  = 1.90/3  = 0.63 ← intent_1 failure is visible
    """

    # ── Step 1: similarity matrix ──────────────────────────────────
    # dot product of L2-normalised vectors = cosine similarity
    # sim[i][j] = how similar query i is to corpus entry j
    # shape: (n_queries, n_corpus)
    sim = np.dot(query_vecs, corpus_vecs.T)

    # ── Step 2: rank corpus entries for each query ─────────────────
    # argsort ascending → flip for descending (most similar first)
    ranked = np.argsort(-sim, axis=1)   # shape: (n_queries, n_corpus)

    corpus_labels_arr = np.array(corpus_labels)

    # ── Step 3: per-query results ──────────────────────────────────
    # store individual results so we can aggregate both ways
    per_query_correct_1 = []   # True/False per query for Recall@1
    per_query_correct_5 = []   # True/False per query for Recall@5
    per_query_rr        = []   # reciprocal rank per query for MRR
    per_query_intents   = []   # which intent each query belongs to

    for i, query_label in enumerate(query_labels):

        ranked_labels = corpus_labels_arr[ranked[i]]

        # Recall@1
        hit_1 = int(ranked_labels[0] == query_label)

        # Recall@5
        hit_5 = int(query_label in ranked_labels[:5])

        # Reciprocal Rank
        correct_positions = np.where(ranked_labels == query_label)[0]
        rr = 1.0 / (correct_positions[0] + 1) if len(correct_positions) > 0 else 0.0

        per_query_correct_1.append(hit_1)
        per_query_correct_5.append(hit_5)
        per_query_rr.append(rr)
        per_query_intents.append(query_label)

    # ── Step 4: micro-average ──────────────────────────────────────
    # every query counts equally — standard IR benchmark number
    n_queries = len(query_labels)
    micro = {
        "recall_at_1": sum(per_query_correct_1) / n_queries,
        "recall_at_5": sum(per_query_correct_5) / n_queries,
        "mrr":         sum(per_query_rr)         / n_queries,
    }

    # ── Step 5: per-intent breakdown ──────────────────────────────
    # group query results by intent, compute metrics per intent
    # e.g. intent_id 5 → all queries that belong to intent 5
    from collections import defaultdict
    intent_correct_1 = defaultdict(list)
    intent_correct_5 = defaultdict(list)
    intent_rr        = defaultdict(list)

    for hit1, hit5, rr, intent_id in zip(
        per_query_correct_1, per_query_correct_5, per_query_rr, per_query_intents
    ):
        intent_correct_1[intent_id].append(hit1)
        intent_correct_5[intent_id].append(hit5)
        intent_rr[intent_id].append(rr)

    # compute metric for each intent individually
    per_intent = {}
    for intent_id in sorted(intent_correct_1.keys()):
        per_intent[intent_id] = {
            "recall_at_1": sum(intent_correct_1[intent_id]) / len(intent_correct_1[intent_id]),
            "recall_at_5": sum(intent_correct_5[intent_id]) / len(intent_correct_5[intent_id]),
            "mrr":         sum(intent_rr[intent_id])         / len(intent_rr[intent_id]),
            "n_queries":   len(intent_correct_1[intent_id]),
        }

    # ── Step 6: macro-average ──────────────────────────────────────
    # every intent counts equally — reveals if some intents are failing
    n_intents = len(per_intent)
    macro = {
        "recall_at_1": sum(v["recall_at_1"] for v in per_intent.values()) / n_intents,
        "recall_at_5": sum(v["recall_at_5"] for v in per_intent.values()) / n_intents,
        "mrr":         sum(v["mrr"]         for v in per_intent.values()) / n_intents,
    }

    return {
        "micro":      micro,
        "macro":      macro,
        "per_intent": per_intent,
    }

import matplotlib.pyplot as plt


# ── Evaluation runner ──────────────────────────────────────────────────────────

def run_evaluation(model_dir: str, results_dir: str, centroid_corpus: bool = False):
    """
    Full evaluation pipeline — loads model, runs all three baselines,
    prints comparison table, saves bar chart.

    Three models evaluated:
        1. Random-init  — same architecture, no training (measures architecture alone)
        2. TF-IDF       — keyword overlap baseline (classic NLP baseline)
        3. Fine-tuned   — our trained model (what contrastive learning adds)

    Args:
        centroid_corpus : if True, select corpus entry per intent as the most-central
                          query (closest to mean embedding) instead of always query[0].
    """

    print("=" * 60)
    print("  EVALUATION — MiniIntentEmbedder vs Baselines")
    print(f"  Corpus mode : {'centroid (most-central query)' if centroid_corpus else 'fixed (query[0])'}")
    print("=" * 60)

    # ── Step 1: load data ──────────────────────────────────────────
    print("\nLoading dataset...")
    _, _, test_data = load_dataset()

    # ── Step 2: load fine-tuned model ─────────────────────────────
    print("\nLoading fine-tuned model...")
    finetuned_model, tokenizer, config = load_finetuned_model(model_dir)

    # ── Build corpus (must happen after model load for centroid mode) ──
    if centroid_corpus:
        print("\nBuilding centroid corpus (embedding all test queries per intent)...")
        corpus_texts, corpus_labels, query_texts, query_labels = \
            build_corpus_centroid(test_data, finetuned_model, tokenizer)
    else:
        corpus_texts, corpus_labels, query_texts, query_labels = \
            build_corpus_and_queries(test_data)

    # ── Step 3: build random-init model ───────────────────────────
    # same architecture + same tokenizer, but weights never updated
    # gives us a floor: what does the transformer give for free?
    print("\nBuilding random-init model...")
    random_model = build_model(tokenizer)
    random_model.eval()
    print(f"  Parameters : {random_model.n_parameters():,}  (random weights, no training)")

    # ── Step 4: embed corpus + queries with all three approaches ───
    print("\nEmbedding corpus and queries...")

    print("  [1/3] Random-init model...")
    random_corpus = embed_with_model(corpus_texts, random_model, tokenizer)
    random_queries = embed_with_model(query_texts,  random_model, tokenizer)

    print("  [2/3] TF-IDF...")
    tfidf_corpus, tfidf_queries = embed_with_tfidf(corpus_texts, query_texts)

    print("  [3/3] Fine-tuned model...")
    ft_corpus  = embed_with_model(corpus_texts, finetuned_model, tokenizer)
    ft_queries = embed_with_model(query_texts,  finetuned_model, tokenizer)

    # ── Step 5: compute retrieval metrics for all three ────────────
    print("\nComputing retrieval metrics...")

    results = {}
    results["Random-init"] = retrieval_metrics(
        random_queries, random_corpus, query_labels, corpus_labels
    )
    results["TF-IDF"] = retrieval_metrics(
        tfidf_queries, tfidf_corpus, query_labels, corpus_labels
    )
    results["Fine-tuned"] = retrieval_metrics(
        ft_queries, ft_corpus, query_labels, corpus_labels
    )

    # ── Step 6: print comparison table ────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS — Micro average (all queries equally weighted)")
    print("=" * 60)
    print(f"  {'Model':<15} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8}")
    print("  " + "─" * 48)
    for model_name, res in results.items():
        m = res["micro"]
        print(f"  {model_name:<15} | "
              f"{m['recall_at_1']*100:>8.1f}% | "
              f"{m['recall_at_5']*100:>8.1f}% | "
              f"{m['mrr']:>8.4f}")

    print("\n" + "=" * 60)
    print("  RESULTS — Macro average (all intents equally weighted)")
    print("=" * 60)
    print(f"  {'Model':<15} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8}")
    print("  " + "─" * 48)
    for model_name, res in results.items():
        m = res["macro"]
        print(f"  {model_name:<15} | "
              f"{m['recall_at_1']*100:>8.1f}% | "
              f"{m['recall_at_5']*100:>8.1f}% | "
              f"{m['mrr']:>8.4f}")

    # uplift
    ft_r1    = results["Fine-tuned"]["micro"]["recall_at_1"]
    tfidf_r1 = results["TF-IDF"]["micro"]["recall_at_1"]
    uplift   = (ft_r1 - tfidf_r1) * 100
    print(f"\n  Fine-tuned vs TF-IDF uplift: {uplift:+.1f}pp Recall@1 (micro)")

    # ── Step 6b: per-intent breakdown for fine-tuned ──────────────
    print("\n" + "=" * 60)
    print("  PER-INTENT BREAKDOWN — Fine-tuned model")
    print("=" * 60)

    ft_per_intent = results["Fine-tuned"]["per_intent"]

    # sort intents by Recall@1 ascending → worst first
    sorted_intents = sorted(ft_per_intent.items(),
                            key=lambda x: x[1]["recall_at_1"])

    print("\n  Worst 10 intents:")
    print(f"  {'Intent ID':<12} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8} | {'Queries':>7}")
    print("  " + "─" * 55)
    for intent_id, m in sorted_intents[:10]:
        print(f"  {str(intent_id):<12} | "
              f"{m['recall_at_1']*100:>8.1f}% | "
              f"{m['recall_at_5']*100:>8.1f}% | "
              f"{m['mrr']:>8.4f} | "
              f"{m['n_queries']:>7}")

    print("\n  Best 10 intents:")
    print(f"  {'Intent ID':<12} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8} | {'Queries':>7}")
    print("  " + "─" * 55)
    for intent_id, m in sorted_intents[-10:][::-1]:
        print(f"  {str(intent_id):<12} | "
              f"{m['recall_at_1']*100:>8.1f}% | "
              f"{m['recall_at_5']*100:>8.1f}% | "
              f"{m['mrr']:>8.4f} | "
              f"{m['n_queries']:>7}")

    # ── Step 7: save results to JSON ──────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Step 8: plot comparison bar chart ─────────────────────────
    plot_path = _plot_eval_comparison(results, results_dir)
    print(f"  Plot    → {plot_path}")

    return results


def _plot_eval_comparison(results: dict, results_dir: str) -> str:
    """
    Save a 3-panel bar chart comparing all three models across all three metrics.

    Panel 1: Recall@1   — primary metric, did we get the right intent first?
    Panel 2: Recall@5   — is the right intent anywhere in top 5?
    Panel 3: MRR        — how high up is the right intent on average?
    """

    models  = list(results.keys())
    colors  = ["#95a5a6", "#e67e22", "#2ecc71"]   # grey, orange, green

    # NEW
    r1  = [results[m]["micro"]["recall_at_1"] * 100 for m in models]
    r5  = [results[m]["micro"]["recall_at_5"] * 100 for m in models]
    mrr = [results[m]["micro"]["mrr"]               for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Evaluation: Fine-tuned vs Baselines", fontsize=14, fontweight="bold")

    for ax, values, title, ylabel, fmt in zip(
        axes,
        [r1,  r5,  mrr],
        ["Recall@1", "Recall@5", "MRR"],
        ["Accuracy (%)", "Accuracy (%)", "MRR Score"],
        [".1f", ".1f", ".4f"],
    ):
        bars = ax.bar(models, values, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.5)

        # label each bar with its value
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.5 if "%" in ylabel else 0.005),
                f"{val:{fmt}}{'%' if '%' in ylabel else ''}",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "eval_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MiniIntentEmbedder")
    parser.add_argument("--model-dir",       type=str, default="../models/finetuned")
    parser.add_argument("--results-dir",     type=str, default="../results")
    parser.add_argument("--centroid-corpus", action="store_true",
                        help="Select corpus entry per intent as the most-central query "
                             "(closest to mean embedding) instead of always query[0].")
    args = parser.parse_args()

    run_evaluation(args.model_dir, args.results_dir,
                   centroid_corpus=args.centroid_corpus)