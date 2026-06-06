import argparse
import copy
import os
import time
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from model import build_model
from data_preparation import load_dataset, create_training_pairs, IntentAwareBatchSampler


# ── Sibling Intent Groups ──────────────────────────────────────────────────────
# Intents within the same group share nearly identical vocabulary but have
# distinct meanings. Standard in-batch negatives rarely co-occur, so the model
# never learns to separate them. Sibling-aware HN injects at least one query
# from each sibling per anchor — forcing the model to learn the distinction.
#
# Groups are derived from failure analysis (Step 12):
#   - Read vs Write  : calendar/calendar_update, reminder/reminder_update, etc.
#   - Check vs Act   : credit_score/improve_credit_score, pay_bill/bill_due
#   - Shared topic   : recipe/ingredients_list, shopping_list/todo_list variants
#   - Pronoun trap   : user_name/change_user_name/what_is_your_name
#   - Surface trap   : yes/no/maybe

CLINC_SIBLING_GROUPS = [
    ["balance", "pto_balance", "rewards_balance"],
    ["calendar", "calendar_update"],
    ["reminder", "reminder_update"],
    ["pay_bill", "bill_due", "min_payment", "bill_balance"],
    ["credit_score", "improve_credit_score"],
    ["shopping_list", "shopping_list_update", "todo_list", "todo_list_update"],
    ["recipe", "ingredients_list", "ingredient_substitution"],
    ["user_name", "change_user_name", "what_is_your_name"],
    ["yes", "no", "maybe"],
    ["order", "order_status", "cancel_order"],
    ["gas", "gas_type"],
]


def build_sibling_map(intent_groups: dict, intent_name_to_id: dict) -> dict:
    """
    Build a mapping {intent_id: [sibling_intent_id, ...]} from CLINC_SIBLING_GROUPS.

    Only includes intents that are present in intent_groups (training data).
    Intents not listed in any sibling group are excluded — they have no known
    sibling confusion problem and don't need special treatment.

    Args:
        intent_groups      : dict {intent_id: [query strings]} — training set
        intent_name_to_id  : dict {intent_name: intent_id}

    Returns:
        dict {intent_id: [list of sibling intent_ids]}
    """
    present_ids = set(intent_groups.keys())
    sibling_map = {}

    for group in CLINC_SIBLING_GROUPS:
        # resolve names → ids, keep only those present in training data
        group_ids = [intent_name_to_id[n] for n in group
                     if n in intent_name_to_id and intent_name_to_id[n] in present_ids]
        if len(group_ids) < 2:
            continue
        # each intent in the group gets all others as siblings
        for iid in group_ids:
            sibling_map[iid] = [s for s in group_ids if s != iid]

    n_intents = sum(1 for iid in sibling_map if sibling_map[iid])
    print(f"  [Sibling HN] {n_intents} intents have sibling groups: "
          + ", ".join(str(k) for k in sorted(sibling_map.keys())[:5]) + " ...")
    return sibling_map


# ── Loss Function ──────────────────────────────────────────────────────────────

class MultipleNegativesRankingLoss(nn.Module):
    """
    Contrastive loss using all other positives in the batch as negatives.

    For a batch of (anchor, positive) pairs:
      - Embed all anchors   → matrix A shape (batch_size, 128)
      - Embed all positives → matrix P shape (batch_size, 128)
      - Similarity matrix   S = A @ P.T    shape (batch_size, batch_size)
      - Diagonal            = correct matches  → should be HIGH
      - Off-diagonal        = wrong matches    → should be LOW
      - Loss = cross entropy treating diagonal as the correct class

    Hard negative extension (when hard_neg_emb is provided):
      - hard_neg_emb shape: (B * k, D) where k = hn_top_k
      - Reshaped to (B, k, D): each anchor has its own k hard negatives
      - Per-anchor similarity computed via bmm: (B, 1, D) @ (B, D, k) → (B, k)
      - Appended to sim matrix: (B, B) → (B, B + k)
      - Correct label unchanged (still the diagonal index)
      - This forces the model to rank each anchor's hard negs below its true positive
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        # temperature scales scores before softmax
        # low  τ = 0.05 → scores more peaked → harder training signal
        # high τ = 1.00 → scores flatter     → softer training signal
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,           # (B, D)
        positive_emb: torch.Tensor,         # (B, D)
        hard_neg_emb: torch.Tensor = None,  # (B*k, D) or None
    ) -> torch.Tensor:

        # standard MNR: similarity of every anchor against every positive
        # sim[i][j] = similarity of anchor_i with positive_j
        sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature

        # correct label for each anchor = its own index (the diagonal)
        labels = torch.arange(sim.shape[0], device=sim.device)

        if hard_neg_emb is not None:
            B, D = anchor_emb.shape
            # infer k from tensor shape: hard_neg_emb is [B*k, D]
            k = hard_neg_emb.shape[0] // B
            # reshape to [B, k, D]: each anchor gets its own k hard negatives
            hn = hard_neg_emb.view(B, k, D)
            # per-anchor hard neg similarity via batched matmul:
            #   [B, 1, D] @ [B, D, k] → [B, 1, k] → squeeze → [B, k]
            hard_sim = torch.bmm(
                anchor_emb.unsqueeze(1),
                hn.transpose(1, 2),
            ).squeeze(1) / self.temperature
            # append hard neg similarities as extra columns
            # sim: [B, B+k] — model must rank column i highest for each row
            sim = torch.cat([sim, hard_sim], dim=1)

        # cross entropy: correct class = diagonal index (unchanged by concatenation)
        return F.cross_entropy(sim, labels)


# ── Dataset ────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Wraps a list of InputExample pairs for PyTorch DataLoader.
    Each item returns (anchor_text, positive_text, intent_id).
    intent_id is passed through the DataLoader so the collate_fn
    can sample hard negatives from the correct intent pool.
    """

    def __init__(self, pairs, intent_ids):
        # pairs      = list of InputExample objects
        # intent_ids = parallel list of intent IDs, one per pair
        self.pairs      = pairs
        self.intent_ids = intent_ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor   = self.pairs[idx].texts[0]
        positive = self.pairs[idx].texts[1]
        intent   = self.intent_ids[idx]
        return anchor, positive, intent


# ── Collate Function ───────────────────────────────────────────────────────────

def make_collate_fn(tokenizer, hard_neg_map=None, hn_top_k=1):
    """
    Returns a closure that tokenizes a batch of (anchor, positive, intent) tuples.

    hard_neg_map : dict {intent_id: [hard_neg_text, ...]} built by
                   mine_hard_negatives(); None = standard training (no hard negs)
    hn_top_k     : how many hard negatives to sample per anchor.
                   The loss infers k from hard_neg_emb.shape[0] // B.

    Returns a 6-tuple:
        anchor_ids, anchor_mask, positive_ids, positive_mask, hn_ids, hn_mask
    hn_ids / hn_mask are None when hard_neg_map is None.
    """

    def collate_fn(batch):
        anchors   = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        intents   = [item[2] for item in batch]

        anchor_ids,   anchor_mask   = tokenizer.encode_batch(anchors)
        positive_ids, positive_mask = tokenizer.encode_batch(positives)

        if hard_neg_map is not None:
            # for each anchor, sample hn_top_k hard negatives from its intent pool
            # result: B * hn_top_k texts (anchor_0's negs, then anchor_1's, ...)
            hard_neg_texts = []
            for intent in intents:
                pool = hard_neg_map[intent]
                for _ in range(hn_top_k):
                    hard_neg_texts.append(random.choice(pool))
            # tokenize to [B * hn_top_k, seq_len]
            hn_ids, hn_mask = tokenizer.encode_batch(hard_neg_texts)
            return anchor_ids, anchor_mask, positive_ids, positive_mask, hn_ids, hn_mask

        return anchor_ids, anchor_mask, positive_ids, positive_mask, None, None

    return collate_fn


# ── Hard Negative Mining ───────────────────────────────────────────────────────

def mine_hard_negatives(model, tokenizer, intent_groups, top_k=1, device="cpu",
                        sibling_map=None):
    """
    Offline hard negative mining: embed all training queries with the current
    model weights and find the top-K most similar queries from a *different* intent.

    Why this helps:
        In-batch negatives are mostly easy (track_order vs reset_password).
        After a few epochs, easy negatives contribute almost no gradient signal
        and val accuracy plateaus. Hard negatives force the model to sharpen
        the boundary between *confusable* intents (cancel_order vs modify_order).

    Algorithm:
        1. embed all training queries → [N, 128] matrix (L2-normalised)
        2. cosine similarity matrix   → [N, N]  (dot product of unit vectors)
        3. for each query i:
               mask same-intent entries (set score = -2.0)
               take top-K highest-similarity entries from different intents
               add them to that intent's hard-neg pool
        4. repeat every hn_refresh_every epochs as the model improves

    Sibling-aware extension (when sibling_map is provided):
        For each intent with known sibling confusion (e.g. calendar vs
        calendar_update), also inject one random query per sibling into the
        hard-neg pool regardless of embedding similarity. This guarantees the
        model always sees the hardest known confusable pair during training,
        even before embeddings have converged enough to surface them via
        similarity mining.

    Args:
        model         : MiniIntentEmbedder (current weights)
        tokenizer     : fitted SimpleTokenizer
        intent_groups : dict {intent_id: [query strings]} — the training set
        top_k         : how many hard negatives to mine per query
        device        : "cpu" or "cuda"
        sibling_map   : dict {intent_id: [sibling_intent_ids]} — optional;
                        built by build_sibling_map()

    Returns:
        dict {intent_id: [list of hard negative text strings]}
        Pool size per intent = top_k * n_queries + sibling injections.
        The collate_fn samples randomly from this pool each batch.
    """
    all_texts, all_labels = [], []
    for intent_id, queries in intent_groups.items():
        for q in queries:
            all_texts.append(q)
            all_labels.append(intent_id)

    labels_arr = np.array(all_labels)

    # embed all training queries — model.encode handles batching + no_grad
    embeddings = model.encode(all_texts, tokenizer, device=device)  # [N, 128]

    # cosine similarity (L2-normalised → dot product = cosine)
    sim = embeddings @ embeddings.T  # [N, N]

    hard_negs = defaultdict(list)
    for i, label in enumerate(all_labels):
        scores = sim[i].copy()
        scores[labels_arr == label] = -2.0   # mask same-intent (including self)
        top_idx = np.argsort(-scores)[:top_k]
        for idx in top_idx:
            hard_negs[label].append(all_texts[idx])

    # ── sibling injection ──────────────────────────────────────────────────────
    # For each sibling group, inject a random sample of each sibling's queries
    # into the hard-neg pool. Pool size injected = n_sibling_queries // 2 each.
    if sibling_map:
        n_injected = 0
        for intent_id, sibling_ids in sibling_map.items():
            if intent_id not in intent_groups:
                continue
            for sib_id in sibling_ids:
                if sib_id not in intent_groups:
                    continue
                sib_queries = intent_groups[sib_id]
                # inject half the sibling's queries (minimum 5)
                n_inject = max(5, len(sib_queries) // 2)
                sample   = random.sample(sib_queries, min(n_inject, len(sib_queries)))
                hard_negs[intent_id].extend(sample)
                n_injected += len(sample)
        print(f"    [Sibling HN] Injected {n_injected} sibling queries into hard-neg pools")

    return dict(hard_negs)


# ── Learning Rate Scheduler ────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup then linear decay to 10% of initial LR.

    warmup: steps 0 → warmup_steps  — lr ramps 0 → initial_lr
    decay:  steps warmup_steps → total_steps — lr ramps initial_lr → 0.1*initial_lr
    """

    def lr_lambda(current_step: int):
        # warmup phase: ramp from 0.0 → 1.0
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # decay phase: ramp from 1.0 → 0.1
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Validation Evaluation ──────────────────────────────────────────────────────

def evaluate_val(
    model,
    val_data: dict,
    tokenizer,
    loss_fn,
    device: str,
    pairs_per_intent: int = 10,
    seed: int = 42,
) -> dict:
    """
    Evaluate the model on the validation set.

    Why we need this:
        Training metrics only tell us how well the model fits training data.
        Validation metrics tell us how well it generalises to unseen data.
        If train loss drops but val loss rises → overfitting.

    How it works:
        1. Create val pairs from val_data (separate from training pairs)
        2. Run model in eval mode (dropout disabled)
        3. Compute val loss and val Recall@1 accuracy
        4. No gradient updates — purely measurement

    Returns:
        dict with val_loss and val_accuracy
    """

    # create validation pairs — different queries from the val split
    # uses a different seed from training to avoid overlap
    val_pairs, val_intent_ids = create_training_pairs(
        val_data,
        pairs_per_intent=pairs_per_intent,
        seed=seed + 1,
        verbose=False,
    )

    # build val dataloader — same structure as training
    n_intents   = len(val_data)
    val_sampler = IntentAwareBatchSampler(val_intent_ids, batch_size=n_intents)
    val_dataset = PairDataset(val_pairs, val_intent_ids)
    val_loader  = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=make_collate_fn(tokenizer),  # no hard negs during validation
    )

    # switch to eval mode — disables dropout so results are deterministic
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_pairs   = 0

    # torch.no_grad() — skip gradient computation entirely
    with torch.no_grad():
        for anchor_ids, anchor_mask, positive_ids, positive_mask, _, _ in val_loader:

            anchor_ids    = anchor_ids.to(device)
            anchor_mask   = anchor_mask.to(device)
            positive_ids  = positive_ids.to(device)
            positive_mask = positive_mask.to(device)

            # forward pass — same as training but no backward
            anchor_emb   = model(anchor_ids,   anchor_mask)
            positive_emb = model(positive_ids, positive_mask)

            # compute val loss (no hard negatives during evaluation)
            loss = loss_fn(anchor_emb, positive_emb)
            total_loss += loss.item()

            # compute val accuracy — argmax of similarity matrix diagonal
            sim       = torch.matmul(anchor_emb, positive_emb.T)
            predicted = sim.argmax(dim=1)
            labels    = torch.arange(sim.shape[0], device=device)
            total_correct += (predicted == labels).sum().item()
            total_pairs   += sim.shape[0]

    # switch back to training mode for next epoch
    model.train()

    return {
        "val_loss":     total_loss / len(val_loader),
        "val_accuracy": total_correct / total_pairs,
    }


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device: str,
    epoch: int,
    val_data: dict = None,
    tokenizer=None,
) -> dict:
    """
    Run one full pass over all training batches.
    Optionally evaluates on val set at end of epoch.

    Hard negatives are embedded inside this function when hn_ids is non-None
    in the batch (provided by make_collate_fn when hard_neg_map is active).

    Returns dict with train_loss, train_accuracy, and optionally val metrics.
    """

    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_pairs   = 0
    start_time    = time.time()

    for anchor_ids, anchor_mask, positive_ids, positive_mask, hn_ids, hn_mask in dataloader:

        # move to device
        anchor_ids    = anchor_ids.to(device)
        anchor_mask   = anchor_mask.to(device)
        positive_ids  = positive_ids.to(device)
        positive_mask = positive_mask.to(device)

        # forward pass
        anchor_emb   = model(anchor_ids,   anchor_mask)
        positive_emb = model(positive_ids, positive_mask)

        # embed hard negatives if provided — shape [B*k, D]
        hard_neg_emb = None
        if hn_ids is not None:
            hard_neg_emb = model(hn_ids.to(device), hn_mask.to(device))

        # compute loss — loss infers k from hard_neg_emb.shape when provided
        loss = loss_fn(anchor_emb, positive_emb, hard_neg_emb)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # track training metrics
        total_loss += loss.item()

        with torch.no_grad():
            sim           = torch.matmul(anchor_emb, positive_emb.T)
            predicted     = sim.argmax(dim=1)
            labels        = torch.arange(sim.shape[0], device=device)
            total_correct += (predicted == labels).sum().item()
            total_pairs   += sim.shape[0]

    elapsed        = time.time() - start_time
    train_loss     = total_loss / len(dataloader)
    train_accuracy = total_correct / total_pairs

    # build metrics dict — start with training metrics
    metrics = {
        "loss":     train_loss,
        "accuracy": train_accuracy,
    }

    # optionally evaluate on validation set
    if val_data is not None and tokenizer is not None:
        val_metrics = evaluate_val(model, val_data, tokenizer, loss_fn, device)
        metrics["val_loss"]     = val_metrics["val_loss"]
        metrics["val_accuracy"] = val_metrics["val_accuracy"]

        # overfitting indicator: gap between train and val loss
        gap = val_metrics["val_loss"] - train_loss
        overfit_flag = " ⚠ overfit?" if gap > 1.0 else ""

        print(f"  Epoch {epoch:02d} | "
              f"train_loss {train_loss:.4f} | train_acc {train_accuracy*100:.1f}% | "
              f"val_loss {val_metrics['val_loss']:.4f} | val_acc {val_metrics['val_accuracy']*100:.1f}% | "
              f"time {elapsed:.1f}s{overfit_flag}")
    else:
        print(f"  Epoch {epoch:02d} | "
              f"loss {train_loss:.4f} | acc {train_accuracy*100:.1f}% | "
              f"time {elapsed:.1f}s")

    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_history(history: list, output_dir: str, label: str = ""):
    """
    Generate and save training charts showing train vs val curves.

    Four subplots:
        Top-left:  Train loss vs Val loss     — overfitting visible as gap
        Top-right: Train acc  vs Val acc      — generalisation visible
        Bottom-left:  Loss gap (val-train)    — direct overfitting signal
        Bottom-right: Val accuracy trend      — best val accuracy highlighted

    If val metrics are not in history, falls back to train-only plots.
    """

    epochs      = list(range(1, len(history) + 1))
    train_loss  = [h["loss"]     for h in history]
    train_acc   = [h["accuracy"] * 100 for h in history]
    has_val     = "val_loss" in history[0]

    if has_val:
        val_loss = [h["val_loss"]     for h in history]
        val_acc  = [h["val_accuracy"] * 100 for h in history]

    title = f"Training History — MiniIntentEmbedder"
    if label:
        title += f" ({label})"

    if has_val:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── top-left: loss curves ──────────────────────────────────
    ax1.plot(epochs, train_loss, color="#e74c3c", linewidth=2,
             marker="o", markersize=4, label="Train loss")
    if has_val:
        ax1.plot(epochs, val_loss, color="#e67e22", linewidth=2,
                 marker="s", markersize=4, linestyle="--", label="Val loss")
        # shade gap between train and val loss to show overfitting
        ax1.fill_between(epochs, train_loss, val_loss,
                         alpha=0.15, color="#e67e22", label="Overfitting gap")
    ax1.fill_between(epochs, train_loss, alpha=0.08, color="#e74c3c")
    ax1.annotate(f"{train_loss[-1]:.4f}", xy=(epochs[-1], train_loss[-1]),
                 xytext=(-35, 8), textcoords="offset points", fontsize=9, color="#e74c3c")
    if has_val:
        ax1.annotate(f"{val_loss[-1]:.4f}", xy=(epochs[-1], val_loss[-1]),
                     xytext=(-35, -15), textcoords="offset points", fontsize=9, color="#e67e22")
    ax1.set_title("Loss: Train vs Validation", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── top-right: accuracy curves ─────────────────────────────
    ax2.plot(epochs, train_acc, color="#2ecc71", linewidth=2,
             marker="o", markersize=4, label="Train acc")
    if has_val:
        ax2.plot(epochs, val_acc, color="#27ae60", linewidth=2,
                 marker="s", markersize=4, linestyle="--", label="Val acc")
        # mark best val accuracy
        best_val_epoch = val_acc.index(max(val_acc)) + 1
        best_val_value = max(val_acc)
        ax2.axvline(x=best_val_epoch, color="gray", linestyle=":", alpha=0.7)
        ax2.annotate(f"best val\n{best_val_value:.1f}%",
                     xy=(best_val_epoch, best_val_value),
                     xytext=(8, -20), textcoords="offset points",
                     fontsize=8, color="#27ae60")
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax2.annotate(f"{train_acc[-1]:.1f}%", xy=(epochs[-1], train_acc[-1]),
                 xytext=(-35, 8), textcoords="offset points", fontsize=9, color="#2ecc71")
    ax2.set_title("Recall@1: Train vs Validation", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if has_val:
        # ── bottom-left: overfitting gap ──────────────────────
        gap = [v - t for v, t in zip(val_loss, train_loss)]
        colors = ["#e74c3c" if g > 0.5 else "#f39c12" if g > 0.2 else "#2ecc71" for g in gap]
        ax3.bar(epochs, gap, color=colors, alpha=0.7)
        ax3.axhline(y=0,   color="black", linewidth=0.8)
        ax3.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Overfit threshold")
        ax3.axhline(y=0.2, color="#f39c12", linestyle="--", alpha=0.5, label="Warning threshold")
        ax3.set_title("Overfitting Gap (val_loss − train_loss)", fontsize=12)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss Gap")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # ── bottom-right: val accuracy with trend ─────────────
        ax4.plot(epochs, val_acc, color="#3498db", linewidth=2,
                 marker="o", markersize=4, label="Val accuracy")
        ax4.fill_between(epochs, val_acc, alpha=0.1, color="#3498db")
        # highlight best epoch
        ax4.scatter([best_val_epoch], [best_val_value],
                    color="#e74c3c", s=100, zorder=5, label=f"Best: {best_val_value:.1f}% @ epoch {best_val_epoch}")
        ax4.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
        ax4.set_title("Validation Accuracy Trend", fontsize=12)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_ylim(0, 108)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # save plot
    fname     = f"training_curves_{label}.png" if label else "training_curves.png"
    plot_path = os.path.join(output_dir, fname)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot      → {plot_path}")
    return plot_path


def plot_experiment_comparison(experiments: list, output_dir: str):
    """
    Compare multiple training runs on a rich 4-panel chart.

    experiments: list of dicts:
        {"label": "temp=0.05", "history": [...], "color": "#e74c3c"}

    Four panels:
        Top-left:     Validation loss curves per experiment
        Top-right:    Validation accuracy curves per experiment
        Bottom-left:  Overfitting gap (val_loss - train_loss) per experiment
        Bottom-right: Summary bar chart — best val accuracy per experiment
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Temperature Experiment Comparison — MiniIntentEmbedder",
                 fontsize=14, fontweight="bold")
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    summary_labels = []
    summary_best_accs = []
    summary_colors = []

    for exp in experiments:
        history    = exp["history"]
        label      = exp["label"]
        color      = exp["color"]
        epochs     = list(range(1, len(history) + 1))
        has_val    = "val_loss" in history[0]

        train_loss = [h["loss"]     for h in history]

        if has_val:
            val_loss   = [h["val_loss"]     for h in history]
            val_acc    = [h["val_accuracy"] * 100 for h in history]
            gap        = [v - t for v, t in zip(val_loss, train_loss)]
            best_acc   = max(val_acc)
            best_epoch = val_acc.index(best_acc) + 1

            # panel 1: val loss
            ax1.plot(epochs, val_loss, color=color, linewidth=2,
                     marker="o", markersize=3, label=label)
            ax1.annotate(f"{val_loss[-1]:.2f}",
                         xy=(epochs[-1], val_loss[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=8, color=color)

            # panel 2: val accuracy
            ax2.plot(epochs, val_acc, color=color, linewidth=2,
                     marker="o", markersize=3,
                     label=f"{label}  (best {best_acc:.1f}% @ ep{best_epoch})")
            # mark best epoch
            ax2.scatter([best_epoch], [best_acc], color=color, s=80,
                        zorder=5, edgecolors="white", linewidth=1.5)

            # panel 3: overfitting gap
            ax3.plot(epochs, gap, color=color, linewidth=2,
                     marker="o", markersize=3, label=label)
            ax3.annotate(f"{gap[-1]:.2f}",
                         xy=(epochs[-1], gap[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=8, color=color)

            # collect for summary bar chart
            summary_labels.append(label)
            summary_best_accs.append(best_acc)
            summary_colors.append(color)

    # ── panel 1: val loss ─────────────────────────────────────
    ax1.set_title("Validation Loss per Experiment", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── panel 2: val accuracy ─────────────────────────────────
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax2.set_title("Validation Accuracy per Experiment", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── panel 3: overfitting gap ──────────────────────────────
    ax3.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Overfit threshold (0.5)")
    ax3.axhline(y=0.2, color="#f39c12", linestyle="--", alpha=0.5, label="Warning threshold (0.2)")
    ax3.axhline(y=0.0, color="black",   linewidth=0.8)
    ax3.set_title("Overfitting Gap (val_loss − train_loss)", fontsize=12)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss Gap")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── panel 4: summary bar chart ────────────────────────────
    bars = ax4.bar(summary_labels, summary_best_accs,
                   color=summary_colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    # label each bar with its value
    for bar, acc in zip(bars, summary_best_accs):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{acc:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax4.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax4.set_title("Best Validation Accuracy per Experiment", fontsize=12)
    ax4.set_ylabel("Best Val Accuracy (%)")
    ax4.set_ylim(0, 80)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "experiment_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Comparison plot → {plot_path}")
    return plot_path


def plot_hn_experiment_comparison(experiments: list, output_dir: str):
    """
    4-panel comparison chart for hard negative top_k experiments.

    experiments: list of dicts with "label", "history", "color".

    Panels:
        Top-left:     Validation loss curves per top_k config
        Top-right:    Validation accuracy curves per top_k config
        Bottom-left:  Overfitting gap (val_loss - train_loss) per config
        Bottom-right: Best val accuracy summary bar chart
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hard Negative Mining — top_k Comparison",
                 fontsize=14, fontweight="bold")
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    summary_labels    = []
    summary_best_accs = []
    summary_colors    = []

    for exp in experiments:
        history    = exp["history"]
        label      = exp["label"]
        color      = exp["color"]
        epochs     = list(range(1, len(history) + 1))
        has_val    = "val_loss" in history[0]
        train_loss = [h["loss"] for h in history]

        if has_val:
            val_loss   = [h["val_loss"]     for h in history]
            val_acc    = [h["val_accuracy"] * 100 for h in history]
            gap        = [v - t for v, t in zip(val_loss, train_loss)]
            best_acc   = max(val_acc)
            best_epoch = val_acc.index(best_acc) + 1

            ax1.plot(epochs, val_loss, color=color, linewidth=2,
                     marker="o", markersize=3, label=label)
            ax1.annotate(f"{val_loss[-1]:.2f}", xy=(epochs[-1], val_loss[-1]),
                         xytext=(4, 0), textcoords="offset points", fontsize=8, color=color)

            ax2.plot(epochs, val_acc, color=color, linewidth=2,
                     marker="o", markersize=3,
                     label=f"{label}  (best {best_acc:.1f}% @ ep{best_epoch})")
            ax2.scatter([best_epoch], [best_acc], color=color, s=80,
                        zorder=5, edgecolors="white", linewidth=1.5)

            ax3.plot(epochs, gap, color=color, linewidth=2,
                     marker="o", markersize=3, label=label)

            summary_labels.append(label)
            summary_best_accs.append(best_acc)
            summary_colors.append(color)

    ax1.set_title("Validation Loss per Config", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax2.set_title("Validation Accuracy per Config", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Overfit threshold")
    ax3.axhline(y=0.0, color="black", linewidth=0.8)
    ax3.set_title("Overfitting Gap (val_loss − train_loss)", fontsize=12)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss Gap")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    if summary_best_accs:
        bars = ax4.bar(summary_labels, summary_best_accs,
                       color=summary_colors, alpha=0.8,
                       edgecolor="white", linewidth=1.5)
        for bar, acc in zip(bars, summary_best_accs):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{acc:.1f}%",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax4.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax4.set_title("Best Validation Accuracy per Config", fontsize=12)
    ax4.set_ylabel("Best Val Accuracy (%)")
    ax4.set_ylim(0, (max(summary_best_accs) * 1.25) if summary_best_accs else 100)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "hn_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  HN comparison plot → {plot_path}")
    return plot_path


# ── Main Train Function ────────────────────────────────────────────────────────

def train(args) -> tuple:
    """
    Full training pipeline with validation tracking and optional hard negative mining.

    Steps:
        1. Load dataset (train + val + test)
        2. Build tokenizer on train data
        3. Build model
        4. Create training pairs + PairDataset
        5. Optimizer + scheduler + loss
        6. Training loop — each epoch:
               a. Optionally mine hard negatives (offline, every N epochs)
               b. Build DataLoader with current hard_neg_map
               c. train_one_epoch → evaluate on val set
        7. Save model, tokenizer, config, plots
    """

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    train_data, val_data, test_data = load_dataset()

    # auto-set batch size to n_intents for zero false negatives
    n_intents = len(train_data)
    if args.batch_size != n_intents:
        print(f"\n  Adjusting batch size: {args.batch_size} → {n_intents} (= n_intents)")
        args.batch_size = n_intents

    # ── Step 2: Build tokenizer ────────────────────────────────
    print("\nBuilding tokenizer...")
    all_texts = [text for queries in train_data.values() for text in queries]
    tokenizer = SimpleTokenizer(max_vocab_size=8000, max_seq_len=32)
    tokenizer.fit(all_texts)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Step 3: Build model ────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(tokenizer)
    model = model.to(device)
    print(f"  Parameters: {model.n_parameters():,}")

    # ── Step 4: Pairing mode ───────────────────────────────────
    # Dynamic (default): pairs re-created every epoch with a new seed →
    #     the model sees different anchor/positive combinations each epoch.
    # Static (--static-pairing): same pairs every epoch → faster to overfit.
    static_pairing = getattr(args, 'static_pairing', False)
    pairing_label  = "static — same pairs every epoch" if static_pairing \
                     else "dynamic — re-shuffled each epoch"
    batches_per_epoch = (args.pairs_per_intent * n_intents) // args.batch_size
    print(f"\n  Pairs/epoch   : {args.pairs_per_intent * n_intents:,}  ({pairing_label})")
    print(f"  Batches/epoch : {batches_per_epoch}")

    # ── Step 5: Optimizer + scheduler + loss ───────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = batches_per_epoch * args.epochs
    scheduler   = make_scheduler(optimizer, args.warmup_steps, total_steps)
    loss_fn     = MultipleNegativesRankingLoss(temperature=args.temperature)

    # ── Step 6: Training loop ──────────────────────────────────
    use_hn      = getattr(args, 'hard_negatives', False)
    hn_top_k    = getattr(args, 'hn_top_k', 1)
    hn_start    = getattr(args, 'hn_start_epoch', 4)
    hn_refresh  = getattr(args, 'hn_refresh_every', 3)
    use_sibling = getattr(args, 'sibling_hn', False)

    hard_neg_map = None   # starts None — enabled after warm-up

    # build sibling map once (only needs intent names → ids, no embeddings)
    sibling_map = None
    if use_hn and use_sibling:
        from datasets import load_dataset as hf_load
        ds = hf_load("clinc/clinc_oos", "small")
        label_names    = ds["train"].features["intent"].names
        name_to_id     = {n: i for i, n in enumerate(label_names)}
        sibling_map    = build_sibling_map(train_data, name_to_id)

    print(f"\nTraining for {args.epochs} epochs...")
    if use_hn:
        print(f"  Hard negatives: top_k={hn_top_k}, "
              f"start_epoch={hn_start}, refresh_every={hn_refresh}"
              + (f", sibling_hn=True" if use_sibling else ""))
    print(f"  {'Epoch':>5} | {'train_loss':>10} | {'train_acc':>9} | "
          f"{'val_loss':>8} | {'val_acc':>7} | {'time':>6}")
    print("  " + "─" * 65)

    history = []

    for epoch in range(1, args.epochs + 1):

        # pairing mode: static keeps seed fixed every epoch (same pairs each time)
        #               dynamic uses seed+epoch (different pairs each epoch)
        pair_seed = args.seed if static_pairing else args.seed + epoch
        pairs, intent_ids = create_training_pairs(
            train_data,
            pairs_per_intent=args.pairs_per_intent,
            seed=pair_seed,
            verbose=(epoch == 1),   # print once; suppress for remaining epochs
        )
        dataset = PairDataset(pairs, intent_ids)

        # mine hard negatives at configured intervals (after warm-up)
        if use_hn and epoch >= hn_start and (epoch - hn_start) % hn_refresh == 0:
            print(f"  [HN] Mining hard negatives (top_k={hn_top_k}"
                  + (", +siblings" if use_sibling else "") + ")...")
            hard_neg_map = mine_hard_negatives(
                model, tokenizer, train_data, top_k=hn_top_k, device=device,
                sibling_map=sibling_map,
            )

        # rebuild DataLoader each epoch — collate_fn may carry updated hard_neg_map
        sampler    = IntentAwareBatchSampler(intent_ids, batch_size=args.batch_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=make_collate_fn(tokenizer, hard_neg_map, hn_top_k),
        )

        metrics = train_one_epoch(
            model, dataloader, optimizer, scheduler,
            loss_fn, device, epoch,
            val_data=val_data,
            tokenizer=tokenizer,
        )
        history.append(metrics)

    # print overfitting summary
    if "val_loss" in history[-1]:
        final_gap = history[-1]["val_loss"] - history[-1]["loss"]
        best_val  = max(h["val_accuracy"] for h in history) * 100
        best_epoch = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) + 1
        print(f"\n  Best val accuracy : {best_val:.1f}% at epoch {best_epoch}")
        print(f"  Final loss gap    : {final_gap:.4f} "
              f"({'⚠ overfitting' if final_gap > 0.5 else 'healthy'})")

    # ── Step 7: Save artifacts ─────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    model_path  = os.path.join(args.output_dir, "model.pt")
    tok_path    = os.path.join(args.output_dir, "tokenizer.json")
    config_path = os.path.join(args.output_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    tokenizer.save(tok_path)

    with open(config_path, "w") as f:
        json.dump({
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "temperature":      args.temperature,
            "pairs_per_intent": args.pairs_per_intent,
            "vocab_size":       tokenizer.vocab_size,
            "hard_negatives":   use_hn,
            "hn_top_k":         hn_top_k if use_hn else 0,
            "history":          history,
        }, f, indent=2)

    # generate plots — use plot_dir if set, otherwise same as output_dir
    label    = getattr(args, "experiment_label", "")
    plot_dir = getattr(args, "plot_dir", None) or args.output_dir
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_history(history, plot_dir, label=label)

    print(f"\n  Model     → {model_path}")
    print(f"  Tokenizer → {tok_path}")
    print(f"  Config    → {config_path}")

    return model, tokenizer, args.output_dir


# ── Temperature Experiment Runner ──────────────────────────────────────────────

def run_experiments(base_args, results_dir: str):
    """
    Run multiple training experiments with different configs and compare them.

    Experiments:
        1. temperature=0.01  — very sharp signal, very hard negatives
        2. temperature=0.05  — default, sharp signal
        3. temperature=0.20  — soft signal, easier negatives

    Why temperature is interesting:
        - Low temp  → loss is very sensitive to any confusion → fast but risky
        - High temp → loss is forgiving → slower but more stable
        - Finding the sweet spot matters for generalisation

    Saves individual plots per experiment + one comparison plot.
    """

    experiments_config = [
        {"label": "temp=0.01", "temperature": 0.01, "color": "#e74c3c"},
        {"label": "temp=0.05", "temperature": 0.05, "color": "#3498db"},
        {"label": "temp=0.20", "temperature": 0.20, "color": "#2ecc71"},
    ]

    os.makedirs(results_dir, exist_ok=True)
    all_experiments = []

    for config in experiments_config:
        print(f"\n{'='*65}")
        print(f"  EXPERIMENT: {config['label']}")
        print(f"{'='*65}")

        exp_args = copy.deepcopy(base_args)
        exp_args.temperature      = config["temperature"]
        exp_args.experiment_label = config["label"]

        safe_label            = config["label"].replace("=", "_")
        models_subdir         = os.path.join(
            os.path.dirname(results_dir), "models", "experiments", safe_label
        )
        exp_args.output_dir   = models_subdir
        exp_args.plot_dir     = os.path.join(results_dir, "experiments", safe_label)

        _, _, _ = train(exp_args)

        config_path = os.path.join(exp_args.output_dir, "config.json")
        with open(config_path) as f:
            saved = json.load(f)

        all_experiments.append({
            "label":   config["label"],
            "history": saved["history"],
            "color":   config["color"],
        })

    # generate comparison plot — saved at top of results_dir
    plots_root = os.path.join(results_dir, "experiments")
    os.makedirs(plots_root, exist_ok=True)
    plot_experiment_comparison(all_experiments, plots_root)

    # print summary table
    print(f"\n{'='*65}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Experiment':<15} | {'Best Val Acc':>12} | {'Best Epoch':>10} | {'Final Val Loss':>14}")
    print("  " + "─" * 58)

    for exp in all_experiments:
        history   = exp["history"]
        has_val   = "val_loss" in history[0]
        if has_val:
            best_acc   = max(h["val_accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) + 1
            final_loss = history[-1]["val_loss"]
        else:
            best_acc   = max(h["accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["accuracy"]) + 1
            final_loss = history[-1]["loss"]
        print(f"  {exp['label']:<15} | {best_acc:>11.1f}% | {best_epoch:>10} | {final_loss:>14.4f}")

    # save summary to file
    summary_path = os.path.join(plots_root, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n  Summary → {summary_path}")
    return all_experiments


# ── Hard Negative Experiment Runner ───────────────────────────────────────────

def run_hn_experiments(base_args, results_dir: str, version: str = "v1"):
    """
    Compare training with different hard negative top_k values:
        top_k=0 — no hard negatives (identical to standard training)
        top_k=1 — 1 hard negative per anchor added to the loss each step
        top_k=3 — 3 hard negatives per anchor added to the loss each step
        top_k=5 — 5 hard negatives per anchor added to the loss each step

    With top_k > 0, the loss similarity matrix grows from [B, B] to [B, B+k].
    Each anchor must rank its k hard negatives lower than its true positive.
    Hard negatives are re-mined every hn_refresh_every epochs using current
    model weights, so the difficulty adapts as training progresses.

    Versioned output layout (example with version="v1"):
        models/hn_experiments/v1/no_hn/     model.pt + tokenizer.json + config.json
        models/hn_experiments/v1/hn_k_1/
        models/hn_experiments/v1/hn_k_3/
        models/hn_experiments/v1/hn_k_5/
        results/hn_experiments/v1/no_hn/    training_curves_*.png
        results/hn_experiments/v1/hn_k_1/
        results/hn_experiments/v1/hn_comparison.png
        results/hn_experiments/v1/hn_summary.json

    Args:
        base_args  : parsed argparse namespace (copied per experiment)
        results_dir: base results directory (e.g. "../results")
        version    : version tag — organises runs so old results are never overwritten
                     Use "v0", "v1", "v2", ... to track experiment generations
    """

    hn_configs = [
        {"label": "no_hn",   "top_k": 0, "color": "#95a5a6"},
        {"label": "hn_k=1",  "top_k": 1, "color": "#e74c3c"},
        {"label": "hn_k=3",  "top_k": 3, "color": "#3498db"},
        {"label": "hn_k=5",  "top_k": 5, "color": "#2ecc71"},
    ]

    # pairing mode label — embedded in paths so static and dynamic never collide
    static_pairing = getattr(base_args, 'static_pairing', False)
    pairing_tag    = "static" if static_pairing else "dynamic"

    # versioned roots — all output lives under version/pairing_tag/
    repo_root    = os.path.dirname(results_dir)
    models_root  = os.path.join(repo_root,   "models",  "hn_experiments", version, pairing_tag)
    plots_root   = os.path.join(results_dir, "hn_experiments", version, pairing_tag)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(plots_root,  exist_ok=True)

    print(f"\n  Version    : {version}")
    print(f"  Pairing    : {pairing_tag}")
    print(f"  Models  →  {models_root}")
    print(f"  Results →  {plots_root}")

    all_experiments = []

    for cfg in hn_configs:
        print(f"\n{'='*65}")
        print(f"  HN EXPERIMENT: {cfg['label']}  (top_k={cfg['top_k']})")
        print(f"{'='*65}")

        exp_args = copy.deepcopy(base_args)
        exp_args.hard_negatives   = cfg["top_k"] > 0
        exp_args.hn_top_k         = cfg["top_k"]
        exp_args.experiment_label = cfg["label"]

        safe_label          = cfg["label"].replace("=", "_")
        # model artifacts → models/hn_experiments/<version>/<label>/
        exp_args.output_dir = os.path.join(models_root, safe_label)
        # training curve plots → results/hn_experiments/<version>/<label>/
        exp_args.plot_dir   = os.path.join(plots_root,  safe_label)

        train(exp_args)

        config_path = os.path.join(exp_args.output_dir, "config.json")
        with open(config_path) as f:
            saved = json.load(f)

        all_experiments.append({
            "label":        f"{cfg['label']} ({pairing_tag})",
            "history":      saved["history"],
            "color":        cfg["color"],
            "version":      version,
            "pairing":      pairing_tag,
            "hn_top_k":     cfg["top_k"],
        })

    # comparison plot → results/hn_experiments/<version>/hn_comparison.png
    plot_hn_experiment_comparison(all_experiments, plots_root)

    # print summary table
    print(f"\n{'='*65}")
    print(f"  HN EXPERIMENT SUMMARY  [{version}]")
    print(f"{'='*65}")
    print(f"  {'Config':<12} | {'Best Val Acc':>12} | {'Best Epoch':>10} | {'Final Val Loss':>14}")
    print("  " + "─" * 55)

    for exp in all_experiments:
        history = exp["history"]
        has_val = "val_loss" in history[0]
        if has_val:
            best_acc   = max(h["val_accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) + 1
            final_loss = history[-1]["val_loss"]
        else:
            best_acc   = max(h["accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["accuracy"]) + 1
            final_loss = history[-1]["loss"]
        print(f"  {exp['label']:<12} | {best_acc:>11.1f}% | {best_epoch:>10} | {final_loss:>14.4f}")

    # summary JSON → results/hn_experiments/<version>/hn_summary.json
    summary_path = os.path.join(plots_root, "hn_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n  Summary → {summary_path}")
    return all_experiments


# ── Stability Experiment Runner ────────────────────────────────────────────────

def run_stability_experiments(base_args, results_dir: str, n_seeds: int = 5,
                              explicit_seeds: list = None):
    """
    Assess training stability by running the best config N times from scratch,
    each with a different random seed.  Measures how much variance exists in
    val accuracy, losses, and the train/val gap across independent runs.

    Best config (hardcoded):
        dynamic pairing  +  hard_negatives=True  +  hn_top_k=5
        (same temperature / lr / epochs / pairs_per_intent as base_args)

    Seeds used:
        If explicit_seeds is provided → use those directly
        Otherwise → base_args.seed, seed+1, ..., seed+(n_seeds-1)

    Output layout:
        models/stability/seed_N/     model.pt + tokenizer.json + config.json
        results/stability/seed_N/    training_curves_*.png
        results/stability/stability_summary.json
        results/stability/stability_plot.png

    Args:
        base_args      : parsed argparse namespace (copied per run)
        results_dir    : base results directory (e.g. "../results")
        n_seeds        : number of runs when explicit_seeds is not provided
        explicit_seeds : list of specific seed values to use (overrides n_seeds)
    """
    repo_root   = os.path.dirname(results_dir)
    models_root = os.path.join(repo_root,   "models",  "stability")
    plots_root  = os.path.join(results_dir, "stability")
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(plots_root,  exist_ok=True)

    if explicit_seeds:
        seeds = explicit_seeds
    else:
        seeds = [base_args.seed + i for i in range(n_seeds)]

    print(f"\n{'='*65}")
    print(f"  STABILITY EXPERIMENT — {n_seeds} independent runs")
    print(f"  Config : dynamic pairing + hard_negatives + hn_top_k=5")
    print(f"  Seeds  : {seeds}")
    print(f"  Models → {models_root}")
    print(f"  Results→ {plots_root}")
    print(f"{'='*65}")

    all_runs = []

    for seed in seeds:
        print(f"\n{'─'*65}")
        print(f"  RUN  seed={seed}  ({seeds.index(seed)+1}/{n_seeds})")
        print(f"{'─'*65}")

        run_args = copy.deepcopy(base_args)
        run_args.seed             = seed
        run_args.hard_negatives   = True
        run_args.hn_top_k         = 5
        run_args.static_pairing   = False          # dynamic pairing
        run_args.experiment_label = f"seed_{seed}"

        run_args.output_dir = os.path.join(models_root, f"seed_{seed}")
        run_args.plot_dir   = os.path.join(plots_root,  f"seed_{seed}")

        train(run_args)

        config_path = os.path.join(run_args.output_dir, "config.json")
        with open(config_path) as f:
            saved = json.load(f)

        history = saved["history"]
        has_val = "val_loss" in history[0]

        if has_val:
            best_idx   = max(range(len(history)), key=lambda i: history[i]["val_accuracy"])
            best_epoch = best_idx + 1
            best_h     = history[best_idx]
            best_val_acc  = best_h["val_accuracy"] * 100
            best_val_loss = best_h["val_loss"]
            train_loss    = best_h["loss"]
            train_acc     = best_h["accuracy"] * 100
            gap           = best_val_loss - train_loss
        else:
            best_idx      = max(range(len(history)), key=lambda i: history[i]["accuracy"])
            best_epoch    = best_idx + 1
            best_h        = history[best_idx]
            best_val_acc  = best_h["accuracy"] * 100
            best_val_loss = best_h["loss"]
            train_loss    = best_h["loss"]
            train_acc     = best_h["accuracy"] * 100
            gap           = 0.0

        all_runs.append({
            "seed":        seed,
            "best_epoch":  best_epoch,
            "val_acc":     best_val_acc,
            "train_loss":  train_loss,
            "train_acc":   train_acc,
            "val_loss":    best_val_loss,
            "gap":         gap,
            "history":     history,
        })

        print(f"  → seed={seed}  val_acc={best_val_acc:.1f}%  gap={gap:.3f}")

    # ── summary stats ──────────────────────────────────────────────────────────
    metrics = ["val_acc", "train_loss", "train_acc", "val_loss", "gap"]
    stats   = {}
    for m in metrics:
        vals        = [r[m] for r in all_runs]
        stats[m]    = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "min":  float(np.min(vals)),  "max": float(np.max(vals))}

    print(f"\n{'='*65}")
    print(f"  STABILITY SUMMARY  (n={n_seeds} seeds)")
    print(f"{'='*65}")
    print(f"  {'Seed':<8} | {'Val Acc':>8} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>9} | {'Gap':>7}")
    print("  " + "─" * 61)
    for r in all_runs:
        print(f"  {r['seed']:<8} | {r['val_acc']:>7.1f}% | {r['train_loss']:>10.3f} | "
              f"{r['train_acc']:>9.1f}% | {r['val_loss']:>9.3f} | {r['gap']:>7.3f}")
    print("  " + "─" * 61)
    print(f"  {'mean':<8} | {stats['val_acc']['mean']:>7.1f}% | {stats['train_loss']['mean']:>10.3f} | "
          f"{stats['train_acc']['mean']:>9.1f}% | {stats['val_loss']['mean']:>9.3f} | {stats['gap']['mean']:>7.3f}")
    print(f"  {'± std':<8} | {stats['val_acc']['std']:>7.2f}% | {stats['train_loss']['std']:>10.3f} | "
          f"{stats['train_acc']['std']:>9.2f}% | {stats['val_loss']['std']:>9.3f} | {stats['gap']['std']:>7.3f}")
    print(f"  {'range':<8} | {stats['val_acc']['min']:>6.1f}–{stats['val_acc']['max']:.1f}%")

    # ── save summary JSON ──────────────────────────────────────────────────────
    summary = {
        "config": {
            "dynamic_pairing": True,
            "hard_negatives":  True,
            "hn_top_k":        5,
            "epochs":          base_args.epochs,
            "temperature":     base_args.temperature,
            "pairs_per_intent": base_args.pairs_per_intent,
            "lr":              base_args.lr,
            "n_seeds":         n_seeds,
            "seeds":           seeds,
        },
        "runs":  all_runs,
        "stats": stats,
    }
    summary_path = os.path.join(plots_root, "stability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {summary_path}")

    # ── stability plot ─────────────────────────────────────────────────────────
    _plot_stability(all_runs, stats, plots_root)

    return summary


def _plot_stability(all_runs, stats, plots_root):
    """
    Two-panel stability plot:
      Left  — Val accuracy per seed (bar) + mean ± std band
      Right — Val accuracy learning curves overlaid for all seeds
    """
    seeds     = [r["seed"] for r in all_runs]
    val_accs  = [r["val_acc"] for r in all_runs]
    mean_acc  = stats["val_acc"]["mean"]
    std_acc   = stats["val_acc"]["std"]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: bar chart of val acc per seed ───────────────────────────────────
    ax = axes[0]
    bars = ax.bar([str(s) for s in seeds], val_accs,
                  color=[colors[i % len(colors)] for i in range(len(seeds))],
                  alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.axhline(mean_acc, color="black", linewidth=1.5, linestyle="--", label=f"mean={mean_acc:.1f}%")
    ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, alpha=0.12, color="black", label=f"±1 std ({std_acc:.2f}%)")
    for bar, val in zip(bars, val_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
    ax.set_title("Val Accuracy per Seed", fontsize=12, fontweight="bold")
    ax.set_ylim(max(0, mean_acc - 5), min(100, mean_acc + 5))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: learning curves overlaid ───────────────────────────────────────
    ax2 = axes[1]
    for i, run in enumerate(all_runs):
        history = run["history"]
        has_val = "val_accuracy" in history[0]
        epochs  = list(range(1, len(history) + 1))
        if has_val:
            vals = [h["val_accuracy"] * 100 for h in history]
        else:
            vals = [h["accuracy"] * 100 for h in history]
        ax2.plot(epochs, vals, color=colors[i % len(colors)], alpha=0.8,
                 linewidth=1.8, label=f"seed {run['seed']}")

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Val Accuracy (%)", fontsize=11)
    ax2.set_title("Val Accuracy Curves — All Seeds", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Stability Analysis — Dynamic Pairing + HN k=5  (n={len(all_runs)} seeds)\n"
        f"Mean={mean_acc:.1f}%  ±{std_acc:.2f}%  Range=[{stats['val_acc']['min']:.1f}–{stats['val_acc']['max']:.1f}%]",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_path = os.path.join(plots_root, "stability_plot.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot    → {out_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniIntentEmbedder")

    # ── core training args ─────────────────────────────────────
    parser.add_argument("--epochs",           type=int,   default=15)
    parser.add_argument("--batch-size",       type=int,   default=20,
                        help="Auto-overridden to n_intents after data load")
    parser.add_argument("--pairs-per-intent", type=int,   default=25,
                        help="Pairs sampled per intent per epoch (max = n_queries // 2)")
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--temperature",      type=float, default=0.05)
    parser.add_argument("--warmup-steps",     type=int,   default=50)
    parser.add_argument("--output-dir",       type=str,   default="../models/finetuned")
    parser.add_argument("--results-dir",      type=str,   default="../results")
    parser.add_argument("--seed",             type=int,   default=42)

    # ── experiment runners ─────────────────────────────────────
    parser.add_argument("--experiments",    action="store_true",
                        help="Run temperature comparison experiments")
    parser.add_argument("--hn-experiments", action="store_true",
                        help="Run top_k hard negative comparison (0, 1, 3, 5)")
    parser.add_argument("--exp-version",    type=str, default="v1",
                        help="Version tag for experiment output dirs (default: v1). "
                             "Models → models/hn_experiments/<version>/<label>  "
                             "Plots  → results/hn_experiments/<version>/<label>")
    parser.add_argument("--static-pairing", action="store_true",
                        help="Use the same pairs every epoch instead of re-shuffling. "
                             "Default: dynamic (re-shuffle each epoch).")
    parser.add_argument("--stability",     action="store_true",
                        help="Run stability sweep: best config (dynamic+hn_k=5) "
                             "repeated --n-seeds times with different seeds.")
    parser.add_argument("--n-seeds",       type=int, default=5,
                        help="Number of seeds to run in stability sweep (default: 5)")
    parser.add_argument("--seeds",         type=int, nargs="+", default=None,
                        help="Explicit seed list for stability sweep, e.g. --seeds 7 23 99 137 256. "
                             "Overrides --seed and --n-seeds.")
    parser.add_argument("--sibling-hn",    action="store_true",
                        help="Inject known sibling-intent queries into hard-neg pools "
                             "during mining, in addition to similarity-mined negatives. "
                             "Requires --hard-negatives. See CLINC_SIBLING_GROUPS.")

    # ── hard negative mining args ──────────────────────────────
    parser.add_argument("--hard-negatives",   action="store_true",
                        help="Enable offline hard negative mining during training")
    parser.add_argument("--hn-top-k",         type=int, default=1,
                        help="Hard negatives per anchor added to loss (default: 1)")
    parser.add_argument("--hn-start-epoch",   type=int, default=4,
                        help="Epoch to start mining hard negatives (default: 4)")
    parser.add_argument("--hn-refresh-every", type=int, default=3,
                        help="Re-mine hard negatives every N epochs (default: 3)")

    args = parser.parse_args()

    if args.stability:
        # stability sweep: best config (dynamic + hn_k=5) × n_seeds
        run_stability_experiments(args, args.results_dir, n_seeds=args.n_seeds,
                                  explicit_seeds=args.seeds)
    elif args.hn_experiments:
        # compare top_k = 0, 1, 3, 5 hard negatives
        run_hn_experiments(args, args.results_dir, version=args.exp_version)
    elif args.experiments:
        # run temperature comparison experiments
        run_experiments(args, args.results_dir)
    else:
        # single training run
        train(args)
