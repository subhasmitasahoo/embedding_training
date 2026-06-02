import json
import os
import argparse
from datasets import load_dataset


# ── Intent name lookup ─────────────────────────────────────────────────────────

def load_intent_names() -> dict:
    """
    Load the intent ID → intent name mapping from the CLINC dataset.

    Returns:
        dict of {intent_id (int): intent_name (str)}
        e.g. {8: "balance", 44: "travel_alert", ...}
    """
    print("Loading intent names from CLINC dataset...")
    ds = load_dataset("clinc/clinc_oos", "small")

    # ds["train"].features["intent"] is a ClassLabel object
    # .names is a list where index = intent_id, value = intent_name
    label_names = ds["train"].features["intent"].names

    # build dict for easy lookup
    id_to_name = {i: name for i, name in enumerate(label_names)}
    print(f"  Loaded {len(id_to_name)} intent names")
    return id_to_name


# ── Analyse eval results ───────────────────────────────────────────────────────

def analyze_eval_results(eval_results_path: str, id_to_name: dict):
    """
    Load eval_results.json and print a full named breakdown of per-intent
    metrics — sorted worst to best, with intent names resolved.

    Helps answer:
        - Which intents are failing and what are they called?
        - Which intents are succeeding?
        - Are there patterns in which kinds of intents are hard?

    Args:
        eval_results_path : path to eval_results.json from evaluate.py
        id_to_name        : dict of {intent_id: intent_name}
    """

    with open(eval_results_path) as f:
        results = json.load(f)

    ft_per_intent = results["Fine-tuned"]["per_intent"]

    # json keys are strings — convert back to int for name lookup
    per_intent = {int(k): v for k, v in ft_per_intent.items()}

    # sort by Recall@1 ascending (worst first)
    sorted_intents = sorted(per_intent.items(), key=lambda x: x[1]["recall_at_1"])

    # ── Full ranked table ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  FULL PER-INTENT RANKING — Fine-tuned model (worst → best)")
    print("=" * 75)
    print(f"  {'ID':>4}  {'Intent Name':<30} | {'R@1':>7} | {'R@5':>7} | {'MRR':>7}")
    print("  " + "─" * 62)

    for intent_id, m in sorted_intents:
        name = id_to_name.get(intent_id, f"unknown_{intent_id}")
        r1   = m["recall_at_1"] * 100
        r5   = m["recall_at_5"] * 100
        mrr  = m["mrr"]

        # flag pattern A (completely lost) vs pattern B (right area, wrong top-1)
        if r1 == 0 and r5 == 0:
            tag = "  ✗ lost"
        elif r1 == 0 and r5 > 0:
            tag = "  △ confused"
        elif r1 == 100:
            tag = "  ✓ perfect"
        else:
            tag = ""

        print(f"  {intent_id:>4}  {name:<30} | {r1:>6.1f}% | {r5:>6.1f}% | {mrr:>7.4f}{tag}")

    # ── Summary by pattern ─────────────────────────────────────────
    lost      = [(i, v) for i, v in sorted_intents if v["recall_at_1"] == 0 and v["recall_at_5"] == 0]
    confused  = [(i, v) for i, v in sorted_intents if v["recall_at_1"] == 0 and v["recall_at_5"] > 0]
    partial   = [(i, v) for i, v in sorted_intents if 0 < v["recall_at_1"] < 1.0]
    perfect   = [(i, v) for i, v in sorted_intents if v["recall_at_1"] == 1.0]

    print("\n" + "=" * 75)
    print("  SUMMARY BY PATTERN")
    print("=" * 75)
    print(f"  ✗ Completely lost  (R@1=0%, R@5=0%)  : {len(lost):>3} intents")
    for i, _ in lost:
        print(f"      {i:>4}  {id_to_name.get(i, '?')}")

    print(f"\n  △ Confused         (R@1=0%, R@5>0%)  : {len(confused):>3} intents")
    for i, v in confused:
        print(f"      {i:>4}  {id_to_name.get(i, '?'):<30}  R@5={v['recall_at_5']*100:.1f}%")

    print(f"\n  ~ Partial          (0%<R@1<100%)      : {len(partial):>3} intents")
    for i, v in partial:
        print(f"      {i:>4}  {id_to_name.get(i, '?'):<30}  R@1={v['recall_at_1']*100:.1f}%")

    print(f"\n  ✓ Perfect          (R@1=100%)         : {len(perfect):>3} intents")
    for i, _ in perfect:
        print(f"      {i:>4}  {id_to_name.get(i, '?')}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse per-intent eval results")
    parser.add_argument("--eval-results", type=str,
                        default="../results/eval_results.json",
                        help="Path to eval_results.json from evaluate.py")
    args = parser.parse_args()

    id_to_name = load_intent_names()
    analyze_eval_results(args.eval_results, id_to_name)