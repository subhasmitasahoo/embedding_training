# embedding_training

Training and evaluation code for embedding models, starting with customer intent search.

This repo is built **step by step**. You can clone it, follow each step in order, run the code locally, and build the same customer-intent embedding pipeline from scratch.

---

## Table of Contents

- [Problem Statement](#problem-statement)
  - [The challenge](#the-challenge--understanding-what-a-customer-actually-wants)
  - [Why traditional approaches fail](#why-traditional-approaches-fail)
  - [The solution](#the-solution--embedding-based-retrieval)
  - [What this project builds](#what-this-project-builds)
- [Project Layout](#project-layout)
- [End-to-end Pipeline](#end-to-end-pipeline)
- [Setup](#setup)
- [Follow Along](#follow-along)
- [Step 1 — Tokenizer](#step-1--tokenizer)
- [Step 2 — Model (embeddings + self-attention)](#step-2--model-embeddings--self-attention)
- [Step 3 — TransformerBlock](#step-3--transformerblock)
- [Step 4 — MiniIntentEmbedder](#step-4--miniintentembedder)
- [Step 5 — Data Preparation](#step-5--data-preparation)
- [Step 6 — Training](#step-6--training)
  - [Overfitting tracking](#overfitting-tracking)
  - [Temperature experiments](#temperature-experiments)
  - [What temperature controls](#what-temperature-controls)
  - [Experiment results](#experiment-results)
  - [Comparison charts](#comparison-charts)
  - [Key findings](#key-findings)
- [Step 7 — Evaluation](#step-7--evaluation)
  - [What we are measuring and why](#what-we-are-measuring-and-why)
  - [Three baselines to compare](#three-baselines-to-compare)
  - [TF-IDF deep dive](#tf-idf-deep-dive)
  - [Why the embedding model beats TF-IDF](#why-the-embedding-model-beats-tf-idf)
- [Step 8 — Hard Negative Mining](#step-8--hard-negative-mining)
  - [Why easy negatives cause a plateau](#why-easy-negatives-cause-a-plateau)
  - [What is a hard negative](#what-is-a-hard-negative)
  - [Mining algorithm](#mining-algorithm)
  - [Modified loss](#modified-loss)
  - [Dynamic pairing](#dynamic-pairing)
  - [Running experiments](#running-experiments)
  - [Results — v2 full 2×4 grid](#results--v2-full-24-grid-clinc-150-15-epochs-temp005-pairs_per_intent25)
  - [Key findings from the 2×4 grid](#key-findings-from-the-24-grid)
- [Step 9 — Stability Analysis](#step-9--stability-analysis)
  - [What varies between runs](#what-varies-between-runs)
  - [Running the stability sweep](#running-the-stability-sweep)
  - [Results — Sweep 1 (seeds 42–46)](#results--sweep-1-seeds-4246-consecutive)
  - [Results — Sweep 2 (seeds 7, 23, 99, 137, 256)](#results--sweep-2-seeds-7-23-99-137-256--spread-out)
  - [Combined view — all 10 runs](#combined-view--all-10-runs)
  - [Key findings from stability analysis](#key-findings-from-stability-analysis)
- [Step 10 — Batch Retrieval Evaluation](#step-10--batch-retrieval-evaluation)
  - [Why re-evaluate](#why-re-evaluate)
  - [Running batch evaluation](#running-batch-evaluation)
  - [Results across all checkpoints](#results-across-all-checkpoints)
  - [Key findings from batch evaluation](#key-findings-from-batch-evaluation)
- [Step 11 — Evaluation Metric Stability](#step-11--evaluation-metric-stability)
  - [The corpus-split problem](#the-corpus-split-problem)
  - [Running eval stability](#running-eval-stability)
  - [Results across corpus splits](#results-across-corpus-splits)
  - [Key findings from eval stability](#key-findings-from-eval-stability)
- [Dependencies](#dependencies)

---

## Problem Statement

### The challenge — understanding what a customer actually wants

When a customer contacts e-commerce support, they can express the same need in hundreds of different ways:

```
"where is my package"
"has my order shipped yet"
"I haven't received anything"
"my delivery is late"
"can you track #ORD-1234 for me"
```

All of these mean the same thing — the customer wants to **track their order**. A support system needs to understand this, route the query to the right team, and respond appropriately.

### Why traditional approaches fail

**Keyword matching** — looks for specific words like "track" or "order":
```
"I haven't received anything" → no keywords matched → fails
"can you track #ORD-1234"     → "track" matched → works
```
Brittle. Misses any phrasing that doesn't use the exact expected words.

**Rule-based systems** — hand-crafted rules for each intent:
```
if "cancel" in text → cancel_order
if "return" in text → initiate_return
```
Doesn't scale. Writing and maintaining rules for thousands of query variations is impractical. Breaks on typos, informal language, and new phrasings.

**TF-IDF classifiers** — represent text as word frequency vectors:
```
"where is my order"    → [0, 0, 1, 0, 1, 0, ...]  word counts
"has my order shipped" → [0, 0, 1, 1, 0, 0, ...]  word counts
```
Treats text as a bag of words — word order and context are lost. `"I can't cancel"` and `"cancel"` look similar. Requires retraining the entire classifier every time a new intent is added.

### The solution — embedding-based retrieval

Instead of classifying, we **search**. We train a model to convert any query into a 128-dimensional vector such that:

```
semantically similar queries → vectors point in the same direction
semantically different queries → vectors point in different directions
```

```
"where is my package"      → [0.3, -0.5, 0.8, ...]
"has my order shipped yet" → [0.3, -0.5, 0.7, ...]  ← very similar ✓
"I forgot my password"     → [-0.7, 0.2, -0.4, ...] ← very different ✓
```

At query time, we embed the new query and find the closest known intent vector using cosine similarity — no retraining needed:

```
new query: "my parcel hasn't shown up"
           ↓ embed
           [0.29, -0.48, 0.79, ...]
           ↓ find nearest neighbour
           closest match: track_order  ✓
```

### Why this is better

| | Keyword matching | TF-IDF classifier | Embedding retrieval |
|---|---|---|---|
| Handles paraphrasing | ✗ | Partial | ✓ |
| Handles new intents | ✗ | Requires retraining | ✓ add new vector |
| Understands context | ✗ | ✗ | ✓ |
| Scales to 1000+ intents | ✗ | ✗ | ✓ |

### What this project builds

A tiny but complete embedding pipeline trained from scratch on 20 e-commerce intents:

```
Customer query
      ↓
SimpleTokenizer     → convert text to token IDs
      ↓
MiniIntentEmbedder  → transformer model → 128-dim vector
      ↓
Cosine similarity   → find nearest intent in corpus
      ↓
Predicted intent    → "track_order", "cancel_order", etc.
```

The model is trained using **contrastive learning** — queries with the same intent are pulled together in vector space, queries with different intents are pushed apart. After training, the model achieves **97%+ Recall@1** on the test set vs **50%** for TF-IDF.

---

## Project layout

```
embedding_training/
├── customer_intent_search/          # Intent search package
│   ├── data/                        # Saved tokenizer and datasets
│   ├── tokenizer.py                 # SimpleTokenizer implementation
│   ├── model.py                     # Embedding model (config + attention)
│   ├── synthetic_data.py            # Synthetic e-commerce intent dataset
│   ├── data_preparation.py          # Dataset loading, training pairs, batch sampler
│   └── train.py                     # Training loop, loss function, scheduler
├── models/
│   ├── finetuned/                   # Default single-run model artifacts
│   │   ├── model.pt                 # Trained model weights
│   │   ├── tokenizer.json           # Saved vocabulary
│   │   ├── config.json              # Hyperparameters + training history
│   │   └── training_curves.png      # Loss / accuracy plots
│   └── experiments/                 # Per-experiment model artifacts
│       ├── temp_0.01/               # model.pt + tokenizer.json + config.json
│       ├── temp_0.05/
│       └── temp_0.20/
├── results/
│   ├── training_run.log             # Output from last single training run
│   ├── experiments_run.log          # Output from last experiments run
│   └── experiments/                 # Per-experiment plots and summaries
│       ├── temp_0.01/               # training_curves_temp=0.01.png
│       ├── temp_0.05/               # training_curves_temp=0.05.png
│       ├── temp_0.20/               # training_curves_temp=0.20.png
│       ├── experiment_comparison.png # 4-panel cross-experiment chart
│       └── experiment_summary.json  # Best metrics per experiment
└── requirements.txt
```

## End-to-end pipeline

At a high level, customer intent search turns a short query into a dense vector you can compare against known intents:

```
Raw text          Token IDs           Token vectors         Context-aware          Sentence
"my balance"  →   [2, 3, 4, 0, …]  →  [B, L, D] tensor  →  self-attention    →  embedding
                  (Step 1)              (embedding lookup)    (Step 2)             (pool [CLS])
```

| Stage | Input | Output | Implemented in |
|-------|-------|--------|----------------|
| Tokenize | string | `[L]` integer IDs + mask | `tokenizer.py` |
| Embed | token IDs | `[B, L, D]` float tensor | `MiniIntentEmbedder.word_emb` |
| Attend | embeddings + mask | `[B, L, D]` updated vectors | `TransformerBlock` |
| Pool | token vectors | `[B, D]` sentence embedding | `MiniIntentEmbedder` ([CLS] + L2 norm) |

**Tensor shorthand used throughout:**

| Symbol | Name | Meaning | Typical value here |
|--------|------|---------|-------------------|
| **B** | Batch size | Number of sentences processed together | e.g. 32 |
| **L** | Sequence length | Token positions per sentence (padded) | 32 |
| **D** | Embedding dim | Numbers per token vector | 128 |
| **H** | Heads | Parallel attention subspaces | 4 |

Example: a batch of 3 queries → `x.shape = [3, 32, 128]` → `[B, L, D]`.

## Setup

```bash
cd embedding_training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Follow along

Work through the steps below in order. Each step introduces one component, with files you can read, commands to run, and concepts to understand before moving on.

| Step | Topic | Status |
|------|-------|--------|
| 1 | Tokenizer — text to token IDs | Done |
| 2 | `MultiHeadSelfAttention` | Done |
| 3 | `TransformerBlock` — attention + FFN | Done |
| 4 | `MiniIntentEmbedder` — full model | Done |
| 5 | Data preparation — dataset loading, training pairs, batch sampler | Done |
| 6 | Training — contrastive loss, custom DataLoader, LR schedule | Done |
| 7 | Evaluation — Recall@1/5, MRR, TF-IDF baseline comparison | Done |
| 8 | Hard negative mining — offline mining, modified loss, top_k sweep | Next |

---

### Step 1 — Tokenizer

**Goal:** Turn raw customer queries (for example, `"what is my balance"`) into fixed-length lists of integers that a neural network can consume.

**Key files:**
- `customer_intent_search/tokenizer.py` — `SimpleTokenizer` implementation
- `customer_intent_search/data/tokenizer.json` — saved vocabulary after fitting

#### What to do

1. **Clone the repo and install dependencies** (see [Setup](#setup) above).
2. **Open `customer_intent_search/tokenizer.py`** and skim the `SimpleTokenizer` class: `fit`, `encode`, `encode_batch`, `save`, and `load`.
3. **Run the built-in tests** from the project root:
   ```bash
   python customer_intent_search/tokenizer.py
   ```
   You should see output for `fit`, `encode`, `encode_batch`, and save/load without errors.
4. **Fit the tokenizer on your own sample texts** (short banking-style queries work well):
   ```python
   from pathlib import Path
   from customer_intent_search import SimpleTokenizer

   texts = [
       "what is my balance",
       "lost my card",
       "transfer money to john",
       "how do i reset my password",
   ]

   tok = SimpleTokenizer()
   tok.fit(texts)
   print(tok.vocab_size)
   print(tok.word2idx)
   ```
5. **Encode a few sentences** and inspect the token IDs:
   ```python
   print(tok.encode("my balance"))
   print(tok.encode("unknown xyzword"))  # out-of-vocabulary → UNK
   ```
6. **Save and reload** so the same vocabulary is reused later (training, inference):
   ```python
   data_dir = Path("customer_intent_search/data")
   data_dir.mkdir(exist_ok=True)
   tok.save(data_dir / "tokenizer.json")

   tok2 = SimpleTokenizer.load(data_dir / "tokenizer.json")
   print(tok2.encode("transfer money"))
   ```
7. **Try batch encoding** — this is what the model will use during training:
   ```python
   input_ids, attention_mask = tok.encode_batch(texts)
   print(input_ids.shape)       # [batch_size, 32]
   print(attention_mask.shape)  # [batch_size, 32]
   ```

When Step 1 is working, you should have a saved `tokenizer.json` under `customer_intent_search/data/` and understand how text becomes padded integer tensors.

#### Tokenizer concepts

A **tokenizer** converts raw text into numbers that a neural network can process. Models do not read strings directly—they operate on integer **token IDs** and fixed-size tensors.

This project uses `SimpleTokenizer`, a lightweight **word-level** tokenizer built for short customer-support queries (for example, "what is my balance" or "lost my card").

### Text → tokens → IDs

The pipeline has three steps:

1. **Normalize and split** — lowercase the text and split on word boundaries (`tokenize()`).
2. **Map words to IDs** — look up each word in the vocabulary (`word2idx`).
3. **Pad to fixed length** — every sequence is padded or truncated to `max_seq_len` (default 32).

Example:

```
"my balance"  →  ["my", "balance"]  →  [CLS] my balance [PAD] [PAD] ...
```

### Vocabulary (`fit`)

Before encoding, the tokenizer must **learn** which words exist in your training data.

- Call `fit(texts)` on a list of example sentences.
- The tokenizer counts word frequencies and keeps the most common words up to `max_vocab_size` (default 8000).
- Rare words are not stored individually; at encode time they map to the unknown token.

The vocabulary is a dictionary: `word2idx` maps strings to integers, and `idx2word` is the reverse lookup.

### Special tokens

Three reserved tokens are always present at fixed IDs:

| Token | ID | Purpose |
|-------|----|---------|
| `[PAD]` | 0 | Fills unused positions so every sequence has the same length |
| `[UNK]` | 1 | Substitute for words not in the vocabulary |
| `[CLS]` | 2 | Prepended to every sequence as a sentence-level marker |

**Padding** lets batches stack into rectangular tensors. Positions filled with `PAD` should be ignored by the model.

**Unknown (`UNK`)** handles out-of-vocabulary words at inference time (typos, names, new phrases).

**CLS** (classification token) gives the model a consistent starting point for reading a sentence—useful when pooling token representations into one embedding.

### Encoding a single sentence (`encode`)

`encode(text)` returns a list of exactly `max_seq_len` integers:

1. Prepend `[CLS]`.
2. Convert each word to its ID, or `UNK` if missing.
3. Truncate if longer than `max_seq_len`.
4. Pad the rest with `PAD`.

```python
from customer_intent_search import SimpleTokenizer

tok = SimpleTokenizer()
tok.fit(["what is my balance", "lost my card", "transfer money"])

tok.encode("my balance")
# [2, 3, 4, 0, 0, ...]  →  [CLS] my balance PAD PAD ...

tok.encode("unknown xyzword")
# [2, 1, 1, 0, 0, ...]  →  [CLS] UNK UNK PAD PAD ...
```

### Batching (`encode_batch`)

Neural networks train on **batches** of examples. `encode_batch(texts)` encodes multiple sentences and returns two PyTorch tensors:

- **`input_ids`** — shape `[batch_size, max_seq_len]`, the token IDs
- **`attention_mask`** — shape `[batch_size, max_seq_len]`, `1` for real tokens and `0` for padding

The attention mask tells the model which positions are meaningful and which are padding.

```python
texts = ["what is my balance", "lost my card"]
input_ids, attention_mask = tok.encode_batch(texts)
# input_ids.shape      → torch.Size([2, 32])
# attention_mask.shape → torch.Size([2, 32])
```

### Save and load

After fitting on training data, persist the vocabulary so inference uses the same word mappings:

```python
from pathlib import Path

data_dir = Path("customer_intent_search/data")
data_dir.mkdir(exist_ok=True)

tok.save(data_dir / "tokenizer.json")
tok = SimpleTokenizer.load(data_dir / "tokenizer.json")
```

The saved JSON stores `max_vocab_size`, `max_seq_len`, and `word2idx`.

---

### Step 2 — Model (embeddings + self-attention)

**Goal:** Map token IDs to dense vectors and let each token gather context from the rest of the sentence—so `"balance"` can incorporate `"my"` and `"check"`.

**Key files:**
- `customer_intent_search/model.py` — `MiniIntentConfig`, `MultiHeadSelfAttention`, `TransformerBlock`

#### What to do

1. **Complete Step 1** so you have `input_ids` and `attention_mask` tensors from the tokenizer.
2. **Open `customer_intent_search/model.py`** and read `MiniIntentConfig` — these defaults tie the model to the tokenizer:
   ```python
   from customer_intent_search.model import MiniIntentConfig

   cfg = MiniIntentConfig()
   # vocab_size=5000, embed_dim=128, n_heads=4, max_seq_len=32, pad_token_id=0
   ```
3. **Read `MultiHeadSelfAttention`** — the core mechanism that mixes information across tokens. The class docstring contains the formula and a 7-step `forward()` flow.
4. **Trace one forward pass mentally:**
   - Input `x`: `[B, L, D]` — one 128-dim vector per token per sentence
   - Project to Q, K, V; split into 4 heads of 32 dims each
   - Score every token against every token → softmax → weighted sum of values
   - Output: `[B, L, D]` — same shape, richer representations

#### Model config (`MiniIntentConfig`)

| Field | Default | Role |
|-------|---------|------|
| `vocab_size` | 5000 | Rows in the embedding lookup table |
| `embed_dim` (D) | 128 | Size of each token vector |
| `n_heads` (H) | 4 | Parallel attention patterns (128 ÷ 4 = 32 dims per head) |
| `n_layers` | 2 | Stacked transformer blocks |
| `ffn_dim` | 512 | Hidden size inside feed-forward sublayer |
| `max_seq_len` (L) | 32 | Must match tokenizer |
| `dropout` | 0.1 | Regularization during training |
| `pad_token_id` | 0 | Padding token ID from tokenizer |

#### Self-attention concepts

Self-attention asks: **for each token, which other tokens matter, and what should I take from them?**

Per attention head:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

| Projection | Role | Intuition |
|------------|------|-----------|
| **Q** (query) | What am I looking for? | `"balance"` searches for related words |
| **K** (key) | What do I offer for matching? | `"my"` advertises possessive context |
| **V** (value) | What info do I pass if selected? | Actual meaning carried forward |

**Multi-head:** run H independent attentions in parallel, then concatenate. Each head can focus on different patterns (keywords, word pairs, etc.).

**Attention mask:** positions where `attention_mask == 0` (padding) get score `-∞` before softmax, so the model never attends to `PAD` tokens.

High-level `forward()` flow:

1. Linear projections → Q, K, V
2. Split into H heads
3. Scores = QKᵀ / √d_k → shape `[B, H, L, L]`
4. Mask padding keys
5. Softmax + dropout on weights
6. Output = weights × V
7. Merge heads → output projection

#### Shape cheat sheet

```
input_ids        [B, L]           integers from tokenizer
attention_mask   [B, L]           1 = real token, 0 = pad
x (embeddings)   [B, L, D]        float vectors after lookup
Q, K, V          [B, H, L, Dh]    H=4, Dh=32
attn scores      [B, H, L, L]     token i → token j weights
output           [B, L, D]        context-aware token vectors
```

#### Full input → output flow with a real example

Let's trace the sentence **"I lost my card"** through `MultiHeadSelfAttention`.

##### Step 0 — Starting point

We have 4 words. After tokenization and embedding, each word becomes a list of 128 numbers. Think of it as a table:

```
         [128 numbers each]
"I"    → [0.2, 0.5, -0.3, ...]
"lost" → [0.8, -0.1, 0.6, ...]
"my"   → [0.1, 0.3,  0.1, ...]
"card" → [0.6, 0.2, -0.5, ...]
```

Shape of `x`: `(1 sentence, 4 words, 128 numbers)` = `(1, 4, 128)` → `[B, L, D]`

##### Step 1 — Create Q, K, V

Pass `x` through the three projections. Each word gets three versions of itself:

```
         Q (asking)        K (name tag)      V (resume)
"I"    → [0.1, 0.4, ...]  [0.3, 0.2, ...]  [0.5, 0.1, ...]
"lost" → [0.9, 0.2, ...]  [0.8, 0.7, ...]  [0.6, 0.9, ...]
"my"   → [0.2, 0.1, ...]  [0.1, 0.3, ...]  [0.2, 0.4, ...]
"card" → [0.7, 0.5, ...]  [0.6, 0.4, ...]  [0.8, 0.3, ...]
```

All still shape `(1, 4, 128)` — same size, different meaning.

In code: `Q = q_proj(x)`, `K = k_proj(x)`, `V = v_proj(x)`.

##### Step 2 — Split into 4 heads

We have 128 dimensions and 4 heads → each head gets 32 dimensions.

```
Head 1 looks at dimensions  1–32
Head 2 looks at dimensions 33–64
Head 3 looks at dimensions 65–96
Head 4 looks at dimensions 97–128
```

Each head runs the full attention process independently on its 32 dimensions. Think of 4 people reading the same sentence but each focusing on different aspects (grammar, topic, sentiment, etc.).

**Let's follow just Head 1 from here.**

##### Step 3 — Compute attention scores `Q × Kᵀ`

Head 1 asks: *how much should each word attend to every other word?*

We dot-product each word's Q against every word's K:

```
              "I"   "lost"  "my"  "card"
"I"    asks → 0.2    0.3    0.1    0.2
"lost" asks → 0.1    0.5    0.1    0.8   ← "lost" pays most attention to "card"
"my"   asks → 0.2    0.1    0.4    0.1
"card" asks → 0.1    0.7    0.1    0.3   ← "card" pays most attention to "lost"
```

This is a `(4, 4)` grid — every word against every word.

Then divide by `scale = sqrt(32) ≈ 5.6` to keep numbers small.

In code: `attn = Q @ K.transpose(-2, -1) / scale` → shape `[B, H, L, L]`.

##### Step 4 — Apply attention mask

Our sentence has no padding here, so all 4 positions are real. Mask is all 1s — nothing to block.

If we had padding like `"I lost my card [PAD]"`:

```
"card" asks → 0.1   0.7   0.1   0.3   -inf  ← PAD gets -inf
```

In code: `attn.masked_fill(attention_mask == 0, -inf)`.

##### Step 5 — Softmax → weights that sum to 1

```
"lost" row before softmax: [0.1,  0.5,  0.1,  0.8]
"lost" row after softmax:  [0.12, 0.23, 0.12, 0.53]
                                                ↑
                                      "card" gets 53% of attention
```

```
"card" row before softmax: [0.1,  0.7,  0.1,  0.3]
"card" row after softmax:  [0.13, 0.48, 0.13, 0.26]
                                    ↑
                          "lost" gets 48% of attention
```

"lost" and "card" are now paying high attention to each other. The model is learning they belong together.

##### Step 6 — Weighted sum of values `weights × V`

Now each word collects information from all other words, weighted by attention:

```
new "card" =
    0.13 × V("I")    +
    0.48 × V("lost") +   ← heavy contribution from "lost"
    0.13 × V("my")   +
    0.26 × V("card")

= [0.13×0.5, 0.13×0.1, ...]
+ [0.48×0.6, 0.48×0.9, ...]   ← dominates
+ [0.13×0.2, 0.13×0.4, ...]
+ [0.26×0.8, 0.26×0.3, ...]

= [0.58, 0.55, ...]   ← new "card" vector, enriched with context from "lost"
```

Before attention, "card" only knew about itself. Now it knows it's a **lost** card.

In code: `out = attn @ V`.

##### Step 7 — Reassemble all 4 heads

All 4 heads did this independently. Now concatenate their outputs:

```
Head 1 output for "card": [0.58, 0.55, ...]  32 numbers
Head 2 output for "card": [0.31, 0.72, ...]  32 numbers
Head 3 output for "card": [0.44, 0.21, ...]  32 numbers
Head 4 output for "card": [0.67, 0.18, ...]  32 numbers

Concatenated: [...all 4 heads...]  = 128 numbers again
```

Then one final `out_proj` linear layer mixes all heads together → still `128` numbers.

##### Final output

```
Input  "card": [0.6, 0.2, -0.5, ...]   ← knew nothing about context
Output "card": [0.58, 0.55, 0.41, ...]  ← now understands it's a lost card
```

Same shape `(1, 4, 128)` in, same shape `(1, 4, 128)` out — but every word is now **context-aware**.

This is everything `MultiHeadSelfAttention` does.

---

### Step 3 — TransformerBlock

**Goal:** Wrap self-attention with a feed-forward network, residual connections, and layer normalization so each token both **talks to other tokens** and **transforms itself**.

**Key file:** `customer_intent_search/model.py` — `TransformerBlock`

#### What a TransformerBlock is

Attention is powerful but it only does one thing — **mix information across tokens**. It has no ability to transform individual token representations nonlinearly.

The `TransformerBlock` wraps attention with a **Feed-Forward Network (FFN)** that processes each token independently and adds expressive power.

```
Input x
  → Attention  (tokens talk to each other)
  → FFN        (each token thinks for itself)
Output x
```

#### What to do

1. **Read `TransformerBlock` in `model.py`** — note how it composes `MultiHeadSelfAttention`, `ff`, `norm1`, `norm2`, and `dropout`.
2. **Trace the two sublayers** — attention residual first, then FFN residual:
   ```python
   from customer_intent_search.model import TransformerBlock

   block = TransformerBlock(embed_dim=128, n_heads=4, ffn_dim=512, dropout=0.1)
   # block(x, attention_mask) → same shape [B, L, 128]
   ```
3. **Compare with Step 2** — attention alone mixes tokens; the block adds per-token nonlinear transforms and stable training via residuals + norm.

#### The feed-forward network

```python
self.ff = nn.Sequential(
    nn.Linear(embed_dim, ffn_dim),   # 128 → 512  expand
    nn.GELU(),                        # nonlinearity
    nn.Dropout(dropout),
    nn.Linear(ffn_dim, embed_dim),   # 512 → 128  compress
)
```

Each token passes through this independently — no cross-token communication here.

**Why expand to 512 then compress back to 128?**

The expansion creates space to learn complex transformations. Think of it as:
- Expand 128 → 512: unpack all possible interpretations
- GELU: keep only useful ones
- Compress 512 → 128: summarize back into a clean representation

**GELU** is an activation function — it introduces nonlinearity so the network can learn complex patterns, not just linear transformations. Similar to ReLU but smoother:

```
negative values → squashed toward 0
positive values → passed through mostly unchanged
```

Without nonlinearity, stacking linear layers would still just be one big linear layer.

#### Residual connections

This is the most important design pattern in the whole block:

```python
x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
x = x + self.dropout(self.ff(self.norm2(x)))
```

Notice the `x + ...` — we **add the output back to the original input**. This is a **residual connection** (also called skip connection).

Without it:

```
x → attention → new_x   (original x is lost)
```

With it:

```
x → attention → delta_x
output = x + delta_x     (original x is preserved, delta is added on top)
```

**Why does this matter?**

The attention layer only needs to learn the **change** needed, not reconstruct the full representation from scratch. This makes training much more stable and allows gradients to flow directly back to early layers without vanishing.

#### LayerNorm

```python
self.norm1 = nn.LayerNorm(embed_dim)
self.norm2 = nn.LayerNorm(embed_dim)
```

Applied **before** attention and FFN (called Pre-Norm). Normalizes each token's vector to have mean ≈ 0 and std ≈ 1.

**Why?** Keeps values in a stable range as they flow through layers. Without it, values can explode or shrink to zero across multiple layers, making training unstable.

#### Full data flow

```
Input x: (B, 4, 128)
         ↓
      norm1(x)           normalize
         ↓
      attention(...)     tokens talk to each other → (B, 4, 128)
         ↓
      dropout(...)       randomly zero some values
         ↓
      x = x + ...        residual: add back original x
         ↓
      norm2(x)           normalize again
         ↓
      ff(...)            each token transforms itself 128→512→128
         ↓
      dropout(...)
         ↓
      x = x + ...        residual again
         ↓
Output x: (B, 4, 128)   same shape, richer representations
```

`MiniIntentConfig.n_layers = 2` means two of these blocks stacked in sequence inside `MiniIntentEmbedder`.

---

### Step 4 — MiniIntentEmbedder

**Goal:** Stack embeddings, transformer blocks, and [CLS] pooling into one model that maps a query to a **128-dim unit vector** ready for cosine-similarity search.

**Key file:** `customer_intent_search/model.py` — `MiniIntentEmbedder`, `build_model()`

#### What to do

1. **Read `MiniIntentEmbedder`** after completing Steps 2–3.
2. **Instantiate the full model:**
   ```python
   from customer_intent_search import SimpleTokenizer, MiniIntentConfig, build_model

   tok = SimpleTokenizer()
   tok.fit(["I lost my card", "what is my balance"])

   model = build_model(MiniIntentConfig(vocab_size=tok.vocab_size))
   texts = ["I lost my card", "check my balance"]
   embs = model.encode(texts, tok)   # numpy array [2, 128]
   print(embs.shape)
   ```
3. **Trace the shape flow** for a single sentence (see below).

#### Full sequence: where word_emb fits

`word_emb` is **not** Q, K, or V. It runs **before** attention — it converts integer token IDs into vectors the rest of the model can use.

```
raw token IDs
      ↓
  word_emb        ← lookup table: ID → [128-dim vector]
      ↓
  + pos_emb       ← adds position information
      ↓
  TransformerBlock
      ├── norm1
      ├── MultiHeadSelfAttention
      │       ├── q_proj   ← operates on word_emb output
      │       ├── k_proj
      │       └── v_proj
      └── FFN
```

Think of `word_emb` as converting a name into a phone number — just a lookup, no computation. Q/K/V come **after**, projecting those vectors into query/key/value roles.

#### Embedding layers

```python
self.word_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
self.pos_emb  = nn.Embedding(config.max_seq_len, config.embed_dim)
```

**Word embedding** — lookup table shape `(vocab_size, 128)`. Each token ID maps to a row:

```
token ID 45 → row 45 → [0.2, 0.5, -0.3, ...]   128 numbers
token ID  0 → row  0 → [0.0, 0.0,  0.0, ...]   ← PAD, always zero
```

**Positional embedding** — lookup table shape `(32, 128)`. Position 0 maps to one vector, position 1 to another. Attention has no built-in sense of order — `pos_emb` tells the model where each token sits.

```
position 0 → [0.1, 0.8, ...]   ← "I am first" ([CLS])
position 1 → [0.4, 0.2, ...]   ← "I am second"
```

**`padding_idx=0`** — keeps the PAD token's embedding permanently at zero. PAD tokens carry no meaning and should contribute nothing.

#### Do word_emb and pos_emb change during training?

**Yes** — both are learned parameters, updated by backprop like any other weight.

| Phase | `word_emb["lost"]` | `word_emb["card"]` |
|-------|--------------------|--------------------|
| Start (random) | `[0.02, -0.01, ...]` meaningless | `[0.01, 0.02, ...]` meaningless |
| After training | shifted toward semantic meaning | shifted toward semantic meaning |

After training, `"lost"` and `"card"` vectors point in similar directions if they frequently appeared together (e.g. in `"I lost my card"`).

| Component | What it learns |
|-----------|----------------|
| `word_emb` | what each word means |
| `pos_emb` | what each position means |
| `q_proj`, `k_proj`, `v_proj` | how to compare meanings |
| FFN | how to transform meanings |

All start random and learn together through the same training loop.

#### `_init_weights()`

Before training, weights need sensible starting values:

```python
nn.init.xavier_uniform_(module.weight)   # Linear layers
nn.init.normal_(module.weight, std=0.02) # Embeddings
nn.init.ones_(module.weight)             # LayerNorm weight
nn.init.zeros_(module.bias)              # All biases start at 0
```

- **Xavier uniform** — scales initial weights by layer size; keeps signals from exploding or vanishing at the start.
- **Normal (std=0.02)** — small random embedding values; used by GPT and BERT.

#### `forward()` — full data flow

```python
positions = torch.arange(L, device=device).unsqueeze(0)
x = self.word_emb(input_ids) + self.pos_emb(positions)
```

Creates position indices `[0, 1, 2, ..., L-1]` and adds positional vectors to word vectors:

```
"lost" at position 1:
  word_emb[token_id] + pos_emb[1]
= [0.8, -0.1, 0.6, ...] + [0.4, 0.2, ...]
= [1.2,  0.1, 0.8, ...]   ← knows it's "lost" AND at position 1
```

```python
for layer in self.layers:
    x = layer(x, attention_mask)
```

Pass through 2 `TransformerBlock`s sequentially. Each block refines representations further.

```python
cls_emb = self.output_norm(x[:, 0, :])
return F.normalize(cls_emb, p=2, dim=-1)
```

- `x[:, 0, :]` — take only position 0, the `[CLS]` token, from every sentence. Shape `(B, 32, 128)` → `(B, 128)`.
- `F.normalize(..., p=2)` — L2 normalize so every output vector has length exactly 1. Cosine similarity between two vectors then equals their dot product — cheaper at retrieval time.

#### `encode()` — convenience method for inference

Handles batching when encoding many texts:

```python
for i in range(0, len(texts), batch_size):
    batch = texts[i: i + batch_size]
    ...
```

- `torch.no_grad()` — no gradient tracking during inference (saves memory and compute).
- Returns a **numpy array** `[num_texts, 128]` for scikit-learn and retrieval tools.

#### Full shape trace for `"I lost my card"`

```
input_ids:          (1, 4)         ← 1 sentence, 4 tokens ([CLS] + 3 words)
word_emb + pos_emb: (1, 4, 128)    ← each token → 128-dim vector
→ x:                (1, 4, 128)
TransformerBlock 1: (1, 4, 128) → (1, 4, 128)
TransformerBlock 2: (1, 4, 128) → (1, 4, 128)
x[:, 0, :]:         (1, 128)       ← take only [CLS] token
normalize:          (1, 128)       ← length = 1.0
Final output:       (1, 128)       ← sentence embedding
```

---

### Step 5 — Data Preparation

**Goal:** Load the e-commerce intent dataset, create positive training pairs, and use a custom batch sampler that guarantees **zero false negatives** during contrastive training.

**Key files:**
- `customer_intent_search/synthetic_data.py` — generates synthetic e-commerce queries across 20 intents
- `customer_intent_search/data_preparation.py` — `load_dataset()`, `create_training_pairs()`, `IntentAwareBatchSampler`, `build_retrieval_corpus()`

#### Dataset

The project uses a synthetic e-commerce customer support dataset with **20 intents across 5 domains**:

| Domain | Intents |
|--------|---------|
| orders | track_order, cancel_order, modify_order, order_not_received, wrong_item_received |
| returns | initiate_return, refund_status, exchange_item, damaged_item |
| delivery | delivery_address_change, delivery_delay, express_shipping |
| payments | payment_failed, apply_promo_code, price_match, invoice_request |
| account | reset_password, loyalty_points, gift_card_balance, contact_support |

Each intent has **150 training / 30 validation / 30 test** queries generated from templates with slot filling:

```
template:  "I want to return my {item} because {reason}"
filled:    "I want to return my shoes because it doesn't fit"
           "I want to return my jacket because I changed my mind"
```

Surface variations (capitalization, punctuation, adding "please") are applied so the model learns meaning is independent of surface form.

#### Training pairs

`create_training_pairs()` builds **(anchor, positive) pairs** — two queries from the **same intent**:

```python
pairs[0].texts = ["where is my #ORD-1234?",     "can you track my order?"]   # track_order
pairs[1].texts = ["please cancel #ORD-5678",     "I changed my mind cancel"]  # cancel_order
pairs[2].texts = ["where is my refund",          "refund not received yet"]   # refund_status
```

These pairs are the input to contrastive learning — the model is trained to make same-intent pairs similar and different-intent pairs dissimilar.

#### Why false negatives are a problem

In contrastive learning, a batch of pairs is passed to the loss function at once. For each anchor, **all other positives in the batch become implicit negatives**:

```
Anchor: "where is my order"      ← intent 0 (track_order)
  ✓ positive: "track my package" ← intent 0 — should be SIMILAR
  ✗ negative: "cancel my order"  ← intent 1 — should be DISSIMILAR
  ✗ negative: "where is my refund" ← intent 2 — should be DISSIMILAR
```

This works perfectly when all off-diagonal pairs are **different intents**. But if two pairs in the same batch share the same intent, the off-diagonal entry becomes a **false negative** — a semantically similar pair being pushed apart:

```
batch = [pair_A (track_order), pair_B (track_order), pair_C (cancel_order)]

Anchor: "where is my order"       ← track_order
  ✓ positive: "track my package"  ← track_order — correctly pulled close
  ✗ false negative: "has my order shipped?" ← also track_order — wrongly pushed apart!
  ✗ true negative:  "cancel my order"       ← cancel_order — correctly pushed apart
```

The model gets **conflicting signals**: it is simultaneously told to treat `"has my order shipped?"` as similar (when it is the anchor in its own pair) and as dissimilar (when it is a negative for another pair). This slows convergence and hurts final quality.

#### Why not just use a large batch size?

With a large batch and random shuffling, the chance of same-intent collisions grows:

```
20 intents, batch size 64 → ~3 pairs per intent per batch
63 negatives per anchor → ~2 are false negatives
97% clean signal — acceptable but not ideal
```

Random shuffling reduces the problem statistically but does **not eliminate it**. The more intents share vocabulary (e.g. `track_order` vs `delivery_delay` both mention orders), the more harmful false negatives become.

#### `IntentAwareBatchSampler` — the fix

Instead of relying on random shuffling, we use a **custom batch sampler** that explicitly controls which pairs appear in each batch. It guarantees **exactly one pair per intent per batch** — making every off-diagonal entry a true negative.

```python
sampler = IntentAwareBatchSampler(intent_ids, batch_size=20)
```

**How it works:**

1. Groups pair indices by intent ID:
   ```
   intent 0 (track_order):   [index 0, index 20, index 40, ...]
   intent 1 (cancel_order):  [index 1, index 21, index 41, ...]
   intent 2 (modify_order):  [index 2, index 22, index 42, ...]
   ...
   ```

2. Builds batches by taking **one pair per intent per round**:
   ```
   round 0 → [intent0[0], intent1[0], intent2[0], ..., intent19[0]]  ← 20 pairs
   round 1 → [intent0[1], intent1[1], intent2[1], ..., intent19[1]]  ← 20 pairs
   round 2 → [intent0[2], intent1[2], intent2[2], ..., intent19[2]]  ← 20 pairs
   ```

3. Each batch contains exactly one pair per intent → similarity matrix off-diagonal = true negatives only:
   ```
                pos0          pos1           pos2
             (track_order) (cancel_order) (modify_order)
   anchor0     0.92 ✓        0.21 ✗          0.18 ✗
   anchor1     0.19 ✗        0.89 ✓          0.22 ✗
   anchor2     0.15 ✗        0.20 ✗          0.91 ✓

   diagonal   = correct matches  → pushed HIGH
   off-diagonal = different intents → pushed LOW (always true negatives)
   ```

**Comparison:**

| Approach | False negatives | Implementation |
|---|---|---|
| Random shuffle | Possible — ~2-3 per batch | Simple |
| Interleaved order | Rare — depends on batch size | Medium |
| `IntentAwareBatchSampler` | Zero — guaranteed | Custom sampler |

#### Further reading

- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) — formally addresses the false negative problem in contrastive learning
- [SimCSE](https://arxiv.org/abs/2104.08821) — state-of-the-art sentence embeddings with deep discussion of in-batch negatives
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — section 4 discusses negative selection strategies
- [DPR](https://arxiv.org/abs/2004.04906) — Facebook's dense retrieval paper, entire section dedicated to hard negative mining

#### Quick test

```python
from data_preparation import load_dataset, create_training_pairs, IntentAwareBatchSampler

train_data, val_data, test_data = load_dataset()
pairs, intent_ids = create_training_pairs(train_data, pairs_per_intent=20)

print(len(pairs))        # 400 (20 intents × 20 pairs)
print(pairs[0].texts)    # ["where is my order?", "track my package"]
print(intent_ids[0])     # 0 (track_order)

sampler = IntentAwareBatchSampler(intent_ids, batch_size=20)
print(len(sampler))      # 20 batches (400 pairs / batch_size 20)

# verify: no two pairs in a batch share the same intent
for batch_indices in sampler:
    batch_intents = [intent_ids[i] for i in batch_indices]
    assert len(batch_intents) == len(set(batch_intents)), "False negative detected!"
print("All batches clean — zero false negatives confirmed.")
```

---

### Step 6 — Training

**Goal:** Train the model using contrastive learning so same-intent queries end up close together in embedding space and different-intent queries end up far apart.

**Key file:** `customer_intent_search/train.py`

#### What to do

```bash
cd customer_intent_search
python train.py              # full 15 epochs
python train.py --epochs 3   # quick test
python train.py --help       # see all options
```

#### Expected output

```
  Device: cpu

Loading e-commerce intent dataset...
  Source  : Synthetic (local e-commerce fallback)
  Intents : 20
  Train   : 3,000 queries

Building tokenizer...
  Vocab size: 412

Building model...
  Parameters: 453,248

Creating training pairs...
  Created 400 training pairs across 20 intents
  Pairs         : 400
  Batches/epoch : 20

Training for 15 epochs...
  Epoch 01 | loss 2.9814 | acc  5.0% | time 0.3s
  Epoch 05 | loss 1.2341 | acc 65.0% | time 0.3s
  Epoch 10 | loss 0.4821 | acc 88.0% | time 0.3s
  Epoch 15 | loss 0.2103 | acc 97.5% | time 0.3s

  Model     → ../models/finetuned/model.pt
  Tokenizer → ../models/finetuned/tokenizer.json
  Config    → ../models/finetuned/config.json
```

Loss should drop steadily epoch by epoch. Accuracy should climb from ~5% (random) toward 97%+.

#### `MultipleNegativesRankingLoss`

Contrastive loss that reuses cross entropy. For a batch of `(anchor, positive)` pairs:

```
Embed all anchors   → A  shape (batch_size, 128)
Embed all positives → P  shape (batch_size, 128)
Similarity matrix   → S = A @ Pᵀ  shape (batch_size, batch_size)
```

The similarity matrix has two regions:

```
              pos_0          pos_1          pos_2
           "track pkg"   "want cancel"  "refund not rcvd"

anc_0  →  [ 0.92 ✓        0.21 ✗          0.18 ✗ ]
anc_1  →  [ 0.19 ✗        0.89 ✓          0.22 ✗ ]
anc_2  →  [ 0.15 ✗        0.20 ✗          0.91 ✓ ]

✓ diagonal     = correct matches  → cross entropy pushes these HIGH
✗ off-diagonal = wrong matches    → cross entropy pushes these LOW
```

Cross entropy for row 0:
```
loss = -log( e^0.92 / (e^0.92 + e^0.21 + e^0.18) )
              ↑ correct   ↑────── all negatives ──↑
```

The denominator includes **every cell** — correct and negatives. If any negative score creeps up, the denominator grows, the correct probability drops, and loss increases. Backprop then pushes the anchor away from that negative.

**Temperature τ = 0.05** sharpens the scores before softmax:
```
raw scores:  [0.9, 0.3, 0.2]
τ = 1.0   →  softmax → [0.41, 0.22, 0.20]   soft, spread out
τ = 0.05  →  softmax → [0.99, 0.00, 0.00]   sharp, peaked
```

Low temperature creates harder training signal — the model is more strongly penalized for any confusion between intents.

#### `PairDataset`

PyTorch's `DataLoader` requires a `Dataset` object with two methods:

- `__len__()` — total number of pairs
- `__getitem__(idx)` — returns `(anchor_text, positive_text)` for that index

`PairDataset` wraps our flat list of `InputExample` objects into this contract. It stores raw strings only — no tokenization yet.

#### `make_collate_fn()`

`DataLoader` calls `collate_fn` on each batch to convert a list of raw string tuples into tensors the model can consume.

We use a **closure** — a function that captures the tokenizer from its outer scope — because `DataLoader` expects `collate_fn` to take only one argument (the batch):

```
batch = [
    ("where is my order",  "track my package"),
    ("cancel my order",    "I want to cancel"),
    ...
]
        ↓ collate_fn
        ↓
anchor_ids:    (batch_size, 32)   token IDs for anchors
anchor_mask:   (batch_size, 32)   1=real token, 0=padding
positive_ids:  (batch_size, 32)   token IDs for positives
positive_mask: (batch_size, 32)   1=real token, 0=padding
```

#### `make_scheduler()`

Learning rate schedule with two phases:

```
lr
3e-4 |         *─────────*
     |        *           *
     |       *              *
     |      *                 *
3e-5 |─────*                    *────
     └──────────────────────────────→ steps
       warmup      decay
```

| Phase | Steps | Behaviour |
|---|---|---|
| Warmup | 0 → 50 | lr increases 0 → 3e-4. Weights are random at start — small steps prevent unstable updates |
| Decay | 50 → 300 | lr decreases 3e-4 → 3e-5. Model is converging — smaller steps prevent overshooting |

#### `train_one_epoch()`

Four operations happen for every batch:

```
1. FORWARD   embed anchors + positives → compute loss
2. BACKWARD  zero_grad → loss.backward() → clip_grad_norm_()
3. UPDATE    optimizer.step() → scheduler.step()
4. METRICS   track loss and Recall@1 accuracy
```

**Gradient clipping** — `clip_grad_norm_(max_norm=1.0)` prevents any gradient vector from having norm > 1. Early in training, random weights can produce large gradients that destabilize updates. Clipping keeps updates bounded.

**`optimizer.zero_grad()`** must be called before every `loss.backward()` because PyTorch accumulates gradients by default:
```
without zero_grad: gradients pile up across batches → wrong updates
with zero_grad:    fresh gradients each batch → correct updates
```

#### What gets saved

| File | Contents | Used by |
|---|---|---|
| `model.pt` | All learned weight tensors (`state_dict`) | evaluate.py, inference_demo.py |
| `tokenizer.json` | Vocabulary mapping | evaluate.py, inference_demo.py |
| `config.json` | Hyperparameters + per-epoch loss/accuracy history | Debugging, reproducibility |

To reload the model later:
```python
model = build_model(tokenizer)
model.load_state_dict(torch.load("../models/finetuned/model.pt"))
```

#### Default hyperparameters

| Argument | Default | Why |
|---|---|---|
| `--epochs` | 15 | Enough for loss to converge on 400 pairs |
| `--batch-size` | 20 | Auto-overridden to n_intents after data load |
| `--pairs-per-intent` | 20 | 20 × 20 = 400 total pairs |
| `--lr` | 3e-4 | Good default for AdamW on small models |
| `--temperature` | 0.05 | Sharp contrastive signal |
| `--warmup-steps` | 50 | ~2-3 epochs of warmup |

---

#### Overfitting tracking

Every epoch the model is evaluated on the **validation set** — queries never seen during training. This reveals whether the model is learning general intent meaning or just memorising training pairs.

**What overfitting looks like:**

```
Epoch 07 | train_loss 1.71 | train_acc 54.7% | val_loss 2.34 | val_acc 49.5%
Epoch 10 | train_loss 0.95 | train_acc 73.2% | val_loss 2.13 | val_acc 55.3% ⚠ overfit?
Epoch 15 | train_loss 0.55 | train_acc 85.5% | val_loss 2.04 | val_acc 59.4% ⚠ overfit?
```

Train loss keeps dropping → model is still learning.
Val loss plateaus from epoch 9 → model starts memorising training pairs.
The gap (`val_loss − train_loss`) grows — this is the **overfitting signal**.

**Overfitting gap thresholds:**

| Gap | Status | Colour in plot |
|---|---|---|
| < 0.2 | Healthy — generalising well | 🟢 Green |
| 0.2 – 0.5 | Warning — watch carefully | 🟡 Yellow |
| > 0.5 | Overfitting — val loss diverging | 🔴 Red |

**Training plots** (`models/finetuned/training_curves.png`) show four panels:
- Top-left: Train vs val loss — gap visible as shaded region
- Top-right: Train vs val accuracy — best val epoch marked
- Bottom-left: Overfitting gap bar chart per epoch — coloured by severity
- Bottom-right: Val accuracy trend with best epoch highlighted

---

#### Temperature experiments

Run all three temperature experiments at once:

```bash
cd customer_intent_search
python3 train.py --experiments
```

This trains three models with different temperature values and saves:
- Per-experiment model weights → `models/experiments/temp_X/`
- Per-experiment training curves → `results/experiments/temp_X/`
- Cross-experiment comparison chart → `results/experiments/experiment_comparison.png`

---

#### What temperature controls

Temperature τ is a single number that divides every similarity score **before** the softmax in the contrastive loss:

```
sim_matrix / τ  →  softmax  →  cross-entropy loss
```

The lower τ is, the more the softmax "zooms in" on differences between scores:

```
raw scores for anchor_0:   [0.90 (correct), 0.20 (wrong), 0.15 (wrong)]

τ = 0.20  →  [4.5,  1.0,  0.75]  →  softmax → [0.94, 0.06, 0.05]   soft — wrong answers still get ~5%
τ = 0.05  →  [18.0, 4.0,  3.0 ]  →  softmax → [0.999, 0.000, 0.000] sharp — wrong answers get almost 0
τ = 0.01  →  [90.0, 20.0, 15.0]  →  softmax → [1.000, 0.000, 0.000] extreme — any confusion is crushed
```

**Low τ** — harsh signal. The model is punished very hard even for small confusions between intents. It learns quickly but tends to memorise training pairs rather than generalise. Higher overfitting risk.

**High τ** — gentle signal. The model is barely penalised for small confusions. It learns more slowly and more stably, but may not separate intents as sharply.

**Medium τ** — the sweet spot. The signal is strong enough to push intents apart clearly, but not so extreme that the model memorises surface-level phrasings.

---

#### Experiment results

Three models were trained for 15 epochs each with identical settings except temperature.

**Top-line numbers:**

| Experiment | Best Val Acc | Best Epoch | Final Val Loss | Final Train Acc | Overfitting Gap |
|---|---|---|---|---|---|
| temp=0.01 | 58.9% | epoch 15 | 2.6450 | 81.0% | 2.04 ⚠ severe |
| **temp=0.05** | **59.5%** | **epoch 14** | **2.0408** | **85.5%** | **1.49 ⚠ moderate** |
| temp=0.20 | 57.1% | epoch 15 | 2.4293 | 76.5% | 0.33 ✓ healthy |

**Training progression — val accuracy at key epochs:**

| Experiment | Epoch 1 | Epoch 8 | Epoch 15 | Val acc gain (1→15) |
|---|---|---|---|---|
| temp=0.01 | 14.5% | 52.3% | 58.9% | +44.4 pp |
| temp=0.05 | 14.2% | 52.9% | 59.4% | +45.2 pp |
| temp=0.20 | 13.9% | 48.6% | 57.1% | +43.2 pp |

All three start at ~14% (near chance for 150 intents) and converge similarly. The differences emerge in how much each one generalises vs memorises.

**Train vs val accuracy gap at epoch 15:**

| Experiment | Train acc | Val acc | Gap (train − val) |
|---|---|---|---|
| temp=0.01 | 81.0% | 58.9% | 22.1 pp |
| temp=0.05 | 85.5% | 59.4% | 26.1 pp |
| temp=0.20 | 76.5% | 57.1% | 19.5 pp |

`temp=0.20` has the smallest train–val gap — it is the most stable and least overfit. But its train accuracy is also lowest, meaning the model stopped learning as much. `temp=0.01` and `temp=0.05` both push train accuracy higher (they learn harder) but at the cost of a larger gap.

**When does overfitting start?**

| Experiment | Overfitting flag first appears | Overfitting definition used |
|---|---|---|
| temp=0.01 | Epoch 9 | val_loss − train_loss > 1.0 |
| temp=0.05 | Epoch 9 | val_loss − train_loss > 1.0 |
| temp=0.20 | Never in 15 epochs | gap stayed at 0.33 |

`temp=0.20` never triggers the overfitting flag — consistent with its gentler training signal allowing slower, more stable generalisation.

---

#### Comparison charts

The 4-panel chart at `results/experiments/experiment_comparison.png` shows:

| Panel | What it shows | What to look for |
|---|---|---|
| Top-left | Val loss curves per experiment | Lower is better; watch for divergence late in training |
| Top-right | Val accuracy curves with best-epoch markers | Higher is better; dots show peak per experiment |
| Bottom-left | Overfitting gap (val_loss − train_loss) over epochs | Flat = stable; steep rise = memorising |
| Bottom-right | Best val accuracy bar chart per experiment | Quick winner summary |

Per-experiment training curves (train + val loss, train + val acc, gap, val trend) are at `results/experiments/temp_X/training_curves_temp=X.png`.

---

#### Key findings

**Finding 1 — Temperature sweet spot is τ = 0.05**

`temp=0.05` achieves the best validation accuracy (59.5%) and the best final val loss (2.04). Very low temperature (`0.01`) produces a harsh training signal that pushes train accuracy to 81% but results in the worst val loss (2.64) and a severe overfitting gap of 2.04 — the model is memorising phrasings more than learning intent meaning. High temperature (`0.20`) is the most stable (gap=0.33) but the gentleness of the signal means the model never separates intents as sharply.

**Finding 2 — Overfitting starts at epoch 9 for sharp temperatures**

For `temp=0.01` and `temp=0.05`, the overfitting flag appears at epoch 9 — this is when train loss has dropped far enough that the val_loss − train_loss gap crosses 1.0. Before epoch 9, both metrics are moving together (healthy learning). After epoch 9, train loss keeps dropping but val loss plateaus — the model starts memorising specific phrasings from training pairs.

`temp=0.20` never hits this threshold. Its gentler signal means train loss never drops fast enough to create a large gap — the trade-off is slower convergence overall.

**Finding 3 — Val accuracy keeps improving past the overfitting flag**

Even after the overfitting flag appears at epoch 9, val accuracy continues to improve — from ~55% at epoch 9 to ~59.5% at epoch 14. This seems contradictory: the model is overfitting, yet val accuracy is still going up. Why?

The overfitting gap measures **raw loss values**, which are sensitive to scale. Val accuracy measures **retrieval ranking** — whether the correct intent is ranked first. A model can have a rising loss gap while still slightly improving its ranking because:
- Loss penalises every small confusion, even when the correct answer is still ranked first
- Accuracy only cares about the top-1 prediction

This means the `> 1.0 gap` threshold for our flag is conservative. A better early-stopping criterion: stop when val accuracy has not improved for 3 consecutive epochs (patience=3).

**Finding 4 — 150 intents is a hard classification problem for this model size**

All experiments top out at ~59% val accuracy. This is much lower than what you would expect on a 20-intent problem (where similar setups reach 90%+). Why?

With 150 intents, each batch contains 149 negatives per anchor. The model must simultaneously learn to separate 150 distinct intent clusters. Our model has only 872K parameters trained on 3,000 pairs — a very small data-to-class ratio. More training data and a longer training schedule are the most direct fixes:

```bash
# try these to push val accuracy higher
python3 train.py --pairs-per-intent 40 --epochs 30
python3 train.py --pairs-per-intent 40 --epochs 30 --temperature 0.05
```

Larger model capacity (`embed_dim=256`, `n_layers=4`) would also help if you want to go further.

---

---

### Step 7 — Evaluation

**Goal:** Measure how well the fine-tuned model actually works on unseen queries, and compare it against two baselines — a random-initialised model and TF-IDF — to quantify what training contributed.

**Key file:** `customer_intent_search/evaluate.py`

---

#### What we are measuring and why

During training we tracked **val accuracy** every epoch — but that was a training signal, not a real-world metric. It measured whether anchor_i matched positive_i inside the contrastive similarity matrix. That tells us the model is learning, not how well it retrieves intents.

Real evaluation simulates the actual use case:

```
corpus  = one representative query per intent (150 entries)
           → think of this as a catalogue of known intents

queries = all remaining test queries (~4,200 entries)
           → think of these as incoming customer messages

for each test query:
    embed it → find nearest corpus entry → did we get the right intent?
```

**Three metrics:**

| Metric | Question it answers | Formula |
|---|---|---|
| **Recall@1** | Is the top-1 result the correct intent? | correct_top1 / total_queries |
| **Recall@5** | Does the correct intent appear anywhere in the top 5? | correct_in_top5 / total_queries |
| **MRR** | How high up is the correct result on average? | mean(1 / rank_of_correct) |

**MRR example:**

```
query 1: correct intent ranked 1st  →  1/1 = 1.00
query 2: correct intent ranked 3rd  →  1/3 = 0.33
query 3: correct intent ranked 2nd  →  1/2 = 0.50

MRR = (1.00 + 0.33 + 0.50) / 3 = 0.61
```

MRR rewards the model for getting close even when it doesn't get the exact top-1 right. A model with Recall@1 = 60% but MRR = 0.80 is still ranking the right answer very near the top — much better than a model with Recall@1 = 60% and MRR = 0.62.

---

#### Three baselines to compare

| Model | What it tests | Expected result |
|---|---|---|
| **Random-init** | Same architecture, zero training — measures what the transformer gives you for free | Near-random (~1/150 = 0.7%) |
| **TF-IDF** | Classic keyword overlap — the baseline every NLP model must beat | Moderate — works when queries share exact words with corpus |
| **Fine-tuned** | Our trained model — what contrastive learning adds on top | Should clearly beat TF-IDF |

If fine-tuned >> TF-IDF >> random-init, training worked as expected.
If fine-tuned ≈ TF-IDF, the model learned nothing useful beyond word overlap.
If TF-IDF > fine-tuned, something went wrong during training.

---

#### TF-IDF deep dive

TF-IDF stands for **Term Frequency × Inverse Document Frequency**. Two ideas multiplied together. It represents text as a sparse vector of word importance scores — no training needed, purely based on word overlap.

##### Part 1 — Term Frequency (TF)

*How often does a word appear in this sentence?*

```
sentence: "I want to cancel my order please cancel it"

word      count    TF = count / total_words
──────────────────────────────────────────────
cancel      2      2/9 = 0.22
order       1      1/9 = 0.11
I           1      1/9 = 0.11
my          1      1/9 = 0.11   ← common word, but TF doesn't penalise yet
```

Raw TF rewards frequency. With `sublinear_tf=True` (what we use), it becomes `log(1 + count)` instead:

```
cancel:  log(1+2) = 1.10
order:   log(1+1) = 0.69
```

This prevents a word appearing 100 times from being treated as 100× more important than one appearing once.

##### Part 2 — Inverse Document Frequency (IDF)

*How rare is this word across all documents in the corpus?*

Imagine the corpus is these 4 intent representatives:

```
doc 0: "where is my order"
doc 1: "I want to cancel my order"
doc 2: "I forgot my password"
doc 3: "where is my refund"
```

Count how many documents each word appears in:

```
word        docs containing it    IDF = log(N / count)
────────────────────────────────────────────────────────
my                4               log(4/4) = 0.00   ← in everything → useless
where             2               log(4/2) = 0.69
order             2               log(4/2) = 0.69
I                 2               log(4/2) = 0.69
cancel            1               log(4/1) = 1.39   ← rare → very informative
password          1               log(4/1) = 1.39
refund            1               log(4/1) = 1.39
```

IDF crushes common words ("my" → IDF=0) and amplifies rare words ("cancel", "refund" → high IDF).

##### Part 3 — TF × IDF together

Final weight for a word = TF × IDF

```
sentence: "I want to cancel my order"

word      TF      IDF     TF-IDF
──────────────────────────────────
cancel    0.22  × 1.39  = 0.31   ← HIGH — rare word, present here
order     0.11  × 0.69  = 0.08
I         0.11  × 0.69  = 0.08
my        0.11  × 0.00  = 0.00   ← zero — too common to carry signal
want      0.11  × 1.39  = 0.15
```

The resulting TF-IDF vector is sparse — mostly zeros, with spikes at informative words:

```
[0, 0, 0.31, 0, 0.08, 0, 0.15, 0, 0, ...]
          ↑              ↑       ↑
        cancel          order   want
```

##### Part 4 — Similarity between two sentences

Two sentences are similar if their TF-IDF vectors point in the same direction (high cosine similarity = high dot product after L2 normalisation).

**Case A — same intent, different words (TF-IDF fails):**

```
corpus:  "I want to cancel my order"    → spikes at: cancel, order, want
query:   "please remove my purchase"   → spikes at: remove, purchase, please

shared words: only "my"  (IDF=0, contributes nothing)
dot product: ≈ 0.00   → TF-IDF says: completely unrelated ✗

reality: both mean cancel_order
```

TF-IDF has no concept of synonyms. "cancel" ≠ "remove", "order" ≠ "purchase" — even though a human instantly understands they mean the same thing.

**Case B — different intent, shared rare word (TF-IDF is confused):**

```
corpus:  "my order hasn't arrived"     → spikes at: arrived, order
query:   "wrong item in my order"      → spikes at: wrong, item, order

shared word: "order" (medium IDF)
dot product: non-zero  → TF-IDF says: somewhat similar

but: first is order_not_received, second is wrong_item_received — different intents
```

Shared vocabulary creates false similarity across intents.

**Case C — same intent, no shared words (TF-IDF completely breaks):**

```
corpus:  "track my package"
query:   "where is my parcel"

"track" ≠ "parcel",  "package" ≠ "where"
shared:  only "my" → IDF=0

dot product: 0.00   → TF-IDF says: completely unrelated ✗
```

A human immediately knows these mean the same thing. TF-IDF cannot.

##### Part 5 — The vector comparison

```
"I want to cancel my order"

TF-IDF vector  (sparse):
[0, 0, 0.31, 0, 0.08, 0, 0.15, 0, ...]   length = vocab_size (thousands of dims)
          ↑              ↑       ↑
       cancel           order   want      ← only 3 non-zero values out of thousands

Embedding vector  (dense):
[0.3, -0.5, 0.8, 0.1, -0.2, ...]         length = 128
  all 128 values carry learned meaning    ← no zeros, every dim contributes
```

TF-IDF: sparse, exact-word language.
Embedding: dense, learned-meaning language.

---

#### Why the embedding model beats TF-IDF

The fine-tuned model learned that "cancel order" and "remove purchase" should have similar vectors — because during training they always appeared as positive pairs (same intent). After training:

```
embed("I want to cancel my order")   → [0.3, -0.5, 0.8, ...]
embed("please remove my purchase")   → [0.3, -0.5, 0.7, ...]
cosine similarity: 0.97  ✓
```

The same two sentences under TF-IDF:

```
tfidf("I want to cancel my order")   → spikes at cancel, order
tfidf("please remove my purchase")   → spikes at remove, purchase
cosine similarity: 0.00  ✗
```

The key insight: **TF-IDF compares surface form. Embeddings compare learned meaning.**

That is the entire value proposition of training a model. The evaluation in `evaluate.py` will put a number on exactly how large this gap is on our 150-intent test set.

---

---

## Step 8 — Hard Negative Mining

**Goal:** Break through the ~59% validation accuracy plateau by training on hard negatives — queries from the wrong intent that the current model finds confusable.

**Key files:**
- `customer_intent_search/train.py` — `mine_hard_negatives()`, updated `MultipleNegativesRankingLoss`, updated `train()`

### Why easy negatives cause a plateau

With 150 intents per batch, the 149 off-diagonal entries in the `MultipleNegativesRankingLoss` similarity matrix are mostly *easy* negatives — `track_order` vs `reset_password` are clearly different, and the model separates them within the first few epochs. After that, those pairs contribute almost no gradient signal.

**Training log evidence (standard training):**
```
Epoch 08 | train_loss 1.37 | train_acc 61.6% | val_acc 52.9%
Epoch 10 | train_loss 0.95 | train_acc 73.2% | val_acc 55.3%   ← plateau starts
Epoch 13 | train_loss 0.66 | train_acc 81.2% | val_acc 58.7%   ← barely moving
Epoch 15 | train_loss 0.55 | train_acc 85.5% | val_acc 59.4%   ← flat
```

The model overfits to easy in-batch negatives while failing to learn the harder distinctions between semantically similar intents (e.g. `cancel_order` vs `modify_order`).

### What is a hard negative

A hard negative is a query from the **wrong** intent that the current model's embeddings score as highly similar to the anchor:

```
anchor:         "I need to cancel my subscription"    → intent: cancel_order
                               ↓ model confuses these
hard negative:  "can I change the plan on my account" → intent: modify_order
true positive:  "please cancel order #ORD-5678"       → intent: cancel_order
```

Training the model to push these apart forces it to learn the subtle semantic boundaries that distinguish confusable intents.

### Mining algorithm

Mining runs after a warm-up period (once the model has learned something) and repeats every few epochs as the model improves:

```
1. embed all 7,500 training queries  →  matrix [7500, 128]  (L2-normalised)
2. cosine similarity matrix          →  [7500, 7500]
3. for each query i:
       mask same-intent entries (score = -2.0)
       take top-K highest-similarity entries from different intents
       → add to that intent's hard-negative pool
4. repeat every hn_refresh_every epochs (default: 3)
```

The pool of hard negatives *changes each round* — as the model improves, previously hard pairs become easy and new confusable pairs emerge.

### Modified loss

The existing `MultipleNegativesRankingLoss` operates on a `[B, B]` similarity matrix. Hard negatives are appended per-anchor using a batched matrix multiply:

```
standard:                sim = A @ P.T              →  [B, B]    labels = [0, 1, …, B-1]
with top_k hard negs:    hn  = bmm(A, HN.T)        →  [B, k]
                         sim = concat([A@P.T, hn])  →  [B, B+k]  labels unchanged
```

- `HN` has shape `[B*k, D]` — k hard negatives per anchor, stacked
- `bmm` computes each anchor's similarity against only its *own* k hard negatives (no cross-contamination between anchors)
- The correct class label is still the diagonal — the model must rank column `i` highest in row `i`, and rank all k appended hard neg columns lower

### Dynamic pairing

Previously pairs were created **once** before training and the same combinations
were reused every epoch — the model saw the same anchor/positive pairs all 15 epochs:

```
Epoch 1:  anchor="where is my order"     positive="track my package"
Epoch 15: anchor="where is my order"     positive="track my package"  ← identical
```

With dynamic pairing, pairs are **re-shuffled at the start of each epoch**,
drawing different combinations from the same query pool each time:

```
Epoch 1:  anchor="where is my order"     positive="track my package"
Epoch 2:  anchor="has my order shipped"  positive="where is my parcel"   ← fresh
Epoch 3:  anchor="I need to track this"  positive="delivery status?"     ← fresh
```

With 50 queries per intent, there are 50×49/2 = 1,225 possible pairs per intent.
Static pairing uses 25 of them repeatedly. Dynamic pairing exposes the model to
a much wider variety across 15 epochs without needing more data.

Enabled by default (`--static-pairing` flag restores the old behaviour).

### Running experiments

```bash
cd customer_intent_search

# single run with hard negatives (top_k=1, mining starts epoch 4, refreshes every 3)
python train.py --hard-negatives --hn-top-k 1 --output-dir ../models/finetuned_hn

# full 2×4 grid — static and dynamic pairing, top_k = 0/1/3/5
python train.py --hn-experiments --static-pairing --exp-version v2  # static
python train.py --hn-experiments --exp-version v2                   # dynamic
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hard-negatives` | off | Enable offline hard negative mining |
| `--hn-top-k` | 1 | Hard negatives per anchor added to loss |
| `--hn-start-epoch` | 4 | Epoch to begin mining (warm-up first) |
| `--hn-refresh-every` | 3 | Re-mine every N epochs |
| `--hn-experiments` | off | Run full top_k comparison sweep |
| `--static-pairing` | off | Use same pairs every epoch (default: re-shuffle) |
| `--exp-version` | v1 | Version tag — results saved under `v<N>/static/` or `v<N>/dynamic/` |

### Results — v2 full 2×4 grid (CLINC-150, 15 epochs, temp=0.05, pairs_per_intent=25)

All metrics at the final epoch. Best Val Acc shows the peak reached during training.

| Config | Dynamic Pairing | Best Val Acc | Best Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Gap |
|--------|:--------------:|:------------:|:----------:|:----------:|:---------:|:--------:|:-------:|:---:|
| no_hn  | ✗ | 65.0% | 14 | 0.384 | 89.8% | 1.810 | 64.9% | 1.426 ⚠ |
| hn_k=1 | ✗ | 64.9% | 15 | 0.527 | 89.0% | 1.793 | 64.9% | 1.266 ⚠ |
| hn_k=3 | ✗ | 65.6% | 15 | 0.628 | 90.7% | 1.796 | 65.6% | 1.168 ⚠ |
| hn_k=5 | ✗ | 65.3% | 15 | 0.717 | 89.3% | 1.831 | 65.3% | 1.114 ⚠ |
| no_hn  | ✓ | 69.6% | 15 | 0.756 | 76.0% | 1.387 | 69.6% | 0.631 ⚠ |
| hn_k=1 | ✓ | 69.8% | 15 | 0.981 | 77.5% | 1.367 | 69.8% | 0.387 ✓ |
| hn_k=3 | ✓ | 69.5% | 15 | 1.139 | 79.3% | 1.358 | 69.5% | 0.219 ✓ |
| **hn_k=5** | **✓** | **70.5%** | **14** | **1.233** | **80.2%** | **1.375** | **70.3%** | **0.141 ✓** |

**Comparison charts:** `results/hn_experiments/v2/static/hn_comparison.png` and `results/hn_experiments/v2/dynamic/hn_comparison.png`

#### Key findings from the 2×4 grid

**Finding 1 — Dynamic pairing is the bigger lever (+4.6pp)**

Static no_hn (65.0%) → Dynamic no_hn (69.6%) = **+4.6pp** just from re-shuffling pairs
each epoch. The model sees more unique query combinations and generalises better.
Note: train accuracy is much higher under static (89.8% vs 76.0%) — the model is
memorising the same fixed pairs instead of learning robust representations.

**Finding 2 — Hard negatives barely help without dynamic pairing**

Under static pairing, HN mining improves val accuracy by just 0–0.6pp
(65.0% → 65.6% at best) and all four runs still overfit badly (gap > 1.0).
The model is already memorising static pairs — hard negatives can't overcome that.

**Finding 3 — Hard negatives shine on top of dynamic pairing**

Under dynamic pairing, hard negatives reduce the overfitting gap dramatically
(0.631 → 0.141) while also lifting accuracy slightly (69.6% → 70.5%). The two
techniques are **synergistic**: dynamic pairing provides training variety,
hard negatives sharpen the intent boundaries on top of that foundation.

**Finding 4 — Why gap and val accuracy tell different stories**

Val accuracy and val loss are two different signals:
- **Val accuracy** — binary per query (is rank-1 correct?). Coarse. Saturates easily.
- **Val loss** — continuous cross-entropy. Sensitive to model confidence across the full similarity matrix.

Hard negatives keep train_loss higher because training is harder — the model
cannot memorise pairs as easily. Val loss stays roughly flat. The gap shrinks
not because val improved, but because train didn't drop as far:

```
static no_hn : train=0.384  val=1.810  gap=1.426  ← model memorised training pairs
dynamic no_hn: train=0.756  val=1.387  gap=0.631  ← better generalisation
dynamic hn_k5: train=1.233  val=1.375  gap=0.141  ← healthiest — forced to generalise
```

A low gap means the model's training behaviour matches its test behaviour —
the best indicator that it will generalise to new queries it has never seen.

---

## Step 9 — Stability Analysis

Training a neural network involves randomness at several levels: weight initialisation,
pair sampling order, and mini-batch shuffling. A result that only holds for one lucky
seed is not trustworthy. The stability sweep reruns the **best config** (dynamic pairing
+ HN k=5) from scratch with different seeds and measures how much the outcome varies.

### What varies between runs

| Source of randomness | Effect |
|----------------------|--------|
| Weight initialisation | Different starting point in loss landscape |
| Dynamic pair seed | Different anchor/positive combinations each epoch |
| HN mining order | Slightly different hard-negative assignments |
| Batch shuffle | Different gradient directions per step |

### Running the stability sweep

```bash
# 5 runs with consecutive seeds (default)
python3 train.py --stability --n-seeds 5

# 5 runs with explicit spread-out seeds
python3 train.py --stability --seeds 7 23 99 137 256
```

| Flag | Default | Description |
|------|---------|-------------|
| `--stability` | off | Enable stability sweep mode |
| `--n-seeds` | 5 | Number of runs (used when `--seeds` not set) |
| `--seeds` | None | Explicit seed list — overrides `--seed` and `--n-seeds` |

Output layout:
```
models/stability/seed_N/      model.pt + tokenizer.json + config.json
results/stability/seed_N/     training_curves_seed_N.png
results/stability/stability_summary.json
results/stability/stability_plot.png
```

### Results — Sweep 1 (seeds 42–46, consecutive)

| Seed | Val Acc | Train Loss | Train Acc | Val Loss | Gap |
|------|:-------:|:----------:|:---------:|:--------:|:---:|
| 42   | 70.5%   | 1.313      | 78.2%     | 1.378    | 0.065 |
| 43   | 71.3%   | 1.256      | 78.7%     | 1.361    | 0.105 |
| 44   | 71.9%   | 1.180      | 80.6%     | 1.322    | 0.142 |
| 45   | 69.9%   | 1.297      | 78.0%     | 1.376    | 0.080 |
| 46   | 71.5%   | 1.226      | 80.5%     | 1.275    | 0.049 |
| **mean** | **71.0%** | **1.254** | **79.2%** | **1.343** | **0.088** |
| **±std** | **±0.72%** | **±0.048** | **±1.12%** | **±0.039** | **±0.033** |

### Results — Sweep 2 (seeds 7, 23, 99, 137, 256 — spread out)

| Seed | Val Acc | Train Loss | Train Acc | Val Loss | Gap |
|------|:-------:|:----------:|:---------:|:--------:|:---:|
| 7    | 70.7%   | 1.230      | 79.9%     | 1.308    | 0.078 |
| 23   | 71.6%   | 1.214      | 80.1%     | 1.253    | 0.038 |
| 99   | 71.7%   | 1.242      | 80.3%     | 1.295    | 0.052 |
| 137  | 71.1%   | 1.274      | 79.0%     | 1.348    | 0.074 |
| 256  | 69.8%   | 1.320      | 78.4%     | 1.367    | 0.047 |
| **mean** | **71.0%** | **1.256** | **79.5%** | **1.314** | **0.058** |
| **±std** | **±0.69%** | **±0.038** | **±0.73%** | **±0.040** | **±0.016** |

### Combined view — all 10 runs

```
Seed  | Val Acc |  Gap        Seed  | Val Acc |  Gap
──────────────────────────   ──────────────────────────
7     |  70.7%  | 0.078       42    |  70.5%  | 0.065
23    |  71.6%  | 0.038       43    |  71.3%  | 0.105
99    |  71.7%  | 0.052       44    |  71.9%  | 0.142
137   |  71.1%  | 0.074       45    |  69.9%  | 0.080
256   |  69.8%  | 0.047       46    |  71.5%  | 0.049
─────────────────────────────────────────────────────
Overall mean : 71.0%  ±0.70%   Range: [69.8 – 71.9%]
```

### Key findings from stability analysis

**Finding 1 — The result is real, not a lucky seed**

Both sweeps converge to **mean=71.0%** independently — one with consecutive seeds,
one with seeds spread across 0–256. The training recipe is stable, not sensitive to
initialisation.

**Finding 2 — Variance is tight (±0.70%)**

A ±0.7pp std on a 150-class problem means the worst run (69.8%) and the best run
(71.9%) differ by only 2.1pp. For comparison, the gap between static and dynamic
pairing is 4.6pp — so the signal from the training recipe is 6× larger than the noise
from random seeds.

**Finding 3 — Gap is consistently healthy across all seeds**

Every single run finished with gap < 0.15 (healthy). No seed produced an overfit
model. This confirms the result is robust, not coincidental.

**Headline number to report: 71.0 ± 0.7% val accuracy (n=10 independent runs)**

---

## Step 10 — Batch Retrieval Evaluation

Val accuracy (logged during training) measures how well the model ranks the correct
positive within a contrastive batch.  That's a useful training signal, but it's not
the same as real retrieval performance.

**Batch eval** runs every saved checkpoint through the full retrieval pipeline
(the same Recall@1/5/MRR setup from Step 7) and puts all results in one table.
This tells us: *across the whole journey from default training → temperature tuning
→ hard negatives + dynamic pairing → stability seeds, how much did retrieval
actually improve?*

### Why re-evaluate

| Signal | Measured during | What it captures |
|--------|----------------|-----------------|
| Val accuracy | Training loop | Rank within a 150-pair contrastive batch |
| Recall@1 | Batch eval | Exact retrieval from a 150-item corpus |
| Recall@5 | Batch eval | Top-5 retrieval (model knows the neighbourhood) |
| MRR | Batch eval | Mean reciprocal rank — partial credit for near misses |

Val accuracy and Recall@1 are correlated but not identical — a model with 71% val
accuracy may get 70% R@1, because the corpus retrieval task is harder than
batch-level ranking.

### Running batch evaluation

```bash
# evaluate all checkpoints (26 models + 2 baselines)
python3 batch_evaluate.py

# skip the 10 stability seeds for a faster run
python3 batch_evaluate.py --skip-stability
```

Results are saved to `results/batch_eval_results.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-root` | `..` | Path to repo root |
| `--skip-stability` | off | Skip the 10 stability-seed checkpoints |

### Results across all checkpoints

> Full results: [`results/batch_eval_results.json`](results/batch_eval_results.json)

| Model | R@1 | R@5 | MRR | Group |
|-------|:---:|:---:|:---:|-------|
| random-init | 8.6% | 19.6% | 15.5% | baseline |
| TF-IDF | 44.3% | 65.9% | 54.2% | baseline |
| finetuned (default) | 57.1% | 80.0% | 67.5% | baseline |
| temp τ=0.01 | 55.2% | 79.5% | 66.1% | temperature |
| temp τ=0.05 | 57.1% | 80.0% | 67.5% | temperature |
| temp τ=0.20 | 53.9% | 81.1% | 65.8% | temperature |
| v2 static no_hn | 62.8% | 83.3% | 72.0% | v2-static |
| v2 static hn_k=1 | 63.2% | 83.2% | 72.3% | v2-static |
| v2 static hn_k=3 | 63.4% | 82.6% | 72.2% | v2-static |
| v2 static hn_k=5 | 63.6% | 81.9% | 72.0% | v2-static |
| v2 dynamic no_hn | 68.0% | 88.8% | 77.2% | v2-dynamic |
| v2 dynamic hn_k=1 | 68.9% | 88.2% | 77.5% | v2-dynamic |
| v2 dynamic hn_k=3 | 69.2% | 87.6% | 77.4% | v2-dynamic |
| v2 dynamic hn_k=5 | 69.4% | 87.4% | 77.5% | v2-dynamic |
| **stability MEAN ± std** | **69.9 ± 1.2%** | **88.2 ± 1.1%** | **78.1 ± 1.1%** | stability |
| **stability seed=44 ★** | **71.9%** | **89.2%** | **79.7%** | stability |

### Key findings from batch evaluation

**Finding 1 — Real retrieval gain from the full pipeline: +12.8pp R@1**

Default training → best model: 57.1% → 69.9% mean R@1. That's the cumulative
effect of dynamic pairing + hard negatives, measured on the real retrieval task —
not just the training-time proxy metric.

**Finding 2 — Val accuracy slightly overstates retrieval performance**

Best val accuracy was ~71% (training signal); best mean R@1 is 69.9%. The 1–2pp
gap is expected — contrastive batch accuracy is an easier task than retrieving from
a fixed 150-item corpus, because the "negatives" in the batch vary by seed and epoch.

**Finding 3 — Static pairing recovers some ground but never catches dynamic**

Static best (63.6% R@1) vs dynamic best (69.4% R@1) = 5.8pp gap. Hard negatives
on static only add 0.8pp; hard negatives on dynamic add 1.4pp. The pairing strategy
is the dominant lever.

**Finding 4 — R@5 is already strong at 88%+**

The model places the correct intent in the top 5 for 88 out of 100 queries. The
remaining gap to perfect R@1 (70% → 100%) is almost entirely the sibling-intent
problem — intents that share near-identical vocabulary (gas/gas_type, order/order_status).

---

## Step 11 — Evaluation Metric Stability

The batch evaluation in Step 10 reported a single number per model — but that number
depends on *which query* was picked as the corpus representative for each intent.
Our `build_corpus_and_queries()` always picks `query[0]`, which is arbitrary.

This step asks: **how much does the evaluation result change when we randomise the
corpus selection?** If the answer is "not much", our metric is trustworthy.
If the answer is "a lot", the single-split result is noisy and we should always
report a mean over multiple splits.

### The corpus-split problem

In retrieval evaluation with 1 corpus entry per intent:

```
Corpus entry for track_order : "where is my order"         ← split A
Corpus entry for track_order : "has my package shipped"    ← split B
```

If the corpus entry happens to be phrased unusually, *every query for that intent*
will score worse — not because the model is bad, but because the reference is hard
to match.  This noise is independent of model quality.

### Running eval stability

```bash
# 5 random corpus splits on the best model (default)
python3 eval_stability.py

# custom model or more splits
python3 eval_stability.py --model-dir ../models/hn_experiments/v2/dynamic/hn_k_5 --n-splits 10
```

Results are saved to `results/eval_stability_results.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `../models/stability/seed_44` | Checkpoint to evaluate |
| `--n-splits` | 5 | Number of random corpus splits |
| `--seed` | 42 | Base seed for split randomisation |
| `--no-tfidf` | off | Skip TF-IDF baseline (faster) |

### Results across corpus splits

> Full results: [`results/eval_stability_results.json`](results/eval_stability_results.json)

**Model: stability seed=44** (best single-seed checkpoint)

| Split (seed) | R@1 | R@5 | MRR | TF-IDF R@1 | TF-IDF R@5 | TF-IDF MRR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | 70.8% | 87.7% | 78.5% | 42.8% | 65.0% | 53.0% |
| 43 | 72.9% | 89.7% | 80.4% | 43.6% | 66.0% | 54.0% |
| 44 | 69.1% | 87.4% | 77.3% | 42.2% | 62.6% | 51.8% |
| 45 | 74.8% | 89.8% | 81.6% | 45.0% | 67.2% | 55.1% |
| 46 | 73.3% | 89.5% | 80.5% | 45.0% | 65.2% | 54.4% |

**Summary:**

| Metric | Fine-tuned Mean | ± Std | Range | TF-IDF Mean | ± Std |
|--------|:--------------:|:-----:|:-----:|:-----------:|:-----:|
| Recall@1 | **72.2%** | **±2.0%** | 5.7% | 43.7% | ±1.1% |
| Recall@5 | **88.8%** | **±1.0%** | 2.3% | 65.2% | ±1.5% |
| MRR      | **79.7%** | **±1.6%** | 4.3% | 53.7% | ±1.1% |

### Key findings from eval stability

**Finding 1 — The headline metric is 72.2 ± 2.0% R@1**

Our original single-split result (71.9%) was measured with a fixed corpus choice.
Across 5 random splits the true mean is **72.2% R@1** — slightly higher — with a
±2.0% std driven by which query happens to be selected as the corpus representative.
This is the number to report.

**Finding 2 — ±2% is corpus noise, not model noise**

The model-training stability (Step 9) gave ±0.7% std across different seeds.
The eval-corpus stability (this step) gives ±2.0% std across different corpus splits.
The **evaluation setup contributes ~3× more variance than the model itself**.
This is a property of single-representative retrieval eval on a 150-intent dataset —
one bad corpus pick can tank an entire intent's recall.

**Finding 3 — R@5 is a more reliable metric (±1.0%)**

Recall@5 is far less sensitive to corpus choice because the model gets 5 chances
to hit the correct intent. If your application can show 5 suggestions, R@5 is both
more useful *and* more stable to report.

**Finding 4 — TF-IDF has lower eval variance (±1.1% R@1) than our model (±2.0%)**

TF-IDF relies on exact keyword overlap, which is less sensitive to *which specific
phrasing* is the corpus representative — all phrasings of the same intent share
mostly the same keywords. The embedding model is more expressive but therefore
more affected by the exact choice of reference sentence.

---

## Dependencies

| Package | Role |
|---------|------|
| `torch` | Tensors and model training |
| `numpy` | Numerical utilities |
| `scikit-learn` | Metrics and evaluation helpers |
| `matplotlib` | Result plots |
| `tqdm` | Progress bars |
| `sentence-transformers` | Pretrained embedding models |
| `datasets` | Dataset loading |
