# EEG2Graph — Code Tutorial

This document walks through the full codebase: what each module does, how data
flows through it, and why the architecture is shaped the way it is. It assumes
you're comfortable with PyTorch and seq2seq transformers; it does **not**
re-explain ML basics.

---

## 1. The research goal

We want to reconstruct a **Knowledge Graph** — a set of
`(subject, relation, object)` triplets — from **EEG recorded while a person
reads a sentence**.

```
Input:  EEG recorded during reading of "Barack Obama was born in Hawaii."
Output: [{"subject": "Barack Obama",
          "relation": "place of birth",
          "object":   "Hawaii"}]
```

This is a **seq2seq translation problem**: one source sequence (word-level EEG
vectors) is translated into another sequence (a linearized triplet string).

---

## 2. High-level architecture

```
┌───────────────────┐
│  Word-level EEG   │   (B, S, 840)   ← 105 channels × 8 freq bands
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│  Bridge (trainable)│   Linear 840→768 + LayerNorm + GELU + Dropout
│  ~650K params     │   Projects EEG into BART's embedding space
└─────────┬─────────┘
          │  inputs_embeds
┌─────────▼─────────────────────────────┐
│  BART-base (pretrained, low LR)       │
│  Encoder  (6 layers, 768d)            │
│  Decoder  (6 layers, cross-attention) │
│  lm_head (tied to token embeddings)   │
│  ~139M params                         │
└─────────┬─────────────────────────────┘
          │
┌─────────▼─────────┐
│  Linearized graph │   "<triplet> Barack Obama <subj> place of birth
│  as token IDs     │    <rel> Hawaii <obj>"
└───────────────────┘
```

**Why this split?**

- **BART-base** was pretrained on 160 GB of text — it already knows English
  syntax, common entities, relation phrasing. Fine-tuning at a very small LR
  prevents catastrophic forgetting of that prior.
- The **Bridge** is the only layer that sees EEG — it learns the mapping from
  neural features (840-dim) into BART's token-embedding space (768-dim). It
  has the bulk of the *learning signal* per epoch; hence a 10× higher LR.
- We keep BART's encoder and decoder intact and bypass token embedding lookup
  on the encoder side by feeding `inputs_embeds=bridge(eeg)`.

---

## 3. Output format — REBEL-style linearization

A set of triplets is serialized as a single string with four structural
markers:

```
<triplet> Barack Obama <subj> place of birth <rel> Hawaii <obj>
```

Multiple triplets are concatenated; each starts with `<triplet>`. The whole
string is wrapped by BART's `<s>` / `</s>`:

```
<s> <triplet> subj1 <subj> rel1 <rel> obj1 <obj>
    <triplet> subj2 <subj> rel2 <rel> obj2 <obj> </s>
```

The four markers (`<triplet>`, `<subj>`, `<rel>`, `<obj>`) are **added as
special tokens** to BART's vocabulary, so they tokenize to one ID each instead
of being split into subwords. `resize_token_embeddings(len(tokenizer))` extends
BART's embedding table by those 4 rows (initialized from the mean/covariance
of the existing embeddings).

**Why this format?** It's structured enough to parse back reliably
(`delinearize`) but close enough to natural language that BART's pretrained
prior still helps.

---

## 4. Repo layout

```
EEG2Graph/
├── preprocessing/          # ZuCo → word-level EEG arrays (Phase A.1)
│   ├── data_from_gcp.py    # download raw ZuCo from GCS
│   ├── inspect_zuco.py     # HDF5 debug helpers
│   └── preprocess_zuco.py  # produces processed_zuco/
│
├── generateTriplets/       # sentence → triplets (Phase A.2)
│   └── generate_triplets.py
│
├── processed_zuco/         # outputs of Phase A (committed)
│   ├── {train,val,test}_eeg.npy     # object arrays of (n_words, 840)
│   ├── {train,val,test}_meta.json   # text, words, subject_id, has_fixation
│   ├── {train,val,test}_labels.json
│   ├── sentence_triplets.json       # {sentence: {"triplets": [...]}}
│   ├── norm_stats.json              # per-channel normalization
│   └── dataset_info.json            # feature_dim=840, etc.
│
├── model/                  # Phase B: training + inference (the focus here)
│   ├── vocabulary.py       # BART tokenizer + 4 structural markers
│   ├── eeg_graph_dataset.py# PyTorch Dataset + collate_fn + build_dataloaders
│   ├── eeg_graph_model.py  # EEGBartModel: Bridge + BART
│   ├── train.py            # training loop, loss, metrics
│   ├── inference.py        # load checkpoint + run predictions
│   └── test_model.py       # 14 unit/integration tests
│
├── checkpoints/            # model outputs go here
├── requirements.txt
└── TUTORIAL.md             # you are here
```

---

## 5. The data side (already done, here for context)

### 5.1 ZuCo preprocessing — `preprocessing/preprocess_zuco.py`

Turns the raw ZuCo 2.0 HDF5 files into aligned, word-level arrays:

- For each (subject × task × sentence × word), extract the EEG recorded
  during that word's **first gaze duration** (GD fixation window).
- Concatenate 105 channels × 8 frequency bands = **840-dim feature vector**
  per word.
- Split 80/10/10 into train/val/test by sentence (not by subject — same
  sentence read by all 18 subjects stays in the same split).
- Write:
  - `{split}_eeg.npy` — a NumPy object array; entry `i` is shape `(n_words_i, 840)`.
  - `{split}_meta.json` — parallel list of dicts with `text`, `words`,
    `subject_id`, `task`, `has_fixation`, `n_words`.

### 5.2 Triplet generation — `generateTriplets/generate_triplets.py`

For each **unique sentence text** (not per-subject), generate the ground-truth
triplets. Output is `processed_zuco/sentence_triplets.json` keyed by sentence
text:

```json
{
  "Barack Obama was born in Hawaii .": {
    "triplets": [
      {"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}
    ]
  },
  ...
}
```

The dataset loader later joins this to the per-sample `{split}_meta.json`
by matching `meta[i]["text"]` against the keys here.

---

## 6. The model side — module deep-dive

### 6.1 `model/vocabulary.py` — tokenizer helpers

Despite the filename, this module **no longer owns a Vocabulary class** — it's
a thin wrapper around a HuggingFace BART tokenizer. Five functions:

| Function | What it does |
|---|---|
| `build_tokenizer(model_name)` | Loads `AutoTokenizer.from_pretrained(model_name)`, adds the 4 structural markers via `add_tokens(..., special_tokens=True)`. |
| `linearize_triplets(triplets, tokenizer)` | Builds the `"<triplet> ... <subj> ... <rel> ... <obj>"` string and tokenizes it with `add_special_tokens=True` so `<s>` / `</s>` are auto-prepended/appended. Returns a list of ints. |
| `delinearize(token_ids, tokenizer)` | Decodes IDs to a string, strips `<s>`/`</s>`/`<pad>`, splits on `<triplet>`, then on `<subj>`/`<rel>`/`<obj>`. Silently drops malformed blocks — robust to partial decoder output. |
| `save_tokenizer(tok, dir)` | `tok.save_pretrained(dir)`. |
| `load_tokenizer(dir)` | `AutoTokenizer.from_pretrained(dir)`. |

Module constant: `STRUCT_TOKENS = ["<triplet>", "<subj>", "<rel>", "<obj>"]`.

**Why a string-based `delinearize` instead of a per-token state machine?**
BART tokenizes subwords (e.g., `"Hawaii"` → `["H", "awa", "ii"]`). Parsing
structure on the **decoded string** is cleaner than reconstructing words from
token-ID subword pieces.

### 6.2 `model/eeg_graph_dataset.py` — Dataset + collation

#### `EEGGraphDataset`

At construction time it:

1. Loads `{split}_eeg.npy` (object array) and `{split}_meta.json` (list of dicts).
2. Loads `sentence_triplets.json` and indexes it `text → triplets`.
3. For each row `i`:
   - Look up `triplets` for `meta[i]["text"]` (empty list if no match).
   - Call `linearize_triplets(triplets, tokenizer)` to get token IDs, truncate
     to `max_tgt_len`, and ensure the last ID is `eos_token_id` even after
     truncation.
   - Slice EEG to the first `min(n_words, max_src_len)` rows.
   - Build a `"has_fixation"` boolean mask of the same length (currently
     carried through but not consumed by the model yet — reserved for future
     attention-bias experiments).

Returns `{eeg, target_ids, has_fixation, n_src, n_tgt, meta}` per `__getitem__`.

#### `collate_fn(batch, pad_id)`

Pads a list of samples into a batch and **performs the teacher-forcing shift**:

- `src`, `src_mask`, `src_fixation` — padded to `max_src` over the batch.
- `tgt = target_ids[:-1]` — decoder input.
- `tgt_labels = target_ids[1:]` — what the decoder should predict next.
- `tgt_mask` — `True` for non-pad positions (used for eventual masking but
  the loss already ignores `pad_id`).

This is the classic seq2seq shift: with a target sequence
`[<s> A B C </s>]`, we feed `[<s> A B C]` into the decoder and ask it to
predict `[A B C </s>]`.

#### `build_dataloaders(...)`

Creates a fresh tokenizer (or reuses one passed in) and three `DataLoader`s.
Key extras:

- Uses `functools.partial(collate_fn, pad_id=tokenizer.pad_token_id)` so the
  collate closure picks up BART's actual pad ID (`1`) rather than a
  hardcoded constant.
- Supports a `limits={"train": N, ...}` dict — when set, wraps the dataset in
  `torch.utils.data.Subset` for fast CPU sanity runs.

Returns `(loaders, tokenizer)`.

### 6.3 `model/eeg_graph_model.py` — `EEGBartModel`

The whole model is ~90 lines:

```python
class EEGBartModel(nn.Module):
    def __init__(self, tokenizer, eeg_dim=840, bart_name="facebook/bart-base", dropout=0.3):
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        self.bart.resize_token_embeddings(len(tokenizer))  # extend for 4 markers
        self.bridge = nn.Sequential(
            nn.Linear(eeg_dim, self.bart.config.d_model),   # 840 -> 768
            nn.LayerNorm(self.bart.config.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
```

Three important methods:

- **`forward(src, src_mask, tgt) -> logits`** — teacher-forced forward. It
  bypasses BART's encoder token-embedding lookup by passing
  `inputs_embeds=bridge(src)`, giving BART something that lives in its own
  embedding space. `attention_mask=src_mask.long()` tells BART which source
  positions are real. `decoder_input_ids=tgt` is the shifted-right target.
- **`generate(src, src_mask, max_len, num_beams) -> ids`** — wraps HF's
  `model.generate()` for autoregressive decoding. `num_beams=1` is greedy;
  `>1` activates beam search. Same API subsumes both previous `generate()` and
  `beam_search()` paths.
- **`param_groups(bridge_lr, bart_lr, weight_decay)`** — returns two
  optimizer groups so AdamW can apply different LRs. Bridge gets `3e-4`
  (learning from scratch), BART gets `3e-5` (fine-tuning).

**What you'll *not* find here:** no custom `<bos>/<eos>/<pad>` constants, no
label-smoothing inside the forward pass, no beam search logic, no
weight-tying code. All of that is either built into HF BART or handled in the
loss.

### 6.4 `model/train.py` — the training driver

High-level flow of `main()`:

```
1. Build dataloaders + tokenizer            (build_dataloaders)
2. Save tokenizer to output_dir/tokenizer/  (for inference later)
3. Instantiate EEGBartModel                 (downloads BART-base on first run)
4. Build AdamW with two param groups         (bridge_lr, bart_lr)
5. Build CosineAnnealingLR scheduler
6. Build LabelSmoothedCE loss
7. Loop over epochs:
     a. train_epoch  (teacher forcing)
     b. evaluate     (loss + greedy generate + triplet-F1)
     c. Save best_model.pt if val_F1 improved
     d. Save checkpoint_ep{N}.pt every --save_every epochs
8. After last epoch: reload best_model.pt, run test-set eval with beam search
9. Write history.json, test_results.json, test_metrics.json
```

#### `LabelSmoothedCE`

Cross-entropy with label smoothing, ignoring pad tokens:

```python
loss = (1 - smoothing) * nll_loss(log_softmax(logits), targets)
     +      smoothing  * (-log_softmax(logits).mean())
```

Positions where `targets == pad_id` are filtered out before the loss.

#### `train_epoch` (teacher forcing)

For each batch: `logits = model(src, src_mask, tgt)` → `loss(logits,
tgt_labels)` → `loss.backward()` → gradient clipping (1.0) → optimizer step.
After the whole epoch, the LR scheduler steps once.

#### `evaluate`

Two things per batch:

1. Compute teacher-forced validation loss (same as training but `torch.no_grad`).
2. Generate predictions: `model.generate(src, src_mask, max_len, num_beams=1)`
   → `delinearize(ids, tokenizer)` → list of triplet dicts.

Then `compute_triplet_f1` does **exact set matching** on lowercased, stripped
`(subj, rel, obj)` tuples to produce precision/recall/F1.

#### `compute_triplet_f1`

Micro-averaged: sums correct/predicted/gold across the full eval set, then
computes P/R/F1 from those totals. Predicted-gold overlap is via Python set
intersection. This is strict — partial credit for near-matches is not given.

### 6.5 `model/inference.py` — loading and predicting

```
1. load_tokenizer(args.tokenizer_dir)      ← the dir train.py saved
2. load_model(args.checkpoint, tokenizer)  ← instantiates EEGBartModel,
                                             then load_state_dict
3. For each batch of EEG samples:
     predict_batch → model.generate → delinearize → per-sample dict
4. Dump to predictions.json
```

`predict_batch` pads a list of variable-length EEG arrays into a single
`(B, max_src, 840)` tensor + attention mask, runs `model.generate(...,
num_beams=args.beam_size)`, and parses each row back into triplets.

### 6.6 `model/test_model.py` — 14 tests

Run with `python model/test_model.py`. Three categories:

- **Tokenizer (5)** — structural tokens tokenize to one ID, linearize↔delinearize
  round-trips, empty triplet list → `[<s>, </s>]`, tokenizer save/load.
- **Dataset (4)** — dataset loads both triplets formats (dict / list),
  collate shapes correct, `build_dataloaders` works against real
  `processed_zuco/` data.
- **Model (4)** — forward pass shape is `(B, T, vocab)`, greedy decoding is
  deterministic, beam decoding runs without crashing, `param_groups` returns
  two groups with the expected LRs and sizes.
- **Integration (1)** — full pipeline on real data: dataloader → forward →
  generate → delinearize.

Expected runtime: ~2 minutes on CPU (dominated by downloading BART weights
the first time; afterwards HF caches them in `~/.cache/huggingface`).

---

## 7. Training flow — step by step

Take the command:

```bash
python train.py \
    --processed_dir ../processed_zuco \
    --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints \
    --epochs 80 --batch_size 16
```

What actually happens:

1. **`build_dataloaders`** opens `{train,val,test}_eeg.npy` and
   `sentence_triplets.json`. For each split it builds an `EEGGraphDataset`
   which pre-tokenizes every sample's target and caches it in RAM (the whole
   10k-row dataset is small). Returns 3 `DataLoader`s + the tokenizer.
2. **`save_tokenizer(tokenizer, output_dir/tokenizer)`** writes the tokenizer
   alongside checkpoints so inference later doesn't need to re-add special
   tokens.
3. **`EEGBartModel(tokenizer, ...)`** downloads `facebook/bart-base`
   (~500 MB, cached after first run), extends its embedding table by 4
   rows, and builds the Bridge projection.
4. **`AdamW` with two param groups** — bridge_lr on Bridge (~650K params),
   bart_lr on BART (~139M params).
5. **Per epoch:**
   - `train_epoch` iterates all ~640 batches (10216/16). For each batch:
     - `src` = `(16, S, 840)`, `src_mask` = `(16, S)`, `tgt` = `(16, T-1)`.
     - Bridge projects EEG → `(16, S, 768)`.
     - BART encoder → `(16, S, 768)` memory; decoder cross-attends and
       produces logits `(16, T-1, 50269)`.
     - `LabelSmoothedCE(logits, tgt_labels)` → scalar loss.
     - Backprop, grad clip, optimizer step.
   - After epoch: scheduler step (CosineAnnealingLR from initial LRs down to
     0.01× over `args.epochs`).
   - `evaluate` runs the same forward pass on val, plus greedy `generate` and
     `delinearize` → triplet-F1.
   - If F1 improved → `torch.save({"model_state_dict": ..., "args": ...,
     "bart_name": ..., "struct_tokens": ...}, "best_model.pt")`.
   - Every `--save_every` epochs → `checkpoint_ep{N}.pt`.
6. **After last epoch:** reload `best_model.pt`, evaluate on the test split
   with `num_beams=args.beam_size` (default 4). Write:
   - `test_metrics.json` — micro-P/R/F1 + counts.
   - `test_results.json` — gold + pred triplet lists per sample.
   - `history.json` — per-epoch train/val loss, F1, LRs, wall time.
   - `best_examples.json` — 5 gold/pred examples from the best val epoch.

A typical healthy training curve:

- Epoch 1: train_loss ≈ 9, val_F1 ≈ 0.0–0.05 (model is still learning to
  emit the structural markers).
- Epochs 2–10: val_F1 jumps as the model picks up the format; precision
  leads recall because it's easier to say nothing than to say something
  wrong.
- Epochs 20+: slow grind — recall improves as BART gets better at guessing
  specific entity names from the EEG signal.

---

## 8. Inference flow

```bash
python inference.py \
    --checkpoint ../checkpoints/best_model.pt \
    --tokenizer_dir ../checkpoints/tokenizer \
    --processed_dir ../processed_zuco \
    --split test --beam_size 4 \
    --output predictions.json
```

Step by step:

1. `load_tokenizer` reads the saved tokenizer directory — this restores the 4
   special marker tokens and BART's vocabulary identically.
2. `load_model` reads the checkpoint. It extracts `bart_name` from the
   checkpoint (or `args.bart_name` as fallback), constructs a fresh
   `EEGBartModel`, and loads the state dict.
3. The loop pads each chunk of EEG samples to a common length, runs
   `model.generate(..., num_beams=beam_size)`, and calls `delinearize` on
   each row.
4. `predictions.json` contains a list of `{text, subject_id,
   predicted_triplets}` dicts.

**Greedy vs beam:** `--beam_size 1` is greedy. `--beam_size 4` runs
length-normalized beam search — slower but typically better F1 because the
decoder is autoregressive and early mistakes compound.

---

## 9. Checkpoint contents

`best_model.pt` is a dict:

```python
{
    "epoch": int,
    "model_state_dict": {...},        # Bridge + BART weights
    "optimizer_state_dict": {...},    # for resuming training (not used yet)
    "val_f1": float,                  # best-so-far metric
    "args": {...},                    # everything argparse saw (eeg_dim, etc.)
    "bart_name": str,                 # "facebook/bart-base"
    "struct_tokens": [...],           # for reference
}
```

The **tokenizer is NOT in the checkpoint** — it lives in a parallel directory
(`output_dir/tokenizer/`) because HF's `tokenizer.save_pretrained()` writes
several files (vocab.json, merges.txt, tokenizer_config.json,
special_tokens_map.json, added_tokens.json). Keep them together when copying
checkpoints around.

---

## 10. Key design decisions — why it looks this way

- **BART instead of a custom transformer:** 10k sentences is not enough to
  teach a from-scratch transformer English. The pretrained prior is
  essential.
- **Bridge as a Linear+LN+GELU+Dropout stack:** simple, low-parameter,
  and matches the convention used in most EEG-to-text bridge work. LayerNorm
  right after the projection helps stabilize BART's encoder which expects
  input activations in a specific range.
- **Differential LRs:** Bridge at `3e-4`, BART at `3e-5`. Without this, BART
  forgets its pretraining rapidly. With it, BART makes small corrections
  while Bridge learns from scratch.
- **REBEL-style markers instead of JSON or `|`-separated format:** markers
  are single tokens, easy to parse back, and they give the decoder a clear
  grammar to follow without needing to emit quote/comma syntax correctly.
- **Micro-averaged exact-match F1:** harsh but unambiguous. If future
  experiments need partial credit, add a soft-matching metric alongside
  rather than replacing this one.
- **`best_val_f1 = -1.0` initial value:** guarantees at least one checkpoint
  is written even if F1 never moves off 0.0 (can happen on very small subsets
  or early in training). Needed for test-eval to find a checkpoint to load.
- **CPU `--limit_*` flags:** let you exercise the full training codepath
  (save/load/evaluate) on a handful of samples. Catches integration bugs
  without the wall-clock cost of a real epoch.

---

## 11. How to run things — quick reference

### Unit tests
```bash
source venv/bin/activate
cd model && python test_model.py
```

### Quick CPU sanity run (< 1 minute)
```bash
cd model
python train.py \
    --processed_dir ../processed_zuco \
    --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints_smoke \
    --epochs 2 --batch_size 2 \
    --max_src_len 16 --max_tgt_len 24 \
    --limit_train 8 --limit_val 4 --limit_test 4 \
    --beam_size 1
```

### Real training (requires GPU)
```bash
cd model
python train.py \
    --processed_dir ../processed_zuco \
    --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints \
    --epochs 80 --batch_size 16 \
    --bridge_lr 3e-4 --bart_lr 3e-5
```

### Inference
```bash
cd model
python inference.py \
    --checkpoint ../checkpoints/best_model.pt \
    --tokenizer_dir ../checkpoints/tokenizer \
    --processed_dir ../processed_zuco \
    --split test --beam_size 4 \
    --output ../predictions.json
```

---

## 12. Extending the model

A few directions that would be low-risk to try, in rough order of expected
impact:

1. **Use `has_fixation` as an encoder attention bias** — currently unused but
   carried through the dataset. Words without a recorded fixation are
   noise; masking or down-weighting them should help.
2. **Subject-level conditioning** — add a learned embedding per
   `subject_id` and sum it into every Bridge output. Accounts for
   inter-subject variability.
3. **EEG dropout at the channel or band level** — regularization tailored to
   the input modality.
4. **Soft F1 metric** — ROUGE over the linearized string, or fuzzy entity
   matching, to capture partial correctness.
5. **Curriculum on triplet count** — start training on sentences with 1
   triplet, gradually add multi-triplet sentences as loss plateaus.

Each of these is local to one module (usually Bridge inputs or the metric)
and doesn't require changing the BART side.
