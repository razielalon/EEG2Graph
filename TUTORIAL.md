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
┌─────────▼──────────────────┐
│  Bridge (trainable)        │   Linear 840→1024 + LayerNorm + Dropout
│  ~860K params              │   Projects EEG into REBEL's embedding space
└─────────┬──────────────────┘
          │  inputs_embeds
┌─────────▼─────────────────────────────────────┐
│  Babelscape/rebel-large  (low LR or frozen)   │
│  BART-large (12 enc + 12 dec layers, 1024d)   │
│  already fine-tuned on REBEL relation-        │
│  extraction corpus — decoder already knows    │
│  the <triplet>/<subj>/<obj> grammar           │
│  ~406M params                                 │
└─────────┬─────────────────────────────────────┘
          │
┌─────────▼─────────┐
│  Linearized graph │   "<triplet> Barack Obama <subj> Hawaii
│  as token IDs     │    <obj> place of birth"
└───────────────────┘
```

**Why this split?**

- **REBEL** (`Babelscape/rebel-large`) is a BART-large checkpoint **already
  fine-tuned** on a large relation-extraction corpus. It's pretrained to emit
  exactly the `<triplet> subj <subj> obj <obj> relation` format we want, so
  the decoder doesn't have to learn the output grammar from scratch. Given
  how small ZuCo is (~10k word-level samples), this format-fluency prior is
  what makes learning tractable.
- The **Bridge** is the only layer that sees EEG — it learns the mapping from
  neural features (840-dim) into REBEL's token-embedding space (1024-dim). It
  carries the bulk of the *learning signal*; hence a 10× higher LR than REBEL.
- REBEL's encoder token-embedding lookup is bypassed on the encoder side by
  feeding `inputs_embeds=bridge(eeg)`.
- Optionally, REBEL can be **frozen entirely** (`--freeze_bart`). In that
  regime the Bridge is the only thing that trains and the optimizer only
  allocates state for Bridge params. This is the recommended mode for CPU
  smoke-tests and very-small-data runs.

---

## 3. Output format — REBEL's native linearization

A set of triplets is serialized as a single string with **three** structural
markers, in **Subject–Object–Relation order**:

```
<triplet> Barack Obama <subj> Hawaii <obj> place of birth
```

Multiple triplets are concatenated; each starts with `<triplet>`. The whole
string is wrapped by BART's `<s>` / `</s>`:

```
<s> <triplet> subj1 <subj> obj1 <obj> rel1
    <triplet> subj2 <subj> obj2 <obj> rel2 </s>
```

The three markers (`<triplet>`, `<subj>`, `<obj>`) are **already in REBEL's
vocabulary** as native single tokens — no `add_tokens` call, no
`resize_token_embeddings` call. Importantly there is **no `<rel>` marker**:
the relation runs from the end of `<obj> ...` up to the next `<triplet>` (or
`</s>`).

**Why this exact format?** REBEL was pretrained with it. Reusing it means the
pretrained decoder weights are directly useful — the model doesn't have to
relearn where punctuation goes, how entities look, or what order the fields
come in. Reordering to `<subj> <rel> <obj>` or adding a `<rel>` token would
invalidate that prior.

---

## 4. Repo layout

```
EEG2Graph/
├── preprocessing/          # ZuCo → word-level EEG arrays (Phase A.1)
│   ├── data_from_gcp.py    # download raw ZuCo 1.0 or 2.0 from GCS
│   ├── inspect_zuco.py     # HDF5 debug helpers
│   └── preprocess_zuco.py  # ZuCo 1.0 (v5) + 2.0 (v7.3) → processed_zuco*/
│
├── generateTriplets/       # sentence → triplets (Phase A.2)
│   └── generate_triplets.py  # runs REBEL on each unique sentence
│
├── processed_zuco/         # ZuCo 2.0 outputs of Phase A
├── processed_zuco1/        # ZuCo 1.0 outputs of Phase A
│   ├── {train,val,test}_eeg.npy     # object arrays of (n_words, 840)
│   ├── {train,val,test}_meta.json   # text, words, subject_id, has_fixation
│   ├── {train,val,test}_labels.json
│   ├── sentence_triplets.json       # {sentence: {"triplets": [...]}}
│   ├── norm_stats.json              # per-subject normalization
│   └── dataset_info.json            # feature_dim=840, etc.
│
├── model/                  # Phase B: training + inference (focus here)
│   ├── vocabulary.py       # REBEL tokenizer + REBEL linearize/delinearize
│   ├── eeg_graph_dataset.py# PyTorch Dataset + collate_fn + build_dataloaders
│   ├── eeg_graph_model.py  # EEGBartModel: Bridge + REBEL (BART-large)
│   ├── train.py            # training loop, loss, metrics, --freeze_bart
│   ├── inference.py        # load checkpoint + run predictions
│   └── test_model.py       # 16 unit/integration tests
│
├── tests/
│   └── processed_data_test.py  # sanity checks on processed_zuco*
│
├── checkpoints/            # model outputs go here
├── checkpoints_smoke/      # outputs of the CPU sanity command
├── requirements.txt
└── TUTORIAL.md             # you are here
```

---

## 5. The data side (already done, here for context)

### 5.1 Downloading raw data — `preprocessing/data_from_gcp.py`

The GCS bucket `zuco_dataset_bucket` is organized as `ZuCo1/` and `ZuCo2/`.
Pick which one you want:

```bash
python preprocessing/data_from_gcp.py --dataset zuco2  # default -> ./processed_zuco
python preprocessing/data_from_gcp.py --dataset zuco1  # default -> ./processed_zuco1
```

You can override with `--output_dir`. The script uses `google.cloud.storage`,
so GCP credentials need to be set up beforehand (e.g. `gcloud auth
application-default login`).

### 5.2 ZuCo preprocessing — `preprocessing/preprocess_zuco.py`

Turns the raw ZuCo `.mat` files into aligned, word-level arrays. The script
supports **both ZuCo versions** and auto-detects the file format:

- **ZuCo 2.0** ships as **MATLAB v7.3** (HDF5-backed) — read with `h5py` and
  the `h5_to_string` / `h5_to_array` helpers.
- **ZuCo 1.0** ships as **MATLAB v5/v7** (classic binary) — read with
  `scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)`, which
  exposes nested MATLAB structs as Python attribute-access objects.

`_is_matlab_v73(filepath)` sniffs the file header; the dispatcher calls the
appropriate reader (`_process_subject_file_v5` or the v7.3 path) accordingly.

Dataset-specific config lives in `DATASET_CONFIGS` at the top of the file:

- **ZuCo 1.0:** 12 subjects (`ZAB`, `ZDM`, …); tasks `task2-NR` and
  `task3-TSR` (task1-SR is commented out).
- **ZuCo 2.0:** 18 subjects (`YAC`, `YAG`, …); tasks `task1-NR` and `task2-TSR`.

For each (subject × task × sentence × word) the script:

- Extracts the EEG recorded during that word's **first gaze duration** (GD
  fixation window), concatenating 8 frequency bands × 105 channels = **840-dim
  feature vector** per word.
- Zero-pads words that were skipped during reading and tracks them in a
  boolean `has_fixation` mask.
- Per-subject z-score normalization (computed from fixated words only).
- Split 80/10/10 **by sentence text** (not by subject) so the same sentence
  read by multiple subjects stays in one split.

Usage:

```bash
python preprocessing/preprocess_zuco.py \
    --dataset zuco2 --data_dir /path/to/raw_zuco2 --output_dir ./processed_zuco

python preprocessing/preprocess_zuco.py \
    --dataset zuco1 --data_dir /path/to/raw_zuco1 --output_dir ./processed_zuco1
```

### 5.3 Triplet generation — `generateTriplets/generate_triplets.py`

For each **unique sentence text** (not per-subject), runs **REBEL itself** to
extract triplets, parses the REBEL output back into dicts, and writes
`sentence_triplets.json` keyed by sentence text:

```json
{
  "Barack Obama was born in Hawaii .": {
    "triplets": [
      {"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}
    ]
  }
}
```

The dataset loader later joins this to each per-sample `{split}_meta.json`
entry by matching `meta[i]["text"]` to the keys here.

Since REBEL is also what the downstream model decodes with, the training
targets are in exactly the format REBEL was pretrained to emit — which is the
whole point of picking REBEL as the backbone.

---

## 6. The model side — module deep-dive

### 6.1 `model/vocabulary.py` — tokenizer helpers

A thin wrapper around REBEL's tokenizer. Key facts:

- `DEFAULT_MODEL_NAME = "Babelscape/rebel-large"`.
- `STRUCT_TOKENS = ["<triplet>", "<subj>", "<obj>"]` — all three are **native
  tokens in REBEL's vocab**, so we do not call `add_tokens` and do not
  resize the model's embedding table.
- **There is no `<rel>` marker** — relation occupies the trailing slot of
  each triplet, terminated by the next `<triplet>` or `</s>`.

Five functions:

| Function | What it does |
|---|---|
| `build_tokenizer(model_name)` | `AutoTokenizer.from_pretrained(model_name)`. No token-adding dance. |
| `linearize_triplets(triplets, tokenizer)` | Builds `"<triplet> {subj} <subj> {obj} <obj> {rel}"` per triplet, joins them, tokenizes with `add_special_tokens=True` so `<s>` / `</s>` wrap the sequence. Returns a list of ints. |
| `delinearize(token_ids, tokenizer)` | Decodes to a string, strips `<s>`/`</s>`/`<pad>`, splits on `<triplet>`, then on `<subj>`/`<obj>` to recover the three fields. The relation is whatever remains after `<obj>` up to the next `<triplet>`. Silently drops malformed blocks. |
| `save_tokenizer(tok, dir)` | `tok.save_pretrained(dir)`. |
| `load_tokenizer(dir)` | `AutoTokenizer.from_pretrained(dir)`. |

**Why a string-based `delinearize`?** BART tokenizes subwords
(`"Hawaii"` → `["H", "awa", "ii"]`). Parsing structure on the **decoded
string** is cleaner than reconstructing words from subword IDs.

### 6.2 `model/eeg_graph_dataset.py` — Dataset + collation

#### `EEGGraphDataset`

At construction time it:

1. Loads `{split}_eeg.npy` (object array) and `{split}_meta.json` (list of dicts).
2. Loads `sentence_triplets.json` and indexes it `text → triplets`. Supports
   both the dict-keyed format (`{sentence: {"triplets": [...]}}`) and a
   list-of-objects fallback (`[{"text": ..., "triplets": [...]}, ...]`).
3. For each row `i`:
   - Look up `triplets` for `meta[i]["text"]` (empty list if no match).
   - Call `linearize_triplets(triplets, tokenizer)` to get token IDs, truncate
     to `max_tgt_len`, and ensure the last ID is `eos_token_id` even after
     truncation.
   - Slice EEG to the first `min(n_words, max_src_len)` rows.
   - Build a `"has_fixation"` boolean mask (carried through but not
     currently consumed by the model — reserved for future attention-bias
     experiments).
4. Prints a match-rate summary: `N samples, K with triplets (X.X%)`.

Returns `{eeg, target_ids, has_fixation, n_src, n_tgt, meta}` per `__getitem__`.

#### `collate_fn(batch, pad_id)`

Pads a list of samples into a batch and **performs the teacher-forcing shift**:

- `src`, `src_mask`, `src_fixation` — padded to `max_src` over the batch.
- `tgt = target_ids[:-1]` — decoder input.
- `tgt_labels = target_ids[1:]` — what the decoder should predict next.
- `tgt_mask` — `True` for non-pad positions.

Classic seq2seq shift: with target `[<s> A B C </s>]`, we feed `[<s> A B C]`
into the decoder and ask it to predict `[A B C </s>]`.

#### `build_dataloaders(...)`

Creates a fresh tokenizer (or reuses one passed in) and three `DataLoader`s.
Key extras:

- Uses `functools.partial(collate_fn, pad_id=tokenizer.pad_token_id)` so the
  collate closure picks up REBEL's actual pad ID rather than a hardcoded
  constant.
- Supports a `limits={"train": N, ...}` dict — when set, wraps the dataset in
  `torch.utils.data.Subset` for fast CPU sanity runs.
- Default `bart_name="Babelscape/rebel-large"`.

Returns `(loaders, tokenizer)`.

### 6.3 `model/eeg_graph_model.py` — `EEGBartModel`

The whole model is ~130 lines:

```python
class EEGBartModel(nn.Module):
    def __init__(self, tokenizer, eeg_dim=840,
                 bart_name="Babelscape/rebel-large", dropout=0.3):
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        # No-op: REBEL's vocab already contains <triplet>/<subj>/<obj>.
        if self.bart.get_input_embeddings().num_embeddings != len(tokenizer):
            self.bart.resize_token_embeddings(len(tokenizer))

        d_model = self.bart.config.d_model  # 1024 for BART-large
        self.bridge = nn.Sequential(
            nn.Linear(eeg_dim, d_model),       # 840 -> 1024
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
```

Notes on the bridge:

- **No `GELU`** — the nonlinearity was removed because REBEL's encoder
  expects a Transformer-style normalized activation; adding GELU in front of
  it was doubling up on a nonlinearity and hurt early training.
- Output dim is **1024** (BART-large's `d_model`), not 768. This is tested
  explicitly by `test_model_uses_rebel_dim`.

Four methods:

- **`forward(src, src_mask, tgt) -> logits`** — teacher-forced forward.
  Bypasses BART's encoder token-embedding lookup by passing
  `inputs_embeds=bridge(src)`. `attention_mask=src_mask.long()` marks real
  source positions. `decoder_input_ids=tgt` is the shifted-right target.
- **`generate(src, src_mask, max_len, num_beams) -> ids`** — wraps
  `model.generate()`. `num_beams=1` is greedy; `>1` activates beam search
  with `early_stopping=True`. Same API covers both greedy and beam paths.
- **`freeze_bart()`** — sets `requires_grad=False` on every BART/REBEL param.
  After calling, only the Bridge trains.
- **`param_groups(bridge_lr, bart_lr, weight_decay)`** — returns optimizer
  groups so AdamW can apply different LRs. Bridge gets `3e-4`; BART gets
  `3e-5`. **If BART is frozen, the BART group is omitted** — otherwise
  AdamW would still allocate momentum/variance state for parameters it
  never updates. `param_groups` therefore returns either one or two groups.

**What you'll *not* find here:** no custom `<bos>/<eos>/<pad>` constants, no
label-smoothing inside the forward pass, no beam search logic, no
weight-tying code. All of that is either built into HF BART or handled in the
loss.

### 6.4 `model/train.py` — the training driver

High-level flow of `main()`:

```
1. Build dataloaders + tokenizer            (build_dataloaders, REBEL tok)
2. Save tokenizer to output_dir/tokenizer/  (for inference later)
3. Instantiate EEGBartModel                 (downloads REBEL on first run)
4. Optionally freeze REBEL                  (--freeze_bart)
5. Build AdamW with 1 or 2 param groups      (bridge_lr; bart_lr if unfrozen)
6. Build CosineAnnealingLR scheduler
7. Build LabelSmoothedCE loss
8. Loop over epochs:
     a. train_epoch  (teacher forcing)
     b. evaluate     (loss + greedy generate + triplet-F1)
     c. Save best_model.pt if val_F1 improved
     d. Save checkpoint_ep{N}.pt every --save_every epochs
9. After last epoch: reload best_model.pt, run test-set eval with beam search
10. Write history.json, test_results.json, test_metrics.json, best_examples.json
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
2. Generate predictions: `model.generate(src, src_mask, max_len, num_beams)`
   → `delinearize(ids, tokenizer)` → list of triplet dicts.

Then `compute_triplet_f1` does **exact set matching** on lowercased, stripped
`(subj, rel, obj)` tuples to produce precision/recall/F1.

#### `compute_triplet_f1`

Micro-averaged: sums correct/predicted/gold across the full eval set, then
computes P/R/F1 from those totals. Predicted-gold overlap is via Python set
intersection. Strict — no partial credit for near-matches.

### 6.5 `model/inference.py` — loading and predicting

```
1. load_tokenizer(args.tokenizer_dir)      ← the dir train.py saved
2. load_model(args.checkpoint, tokenizer)  ← instantiates EEGBartModel,
                                             then load_state_dict
3. For each batch of EEG samples:
     predict_batch → model.generate → delinearize → per-sample dict
4. Dump to predictions.json
```

`load_model` reads `bart_name` from the checkpoint (falling back to
`args.bart_name`, default `"Babelscape/rebel-large"`) so the checkpoint
always knows which HF model to instantiate.

`predict_batch` pads a list of variable-length EEG arrays into a single
`(B, max_src, 840)` tensor + attention mask, runs `model.generate(...,
num_beams=args.beam_size)`, and parses each row back into triplets.

### 6.6 `model/test_model.py` — 16 tests

Run with `python model/test_model.py`. The tokenizer and model are loaded
once at module scope (`_TOKENIZER`, `_MODEL`) so REBEL's ~1.6 GB checkpoint
is downloaded/loaded only once per test session.

Categories:

- **Tokenizer (5)** — structural tokens tokenize to one ID each,
  `<rel>` is *not* a single token (guard against re-introducing it),
  linearize↔delinearize round-trips (incl. checking S-before-O-before-R
  order in the decoded string), empty triplet list → `[<s>, </s>]`,
  tokenizer save/load.
- **Dataset (4)** — dataset loads both triplets formats (dict / list),
  collate shapes correct, `build_dataloaders` works against real
  `processed_zuco/` data (dir is overridable via `EEG_DATA_DIR` env var).
- **Model (5)** — forward pass shape is `(B, T, vocab)`, greedy decoding is
  deterministic, beam decoding runs without crashing, `param_groups`
  returns two groups with the expected LRs and sizes, **Bridge output dim
  matches REBEL's `d_model=1024`**, and **no embedding resize is needed**
  (`embedding_table_size == len(tokenizer)`).
- **Integration (1)** — full pipeline on real data: dataloader → forward →
  generate → delinearize.

Expected runtime: a couple of minutes on CPU (dominated by downloading
REBEL's weights the first time; afterwards HF caches them in
`~/.cache/huggingface`).

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
   ~10k-row dataset is small). Returns 3 `DataLoader`s + the tokenizer.
2. **`save_tokenizer(tokenizer, output_dir/tokenizer)`** writes the tokenizer
   alongside checkpoints so inference later doesn't need to re-configure it.
3. **`EEGBartModel(tokenizer, ...)`** downloads `Babelscape/rebel-large`
   (~1.6 GB, cached after first run) and builds the Bridge projection
   (Linear 840 → 1024 + LN + Dropout).
4. If `--freeze_bart`, all REBEL params get `requires_grad=False` and
   `param_groups` returns only the Bridge group.
5. **`AdamW` with 1 or 2 param groups** — bridge_lr on Bridge,
   bart_lr on REBEL (if trainable).
6. **Per epoch:**
   - `train_epoch` iterates every batch. For each batch:
     - `src` = `(B, S, 840)`, `src_mask` = `(B, S)`, `tgt` = `(B, T-1)`.
     - Bridge projects EEG → `(B, S, 1024)`.
     - REBEL encoder → `(B, S, 1024)` memory; decoder cross-attends and
       produces logits `(B, T-1, vocab_size)`.
     - `LabelSmoothedCE(logits, tgt_labels)` → scalar loss.
     - Backprop, grad clip, optimizer step.
   - After epoch: scheduler step (CosineAnnealingLR from initial LRs down to
     0.01× `bart_lr` over `args.epochs`).
   - `evaluate` runs the same forward pass on val, plus greedy `generate`
     and `delinearize` → triplet-F1.
   - If F1 improved → `torch.save({"model_state_dict": ..., "args": ...,
     "bart_name": ..., "struct_tokens": ...}, "best_model.pt")` plus a
     5-example gold/pred dump to `best_examples.json`.
   - Every `--save_every` epochs → `checkpoint_ep{N}.pt`.
7. **After last epoch:** reload `best_model.pt`, evaluate on the test split
   with `num_beams=args.beam_size` (default 4). Write:
   - `test_metrics.json` — micro-P/R/F1 + counts.
   - `test_results.json` — gold + pred triplet lists per sample.
   - `history.json` — per-epoch train/val loss, F1, LRs, wall time.

A typical healthy training curve, with REBEL unfrozen at `3e-5`:

- Epoch 1: train_loss drops rapidly (REBEL already knows the grammar;
  most of the initial loss is Bridge randomness). val_F1 is usually nonzero
  from epoch 1 — another benefit of starting from REBEL rather than
  vanilla BART.
- Epochs 2–10: val_F1 climbs as the Bridge converges; precision typically
  leads recall because the decoder is conservative when Bridge output is
  noisy.
- Epochs 20+: slow grind on recall — harder entities and relations.

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

1. `load_tokenizer` reads the saved tokenizer directory — this restores
   REBEL's tokenizer exactly (vocab + merges + special tokens).
2. `load_model` reads the checkpoint. It extracts `bart_name` from the
   checkpoint (or `args.bart_name` as fallback, defaulting to
   `Babelscape/rebel-large`), constructs a fresh `EEGBartModel`, and loads
   the state dict.
3. The loop pads each chunk of EEG samples to a common length, runs
   `model.generate(..., num_beams=beam_size)`, and calls `delinearize` on
   each row.
4. `predictions.json` contains a list of `{text, subject_id,
   predicted_triplets}` dicts.

**Greedy vs beam:** `--beam_size 1` is greedy. `--beam_size 4` runs
beam search with early stopping — slower but typically better F1 because
early autoregressive mistakes compound.

---

## 9. Checkpoint contents

`best_model.pt` is a dict:

```python
{
    "epoch": int,
    "model_state_dict": {...},        # Bridge + REBEL weights
    "optimizer_state_dict": {...},    # for resuming training (not used yet)
    "val_f1": float,                  # best-so-far metric
    "args": {...},                    # everything argparse saw (eeg_dim, etc.)
    "bart_name": str,                 # "Babelscape/rebel-large"
    "struct_tokens": [...],           # for reference: ["<triplet>","<subj>","<obj>"]
}
```

The **tokenizer is NOT in the checkpoint** — it lives in a parallel directory
(`output_dir/tokenizer/`) because HF's `tokenizer.save_pretrained()` writes
several files (`vocab.json`, `merges.txt`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json`). Keep them together when
copying checkpoints around.

---

## 10. Key design decisions — why it looks this way

- **REBEL instead of vanilla BART:** `Babelscape/rebel-large` is BART-large
  already fine-tuned for end-to-end relation extraction. It knows the
  `<triplet> subj <subj> obj <obj> rel` output grammar out of the box. ZuCo
  is tiny (~10k word-level samples) — starting from a format-aware decoder
  means the gradient goes into the EEG→text Bridge rather than into
  teaching BART what triplets look like.
- **S-O-R order with no `<rel>` marker:** this is REBEL's native format.
  Deviating from it (adding `<rel>`, reordering to S-R-O) would invalidate
  the pretraining prior.
- **Bridge = Linear + LayerNorm + Dropout (no GELU):** a nonlinearity in
  front of REBEL's first Transformer block was redundant and slowed
  learning. LayerNorm alone keeps activations in the range REBEL's encoder
  expects.
- **Bridge output dim = 1024:** this is BART-large's `d_model`, not
  BART-base's 768. It's asserted in tests so a future model swap doesn't
  silently break shape compatibility.
- **No `resize_token_embeddings`:** REBEL's structural tokens are already in
  vocab. The constructor only calls `resize_token_embeddings` if the
  tokenizer size drifts away from the embedding table — a defensive no-op
  in the normal case.
- **Differential LRs:** Bridge at `3e-4`, REBEL at `3e-5`. Without this,
  REBEL forgets its pretraining rapidly. With it, REBEL makes small
  corrections while Bridge learns from scratch.
- **`--freeze_bart` option:** useful for CPU/smoke runs where training
  REBEL itself isn't realistic, and as a strong baseline — if the frozen
  setup already hits a given F1, any unfrozen improvement has to beat that.
  Frozen params are also skipped from the optimizer param groups so AdamW
  doesn't allocate unused state.
- **Micro-averaged exact-match F1:** harsh but unambiguous. If future
  experiments need partial credit, add a soft-matching metric alongside
  rather than replacing this one.
- **`best_val_f1 = -1.0` initial value:** guarantees at least one checkpoint
  is written even if F1 never moves off 0.0 (can happen on very small
  subsets or early in training). Needed for test-eval to find a
  checkpoint to load.
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

To run tests against the ZuCo 1.0 data:
```bash
EEG_DATA_DIR=processed_zuco1 python test_model.py
```

### Quick CPU sanity run (< 1 minute, REBEL frozen)
```bash
cd model
python train.py \
    --processed_dir ../processed_zuco \
    --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints_smoke \
    --epochs 2 --batch_size 2 \
    --max_src_len 16 --max_tgt_len 24 \
    --limit_train 8 --limit_val 4 --limit_test 4 \
    --freeze_bart --beam_size 1
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

Swap to ZuCo 1.0 by pointing at `../processed_zuco1/` and its
`sentence_triplets.json`.

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
   inter-subject variability. With both ZuCo 1.0 (12 subjects) and ZuCo 2.0
   (18 subjects) now available, this becomes particularly interesting.
3. **Joint training on ZuCo 1.0 + 2.0** — concatenate the two processed
   splits. Same preprocessing, same 840-dim features; the main question is
   per-subject normalization across the merged set.
4. **EEG dropout at the channel or band level** — regularization tailored
   to the input modality.
5. **Soft F1 metric** — ROUGE over the linearized string, or fuzzy entity
   matching, to capture partial correctness.
6. **Curriculum on triplet count** — start training on sentences with 1
   triplet, gradually add multi-triplet sentences as loss plateaus.

Each of these is local to one module (usually Bridge inputs or the metric)
and doesn't require changing REBEL.
