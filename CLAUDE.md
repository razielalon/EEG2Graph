# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

EEG2Graph translates word-level EEG recordings (ZuCo 1.0 / 2.0) into linearized knowledge-graph triplets. The model is a thin **Bridge** (Linear 840 → 1024 + LayerNorm + Dropout) feeding `inputs_embeds` into **`Babelscape/rebel-large`** (BART-large already fine-tuned for relation extraction). REBEL is what makes this tractable on a ~10k-sample dataset: its decoder already emits the `<triplet> subj <subj> obj <obj> rel` grammar, so the gradient goes into the EEG→text bridge rather than into teaching BART what triplets look like.

For an end-to-end walkthrough of architecture, data flow, and design rationale, read `TUTORIAL.md` — it is the authoritative reference and is kept in sync with the code.

## Repository layout

```
preprocessing/      Phase A.1: raw ZuCo .mat → word-level (n_words, 840) arrays
generateTriplets/   Phase A.2: sentence text → triplets via REBEL itself
model/              Phase B: dataset, model, training, inference, tests
tests/              processed_data_test.py — sanity checks for processed_zuco*/
processed_zuco{,1,2}/   Output of Phase A (gitignored)
checkpoints*/       Training outputs (best_model.pt + tokenizer/ subdir)
train_{smoke,full}.sbatch   Slurm jobs for BGU CIS cluster
```

## Common commands

All commands assume the venv is active (`source venv/bin/activate`).

### Tests
```bash
cd model && python test_model.py                                # full suite (16 tests, ~2 min, downloads REBEL once)
EEG_DATA_DIR=processed_zuco1 python model/test_model.py         # run dataset tests against ZuCo 1.0
python tests/processed_data_test.py --data_dir ./processed_zuco # validate preprocessed splits + leakage check
```
There is no `pytest` runner — `test_model.py` uses `unittest` and is invoked as a script. The tokenizer/model are loaded at module scope (`_TOKENIZER`, `_MODEL`) so REBEL's ~1.6 GB checkpoint downloads only once per session.

### CPU smoke training (< 1 min)
```bash
cd model
python train.py \
    --processed_dir ../processed_zuco --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints_smoke --epochs 2 --batch_size 2 \
    --max_src_len 16 --max_tgt_len 24 \
    --limit_train 8 --limit_val 4 --limit_test 4 \
    --freeze_bart --beam_size 1
```

### Real training (GPU)
```bash
cd model
python train.py \
    --processed_dir ../processed_zuco2 --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir ../checkpoints --epochs 80 --batch_size 16 \
    --bridge_lr 3e-4 --bart_lr 3e-5
```
On the BGU CIS Slurm cluster, submit `train_smoke.sbatch` or `train_full.sbatch` instead (do not run `cd` in the sbatch — they already `cd "$HOME/EEG2Graph/model"`).

### Inference
```bash
cd model
python inference.py \
    --checkpoint ../checkpoints/best_model.pt \
    --tokenizer_dir ../checkpoints/tokenizer \
    --processed_dir ../processed_zuco --split test --beam_size 4 \
    --output ../predictions.json
```

### Data
```bash
python preprocessing/data_from_gcp.py --dataset zuco2          # requires gcloud auth, writes ./processed_zuco
python preprocessing/preprocess_zuco.py --dataset zuco2 --data_dir <raw> --output_dir ./processed_zuco
python preprocessing/inspect_zuco.py <file.mat> --detailed
```

## Architecture invariants (don't break these)

These are enforced by tests in `model/test_model.py` and reflect deliberate design choices documented in `TUTORIAL.md` §10.

- **REBEL's native linearization is `<triplet> subj <subj> obj <obj> rel`** — S → O → R, with **no `<rel>` marker**. The relation runs from the end of `<obj> ...` to the next `<triplet>` or `</s>`. Reordering or adding `<rel>` invalidates the pretraining prior.
- **`<triplet>`, `<subj>`, `<obj>` are already in REBEL's vocab as single tokens** — never call `tokenizer.add_tokens(...)` for them, and don't `resize_token_embeddings` (the model constructor has a defensive resize only if sizes drift).
- **Bridge output dim = 1024** (BART-large's `d_model`), not 768. Asserted by `test_model_uses_rebel_dim`.
- **Bridge has no GELU** — `Linear → LayerNorm → Dropout` only. Adding a nonlinearity in front of REBEL's first Transformer block hurt early training.
- **Differential LRs with phased training:**
  - **Phase 1 (warmup, BART frozen):** single bridge group (`param_groups`) + `CosineAnnealingLR` over `--warmup_epochs`. When `--freeze_bart` is set or BART is frozen, `param_groups` must return **one** group, not two — AdamW would otherwise allocate momentum/variance state for frozen params.
  - **Phase 2 (post-unfreeze):** layer-wise LR decay (`param_groups_llrd`) returns ~26 groups: bridge + 12 enc + 12 dec + 1 embeddings. Top encoder/decoder layers get full `--bart_lr`; layer `k` below the top gets `bart_lr * llrd_gamma**k`. Embeddings (tied to `lm_head`) get the smallest LR. `lm_head.weight is model.shared.weight` — dedup via `id(p)` so the same Parameter never appears in two groups (AdamW would crash).
  - **BART LR warmup at unfreeze:** `LambdaLR` linearly ramps BART groups from `1/W` to `1.0` over `--bart_warmup_epochs` (default 3) before cosine decay. Without this ramp, BART jumps from LR=0 (frozen) to full `--bart_lr` in one step and overfits within a few epochs on the small ZuCo dataset.
  - **Separate weight decay:** bridge uses `--weight_decay` (0.01); BART uses `--bart_weight_decay` (0.05) to regularize fine-tuning.
  - **Patience suppression:** `patience_counter` does not tick during the BART warmup window — val loss is expected to fluctuate while BART thaws.
- **BART dropout overrides:** `--bart_dropout` and `--bart_attention_dropout` (defaults 0.2 / 0.2) raise REBEL's native 0.1 dropout to regularize fine-tuning. Passed through `BartForConditionalGeneration.from_pretrained(..., dropout=..., attention_dropout=..., activation_dropout=...)` so the config and constructed modules stay in sync.
- **Splits are grouped by sentence text**, not by subject. The same sentence read by multiple subjects must stay in one split.
- **Tokenizer lives in `output_dir/tokenizer/`, not inside the `.pt` checkpoint.** Keep them together when copying checkpoints.

## Data conventions

- **Per-sample EEG**: object array of shape `(n_words, 840)` = 105 channels × 8 frequency bands, extracted from the first gaze-duration (GD) fixation window. Skipped words are zero-padded and tracked in a `has_fixation` boolean mask (carried through the dataset but not yet consumed by the model — reserved for attention-bias experiments).
- **Normalization** is per-subject z-score, computed from fixated words only.
- **`sentence_triplets.json`** is keyed by exact sentence text; the dataset joins to each `{split}_meta.json` entry via `meta[i]["text"]`. Both the dict-keyed and list-of-objects formats are supported by `EEGGraphDataset`.
- **ZuCo 1.0 ships as MATLAB v5** (use `scipy.io.loadmat`), **ZuCo 2.0 ships as MATLAB v7.3** (HDF5, use `h5py`). `_is_matlab_v73()` in `preprocess_zuco.py` sniffs the header and dispatches.

## Linearize / delinearize round-trip

`model/vocabulary.py` parses on the **decoded string**, not on token IDs (BART subword-tokenizes entity names, so reconstructing from IDs is brittle). `delinearize` silently drops malformed blocks — if you add validation logic, make sure round-trip tests still pass.

## Working with the model

- `EEGBartModel.forward(src, src_mask, tgt)` is teacher-forced; `src_mask` must be `.long()` for HF's attention mask. The encoder embeddings are bypassed via `inputs_embeds=bridge(src)`.
- `collate_fn` performs the teacher-forcing shift: `tgt = target_ids[:-1]`, `tgt_labels = target_ids[1:]`. Don't shift again elsewhere.
- `model.generate(..., num_beams=1)` is greedy; `>1` activates beam search with `early_stopping=True`. Same API.
- `train.py` initializes `best_val_f1 = -1.0` so at least one checkpoint always saves (val_F1 can stay at 0.0 on small subsets).

## Tooling

- No linter, formatter, or type-checker is configured. Match existing style (the codebase is plain-PyTorch + argparse, no Hydra/Lightning).
- No `pytest`, no CI. Run `model/test_model.py` manually before non-trivial model/dataset changes.
- The Slurm sbatch files reference a `conda` env named `eeg2graph` and email `zivcz@post.bgu.ac.il` — adjust if submitting under a different account.
