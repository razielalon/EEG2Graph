# E5 — Frozen-BART component ablation grid (RQ4): build & run plan

**Status: PLAN ONLY — nothing built yet.** This documents what to build and run.

**Question (RQ4):** under frozen BART, which architectural component actually
moves the needle? Attribute the frozen-baseline F1 to subject-embedding, the
temporal bridge, and the alignment mode.

## The grid (2×2×2 = 8 cells)

| Axis | OFF | ON |
|---|---|---|
| subject embedding | `--n_subject_buckets 0` | `--n_subject_buckets 64` |
| temporal bridge | `--bridge_transformer_layers 0` | `--bridge_transformer_layers 2` (`--bridge_nhead 8`) |
| alignment mode | `--align_mode pooled` | `--align_mode perword` |

All cells: `--freeze_bart`, `--bridge_layers 2`, `--align_weight 1.0`,
`--align_temp 0.07`, `--seed 42`, identical epoch/patience budget. Only the three
axes vary, so any F1 difference is attributable.

| # | subject | temporal | align | status |
|---|---|---|---|---|
| 1 | off | off | pooled | **new** (clean baseline: bridge → frozen REBEL only) |
| 2 | on  | off | pooled | new |
| 3 | off | on  | pooled | new |
| 4 | on  | on  | pooled | **exists** = 0.57% test F1 (`checkpoints_subject_temporal`, epoch 26) |
| 5 | off | off | perword | new |
| 6 | on  | off | perword | new |
| 7 | off | on  | perword | new |
| 8 | on  | on  | perword | new |

→ **7 new runs.** Cell 4 is reused (already trained), but re-run it on the unified
branch too if we want all 8 under byte-identical code/seed (recommended for a clean
table; cheap insurance against "the existing cell used different code").

## Prerequisite — the unified branch (the real work)

No branch has all three axes (see table in session notes):
- `exp/subject-temporal` → subject + temporal flags, **no** `align_mode`
- `exp/align-perword` → `align_mode {pooled,perword}`, **no** subject/temporal
- `exp/tier1-conclusions` → `--train_sentence_frac`, `--exclude_subjects` (for E6/E8 later)

**Build `exp/tier2-grid` by merging, base = `exp/subject-temporal`:**
1. Branch from `exp/subject-temporal` (it has the richer model: `subject_emb`,
   `bridge_transformer`, `_bridge_forward`, `--seed`).
2. Port from `exp/align-perword/model/train.py`:
   - `contrastive_alignment_perword` (the dense per-position InfoNCE; pooled
     plateaued ~2.7 across all 4 prior runs) alongside the existing pooled
     `contrastive_alignment_loss`.
   - the `align_mode` argparse flag (default `pooled`) + the `align_fn` selection
     in `train_epoch` (`align_fn = perword if align_mode=="perword" else pooled`).
   - thread `align_mode` through `train_epoch(...)` / the call site.
3. Port from `exp/tier1-conclusions/model/{train.py,eeg_graph_dataset.py}`:
   `--train_sentence_frac` and `--exclude_subjects` (so E6/E8 reuse this branch).
4. Confirm both alignment fns get per-position `h_eeg` (needs
   `forward(..., return_encoder_states=True)`) and the frozen `encode_text`
   teacher — both already exist on `subject-temporal`; the per-word fn just stops
   pooling before InfoNCE.
5. **Validate before any launch:** `cd model && python test_model.py` (16 tests),
   then a CPU smoke run per the CLAUDE.md smoke command with `--freeze_bart
   --align_mode perword --n_subject_buckets 4 --bridge_transformer_layers 1` to
   exercise every new code path on 8 samples.

### Merge risk notes
- `subject_idx` must reach the loss path: `collate_fn(..., n_subject_buckets=)`
  already emits it on `subject-temporal`; per-word alignment doesn't touch it, so
  no conflict.
- per-word InfoNCE masks same-text off-diagonal pairs (same sentence, two
  subjects, not a negative) — verify that masking survives the merge; it's the
  documented invariant in CLAUDE.md.
- frozen BART ⇒ `param_groups` must return **one** group (CLAUDE.md invariant);
  subject_emb + bridge_transformer params must land in that single bridge group
  (`subject-temporal`'s `param_groups` already adds them — re-check after merge).

## Run mechanics

- Template: clone `train_full.sbatch`. For frozen cells **add `--freeze_bart`** and
  the Phase-2 BART flags (`--bart_warmup_epochs`, `--llrd_gamma`, `--bart_dropout`,
  `--bart_attention_dropout`, `--bart_weight_decay`) are inert — drop them for
  clarity. Match the existing cell's budget: `--epochs 60 --patience 25
  --warmup_epochs 10`.
- One parametrized sbatch is cleaner than 7 files: pass the 3 axes via env vars,
  e.g. `sbatch --export=SUBJ=64,TEMP=2,ALIGN=perword,TAG=s1t1aPW exp_e5_cell.sbatch`,
  with `--output_dir ../checkpoints_e5_${TAG}` and
  `--output logs/eeg2graph-e5-${TAG}-%J.out`. 7 submits, all queue in parallel.
- Each run writes `best_model.pt` (selected by `(val_F1,-val_loss)`) +
  `tokenizer/` + a `test_results.json` for the E5 table and downstream E3-style
  analysis.

## Deliverable & conclusion-either-way

`exp_logs/e5_ablation_grid.{json,md}`: 8-row table of test F1 (+ val_F1, final
align loss) per cell.
- **If a component lifts F1** → names what helps under frozen BART (e.g. per-word
  alignment > pooled, or subject-embedding carries the 0.57%).
- **If the whole grid stays ~0** → strong evidence the frozen-bridge ceiling is
  architecture-independent at this data scale, which hands the story to E6
  (data-scaling) and reinforces E1's fidelity-cliff conclusion.

## Estimated cost
7 × ~4h on one L40S each (≈ the `subject_temporal` 60-epoch budget), all parallel
in the `eliyanac` qos → ~one wall-clock afternoon if slots are free. Inference-only
post-analysis (E3-style) is minutes.

## After E5
- **E6** (data-scaling): best E5 cell at `--train_sentence_frac {0.1,0.25,0.5,1.0}`.
- **E8** (cross-subject): best E5 cell with `--exclude_subjects` held-out sets.
Both reuse `exp/tier2-grid` unchanged — which is why the flags are merged in now.
