# E6 — Frozen-BART data-scaling curve (RQ4 follow-up): build & run plan

**Status: BUILT, not yet launched.** Scaffolding (`exp_e6_scale.sbatch`,
`model/aggregate_e6.py`) is committed on `exp/tier2-grid`; the 4 runs launch on
the BGU cluster.

**Question:** E5 closed the architecture question — under frozen BART, no
architectural axis (subject embedding, temporal bridge, per-word alignment, or any
combination) lifts the ~0 test-F1 ceiling. The one remaining lever is **data
scale**. Does test F1 climb as the model sees more training sentences, or is the
curve flat at the floor?

- **F1 climbs with data (still rising at frac=1.0)** → the frozen ceiling is a
  sample-size artifact; more ZuCo-scale data would help, and the thesis story is
  "data-limited, not fundamental."
- **Curve flat at ~0** → the **E1 fidelity cliff** is the binding constraint.
  Architecture (E5) and data (E6) are both exhausted; the bottleneck is EEG signal
  fidelity itself at word granularity. This is the stronger, cleaner conclusion.

## The curve (4 points)

Hold the **best E5 cell (#2)** fixed and vary only `--train_sentence_frac`:

| axis (fixed) | value |
|---|---|
| subject embedding | `--n_subject_buckets 64` (ON) |
| temporal bridge | `--bridge_transformer_layers 0` (OFF) |
| alignment mode | `--align_mode pooled` |
| decoder | `--freeze_bart` |
| seed / epochs / patience | 42 / 60 / 25 (identical to E5) |

| TAG | `--train_sentence_frac` | note |
|---|---|---|
| f010 | 0.10 | new |
| f025 | 0.25 | new |
| f050 | 0.50 | new |
| f100 | 1.00 | == E5 cell 2 (`checkpoints_e5_s1t0aPL`, test F1 0.0083) |

`--train_sentence_frac` subsamples by **unique sentence** (not by sample), so the
split invariant holds — the same sentence read by multiple subjects stays together
(`_select_sentence_fraction` in `eeg_graph_dataset.py`, deterministic by `--seed`).
val/test are always left full, so every point is scored on the identical test set.

**f100 = E5 cell 2 under byte-identical code/seed.** Re-run it fresh for a
self-contained E6 table (cheap insurance, same rationale as E5 re-running cell 4),
or symlink `checkpoints_e6_f100 -> checkpoints_e5_s1t0aPL` and skip the 4th submit.

## Run mechanics

One parametrized sbatch, `FRAC` + `TAG` via `--export` (mirrors `exp_e5_cell.sbatch`):

```bash
sbatch --export=ALL,FRAC=0.1,TAG=f010  exp_e6_scale.sbatch
sbatch --export=ALL,FRAC=0.25,TAG=f025 exp_e6_scale.sbatch
sbatch --export=ALL,FRAC=0.5,TAG=f050  exp_e6_scale.sbatch
sbatch --export=ALL,FRAC=1.0,TAG=f100  exp_e6_scale.sbatch   # or symlink from cell 2
```

Each writes `checkpoints_e6_<TAG>/{best_model.pt,tokenizer/,test_metrics.json,history.json}`.
Aggregate any time (safe mid-queue) with:

```bash
cd model && python aggregate_e6.py    # -> exp_logs/e6_data_scaling.{json,md}
```

## Deliverable & conclusion-either-way

`exp_logs/e6_data_scaling.{json,md}`: 4-point curve of test F1 vs
`--train_sentence_frac`, plus a slope/monotonicity read-off. Either outcome is a
publishable result: a rising curve reframes the ceiling as data-limited; a flat
curve nails the fidelity cliff as fundamental and closes the RQ4 arc together with
E1 (fidelity cliff), E3 (prediction collapse), and E5 (architecture-independent).

## Caveats / possible follow-ups
- **Single seed (42)** matches E5 and keeps points comparable, but frozen F1 sits
  at floor noise (~0.008 at full), so a 4-point single-seed curve can wobble. If
  the curve is ambiguous (non-monotone within noise), add seeds {123, 7} at the
  informative fractions (0.5, 1.0) and report mean±sd before drawing the slope.
- Points only go **down** from full ZuCo-2; the diagnostic is the slope *near*
  frac=1.0. A curve still climbing there is the case for "more data would help" —
  which, if seen, motivates E7/beyond (more subjects or ZuCo-1 merge), not a
  frozen-vs-unfrozen change.

## Estimated cost
3 new × ~4h on one L40S each (f100 reused or re-run), all parallel in the
`eliyanac` qos → one wall-clock afternoon if slots are free.
