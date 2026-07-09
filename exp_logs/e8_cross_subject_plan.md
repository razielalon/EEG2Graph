# E8 — Cross-subject: shared vs subject-specific EEG codes (RQ5): build & run plan

**Status: BUILT + CPU-smoke-validated, not yet launched.** Scaffolding
(`exp_e8_cross_subject.sbatch`, `model/cross_subject_eval.py`,
`model/aggregate_e8.py`) is committed on `exp/tier2-grid`; the 7 runs launch on the
BGU cluster.

**Question (RQ5):** E5 closed architecture and E6 closed data scale — under frozen
BART, neither lifts the ~0 test-F1 ceiling. The last lever is *who the EEG came
from*. Is the decodable EEG→triplet code **shared across people**, or
**subject-specific**? A model trained without subject S's data is still scored on S
(val/test always keep all 18 subjects), so held-out subjects are the "generalize to
a new person" cases.

- **Removing whole subjects hurts more than removing an equal number of random
  samples** → the code is **subject-specific**: a model never trained on a person
  cannot decode them. The ~0 ceiling would be (partly) a cross-subject-transfer
  wall.
- **Removing subjects costs no more than removing the same #samples** → the code is
  **subject-shared** (or, at floor, equally undecodable for everyone). The ceiling
  is *not* a transfer artifact; combined with E1/E3/E5/E6 it nails EEG signal
  fidelity as the single binding constraint.

Either outcome closes the RQ4/RQ5 arc.

## The confound, and the paired design that removes it

Excluding subjects also shrinks the training set — an effect E6 already measures.
So E8 pairs every exclusion against a **matched sample-count control**:

| arm | knob | what it removes |
|---|---|---|
| **excl** | `--exclude_subjects` | N whole subjects (all their samples) |
| **ctrl** | `--train_sentence_frac` | a matched #samples, but **all 18 subjects kept** |

The subject-specificity signal at each level is `F1(ctrl) − F1(excl)` at matched
kept-sample fraction. ctrl ≫ excl ⇒ subject-specific; ctrl ≈ excl ⇒ subject-shared.

## Fixed axes (differ from E5/E6 in one deliberate way)

| axis | value | note |
|---|---|---|
| subject embedding | `--n_subject_buckets 0` (**OFF**) | **key deviation** — see below |
| temporal bridge | `--bridge_transformer_layers 0` (OFF) | as E5/E6 |
| alignment | `--align_mode pooled` | as E5/E6 |
| decoder | `--freeze_bart` | as E5/E6 |
| seed / epochs / patience | 42 / 60 / 25 | identical to E5/E6 |

**Why subject embedding OFF:** a learned per-subject vector is *by construction*
subject-specific — leaving it on would give every held-out subject an untrained
embedding, trivially confounding "can the content code transfer" with "is this
subject's embedding trained." Turning it off makes E8 a clean test of the
**EEG-content** code. Consequence: the `k0` baseline reproduces **E5 cell
`s0t0aPL`** (subject OFF, temporal OFF, pooled) — a built-in cross-check.

## The 7 points

Held-out sets are the nested alphabetical tail of the 18 sorted subject ids
(YAC…YTL); each larger set contains the smaller, so the excl arm is a monotone
sweep. Kept fraction = kept/10216 train samples (subjects have unequal sizes, so
these are exact, not N/18).

| TAG | arm | held-out subjects | kept samples | frac | matched ctrl |
|---|---|---|---|---|---|
| `k0` | baseline | — (all 18) | 10216 | 1.000 | — |
| `x03` | excl | YSD,YSL,YTL | 8447 | 0.827 | `c03` |
| `x06` | excl | YRH,YRK,YRP,YSD,YSL,YTL | 6781 | 0.664 | `c06` |
| `x09` | excl | YLS,YMD,YMS,YRH,YRK,YRP,YSD,YSL,YTL | 5015 | 0.491 | `c09` |
| `c03` | ctrl | — (frac 0.827) | ~8449 | 0.827 | pairs `x03` |
| `c06` | ctrl | — (frac 0.664) | ~6783 | 0.664 | pairs `x06` |
| `c09` | ctrl | — (frac 0.491) | ~5016 | 0.491 | pairs `x09` |

The `x09`/`c09` pair (**half** the subjects / half the samples) is the headline: the
held-out subjects are then half of TRAIN's subjects, so their test share is large
and the aggregate is least diluted.

## Run mechanics

One parametrized sbatch; `TAG` + `EXCLUDE` (`-`-delimited → CSV, to dodge sbatch's
`--export` comma-splitting) + `FRAC` via `--export`.

**Minimal first cut (3 jobs)** — the headline comparison: `k0` baseline + the
`x09`/`c09` pair (half the subjects held out vs half the samples dropped), the
highest-signal, least-diluted point. Run this first; fill in the intermediate
counts only if the gap is worth resolving:

```bash
sbatch --export=ALL,TAG=k0,EXCLUDE=,FRAC=                                     exp_e8_cross_subject.sbatch
sbatch --export=ALL,TAG=x09,EXCLUDE=YLS-YMD-YMS-YRH-YRK-YRP-YSD-YSL-YTL,FRAC= exp_e8_cross_subject.sbatch
sbatch --export=ALL,TAG=c09,EXCLUDE=,FRAC=0.491                               exp_e8_cross_subject.sbatch
```

**Full sweep (adds 4 jobs)** — turns the ctrl−excl gap into a 3-level curve:

```bash
sbatch --export=ALL,TAG=x03,EXCLUDE=YSD-YSL-YTL,FRAC=                         exp_e8_cross_subject.sbatch
sbatch --export=ALL,TAG=x06,EXCLUDE=YRH-YRK-YRP-YSD-YSL-YTL,FRAC=             exp_e8_cross_subject.sbatch
sbatch --export=ALL,TAG=c03,EXCLUDE=,FRAC=0.827                               exp_e8_cross_subject.sbatch
sbatch --export=ALL,TAG=c06,EXCLUDE=,FRAC=0.664                               exp_e8_cross_subject.sbatch
```

Each job trains, then runs `cross_subject_eval.py` (seen-vs-unseen subject F1 →
`checkpoints_e8_<TAG>/test_by_subject.json`); aggregate test F1 is in
`test_metrics.json` as usual. Aggregate any time (safe mid-queue):

```bash
cd model && python aggregate_e8.py    # -> exp_logs/e8_cross_subject.{json,md}
```

## Why the extra `cross_subject_eval.py`

Aggregate test F1 dilutes the effect: with 3/18 subjects held out, unseen samples
are a small minority, so the aggregate barely moves even if unseen decoding fully
fails. `cross_subject_eval.py` re-scores the test split partitioned into **seen vs
unseen** subjects (plus a per-subject table) using the exact `compute_triplet_f1`
train.py reports — so `unseen F1` is the direct generalize-to-a-new-person number.
It reuses `inference.load_model`/`predict_batch`; it does **not** touch the training
path (zero risk to E5/E6 reproducibility).

## Deliverable & conclusion-either-way

`exp_logs/e8_cross_subject.{json,md}`: the 7-point table + the paired
`ctrl − excl` read-off + seen/unseen split. A flat (~0) gap says the ceiling is not
a transfer wall (subject-shared / uniformly undecodable); a gap growing with
held-out count says subject-specific codes. Both are publishable and close RQ5.

## Validation done (this session, CPU box)

- `--exclude_subjects YSL,YTL` → `train.py` reports `9038/10216 samples kept`;
  frozen 1-epoch smoke trains and writes `test_metrics.json`.
- `cross_subject_eval.py` on that checkpoint partitions correctly: excluding a
  subject present in the slice routed it to `unseen` (seen 10 + unseen 14 = 24
  overall); F1 helper normalizes case/whitespace and handles the empty partition.
- `aggregate_e8.py` renders with 0 and with synthetic DONE points (paired read-off
  table + growing-gap summary).

## Caveats / possible follow-ups

- **Single seed (42)**, matching E5/E6. At floor, a 7-point single-seed set can
  wobble; if the `ctrl − excl` gap is ambiguous within noise, add seeds {123, 7} at
  the `x09`/`c09` pair (the highest-signal point) and report mean±sd.
- `--train_sentence_frac` matches *sample count approximately* (it subsamples whole
  sentences; sample count ≈ fraction since each sentence is read by ~all 18
  subjects). The ctrl frac values are set to the excl arm's exact kept fractions.
- The excl arm holds out a fixed alphabetical tail. If a reviewer worries the tail
  is idiosyncratic, a rotation (hold out a different third) at `x06`/`c06` is the
  cheap robustness check.
