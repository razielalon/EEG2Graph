# E6 — Frozen-BART data-scaling curve (RQ4: is the frozen ceiling sample-size-limited or a fidelity cliff?)

**Question:** E5 closed the *architecture* question — under a frozen REBEL decoder,
no component (subject embedding, temporal bridge, per-word alignment, or any
combination) lifts test F1 off the floor. The one remaining lever is **data scale**.
If the bridge simply hasn't seen enough sentences, test F1 should climb as we feed it
more; if the bottleneck is EEG signal fidelity itself (the E1 cliff), the curve stays
flat at ~0 no matter how much data we add. E6 draws that curve.

**Method:** Hold the best E5 cell fixed (#2: `--n_subject_buckets 64`, temporal off,
`--align_mode pooled`, `--freeze_bart`, `--seed 42`, 60 epochs, patience 25) and vary
only `--train_sentence_frac` ∈ {0.10, 0.25, 0.50, 1.00}. Subsampling is by **unique
sentence**, not by sample, so the split invariant holds (the same sentence read by
multiple subjects stays together) and the curve measures the effect of seeing fewer
*sentences*, not fewer copies of the same one. val/test are always left full, so
every point is scored on the identical test set. `frac 1.00` reproduces E5 cell 2
under byte-identical code/seed. Each point is launched by `exp_e6_scale.sbatch` into
`checkpoints_e6_<TAG>/` (TAG `f010/f025/f050/f100`) and aggregated by
`model/aggregate_e6.py`. Data: `e6_data_scaling.json` / `e6_data_scaling.md`. Runs:
4/4 completed on the BGU cluster.

## Results

| frac | train sentences (≈ samples) | test F1 | test P | test R | best val F1 (ep) | align final |
|---|---|---|---|---|---|---|
| 0.10 | ~1015 | 0.0006 | 0.0007 | 0.0005 | 0.0006 (16) | 3.4062 |
| 0.25 | ~2554 | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1)  | 3.0738 |
| 0.50 | ~5095 | 0.0023 | 0.0026 | 0.0021 | 0.0000 (1)  | 3.0019 |
| 1.00 | 10216 | **0.0083** | 0.0089 | 0.0077 | 0.0014 (31) | 2.8803 |

## Three conclusions

**1. The curve is flat at the floor — data scale does not lift the frozen ceiling.**
A 10× increase in training sentences (frac 0.10 → 1.00) moves test F1 from 0.0006 to
0.0083: an absolute gain of ~0.008, which is the same magnitude as the *entire E5
architecture grid spread* (0.0083). The whole curve lives inside one noise band. This is
the "flat at ~0" outcome, and it is the answer to RQ4's data half: **more ZuCo-scale data
would not help.**

> **CORRECTION (2026-07-12).** This section originally read the falling alignment loss
> (3.41 → 2.88) as evidence that "the bridge is demonstrably learning a better-aligned
> encoder representation with more data." **That was wrong.** The pooled InfoNCE loss has
> a *chance floor* — the value produced by a bridge that has learned nothing is not 0, it
> is ln(B) for batch size B. At `--batch_size 16`, that floor is **2.86** (measured;
> `model/align_chance_floor.py`). So 3.41 → 2.88 is not learning: it is the loss
> descending *toward chance from above* and stopping there. The correct statement is that
> more data moves the bridge closer to **having learned nothing**, and no point on this
> curve — including frac 1.00 at 2.8803 — ever gets below chance. See
> `e7_perword_alignment_writeup.md`, which establishes this across E5/E6/E8.

**2. The apparent uptick is not a trend.** The `aggregate_e6.py` read-off flags a
+0.0060 slope over the top half (0.50 → 1.00), which in isolation could read as "still
climbing." It is not: the curve is **non-monotone** — frac 0.25 scores 0.0000, *below*
frac 0.10's 0.0006 — so the sequence is noise, not a rising function of data. As in
E5, the only points with best-val past epoch 1 (frac 0.10 at ep 16, frac 1.00 at
ep 31) are also the only ones with any non-zero F1, at 0.0006–0.0083: a handful of
lucky tuple matches over 1246 test samples, not signal. Read "marginally less
degenerate," not "working."

**3. Architecture (E5) and data (E6) are both exhausted.** Two orthogonal levers —
model capacity/objective and training-set size — have now each been swept under the
frozen decoder, and neither produces a model that decodes triplets above floor noise.
By elimination, the binding constraint is neither of them. It is the **fidelity of the
EEG signal itself** at word granularity — exactly the wall E1 located directly (the
noise-interpolation cliff) and E3 characterized (input-independent prediction
collapse).

## Where this points

E6 converts E5's "architecture-independent ceiling" into a stronger, cleaner claim:
the ceiling is **fundamental at this data scale, not a sample-size artifact**. The
RQ4 arc now closes as a triangulation — E1 (fidelity cliff), E3 (prediction collapse),
E5 (architecture-independent), E6 (data-independent) — all pointing at the same
bottleneck: a word-level EEG→text bridge cannot carry enough signal through a frozen
REBEL for the decoder to act on, and neither better architecture nor more data
changes that. This is the thesis's central negative result, and it is defensible from
four independent angles.

The one lever E6 deliberately does *not* touch is *whose* brain the signal comes from.
If the ~0 ceiling were a cross-subject-transfer wall — the bridge learning
subject-idiosyncratic codes that don't generalize to held-out readers — neither the
architecture grid nor the data curve would reveal it, because both pool all 18
subjects. **E8** (cross-subject, `--exclude_subjects`, already scaffolded on this
branch) tests exactly that, and is the last piece of the failure-characterization
story.

## Caveats

- **Single seed (42)**, matching E5 for point-to-point comparability. At floor, a
  4-point single-seed curve wobbles within noise (hence the non-monotonicity above);
  the conclusion rests on the *magnitude* of the range (~0.008, i.e. floor) rather
  than the point-to-point ordering. Seeds {123, 7} at frac {0.5, 1.0} would tighten
  the error bars, but cannot rescue a curve whose maximum is 0.0083.
- Points only go **down** from full ZuCo-2, so the diagnostic is the behaviour *near*
  frac=1.0; there is no evidence of a knee that more data would climb past.
