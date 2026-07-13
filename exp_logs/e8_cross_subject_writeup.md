# E8 — Cross-subject: is the frozen ceiling a transfer wall? (RQ5)

**Question:** E5 closed the *architecture* lever and E6 closed the *data-scale* lever —
under a frozen REBEL decoder, neither lifts test F1 off the floor. One explanation
survived both: maybe the bridge learns **subject-idiosyncratic** codes that don't
generalize to a new reader. Neither E5 nor E6 could see this, because both pool all 18
subjects in train *and* test. If the ~0 ceiling were really a cross-subject-transfer
wall, that would be a very different (and far more fixable) story than a signal-fidelity
wall. E8 tests it directly.

**Method:** Paired design, because dropping subjects also drops training samples — a
confound E6 already measured. The **excl** arm (`--exclude_subjects`) removes N whole
subjects from TRAIN; the **ctrl** arm (`--train_sentence_frac`) removes a *matched sample
count* but keeps all 18 subjects. val/test always keep all 18, so held-out subjects are
still scored — they are the "generalize to a new person" cases. Held-out sets are the
nested alphabetical tail of the sorted 18 (3 / 6 / 9 subjects → kept frac 0.827 / 0.664 /
0.491). Subject embedding is **OFF** (`--n_subject_buckets 0`) on purpose: a learned
per-subject vector is subject-specific *by construction*, and would confound "can the
content code transfer" with "is this subject's embedding trained." Everything else matches
E5/E6 (seed 42, 60 epochs, patience 25, temporal off, pooled align, frozen BART), so
`k0` reproduces E5 cell `s0t0aPL` as a built-in cross-check. Two read-offs: the paired
`F1(ctrl) − F1(excl)` at matched data size, and the seen-vs-unseen subject split from
`cross_subject_eval.py`. Data: `e8_cross_subject.json` / `.md`. Runs: 7/7 on the BGU
cluster.

## Results

| TAG | arm | held-out | kept frac | test F1 | seen F1 | unseen F1 | best val F1 (ep) | align final |
|---|---|---|---|---|---|---|---|---|
| k0  | baseline | — | 1.000 | 0.0040 | 0.0040 | — | 0.0015 (39) | 2.8732 |
| x03 | excl | 3 | 0.827 | 0.0023 | 0.0021 | 0.0032 | 0.0006 (43) | 3.0448 |
| x06 | excl | 6 | 0.664 | 0.0021 | 0.0016 | 0.0031 | 0.0013 (42) | 2.9810 |
| x09 | excl | 9 | 0.491 | 0.0023 | 0.0023 | 0.0023 | 0.0000 (1)  | 3.1532 |
| c03 | ctrl | 3 | 0.827 | 0.0000 | 0.0000 | — | 0.0024 (20) | 2.9487 |
| c06 | ctrl | 6 | 0.664 | 0.0000 | 0.0000 | — | 0.0000 (1)  | 3.0219 |
| c09 | ctrl | 9 | 0.491 | 0.0012 | 0.0012 | — | 0.0012 (44) | 3.0820 |

**Validity check — `k0` reproduces E5 cell #1 exactly.** Test F1 0.0040, P 0.0038,
R 0.0041, best val 0.0015 @ ep 39, align final 2.8732 — identical to four decimals on
every field. The E8 harness is byte-for-byte the E5 pipeline with one axis moved.

## Four conclusions

**1. The subject-specificity penalty is not just absent — it has the wrong sign.**

| kept frac | held out | excl F1 | ctrl F1 | ctrl − excl |
|---|---|---|---|---|
| 0.827 | 3 | 0.0023 | 0.0000 | **−0.0023** |
| 0.664 | 6 | 0.0021 | 0.0000 | **−0.0021** |
| 0.491 | 9 | 0.0023 | 0.0012 | **−0.0011** |

Subject-specific codes predict `ctrl ≫ excl`, growing with held-out count. Observed:
`ctrl < excl` at *every* level, with no growth. Dropping nine whole people from TRAIN cost
*less* than dropping the same number of random samples — which is incoherent under any
real mechanism, and that incoherence is the point: it is the signature of noise, not of a
weak effect with the sign flipped.

**2. The seen-vs-unseen split — the undiluted number — says the same thing.** Aggregate
test F1 dilutes transfer failure (held-out subjects are a minority of test), so
`cross_subject_eval.py` scores held-out subjects separately. This is the direct
generalize-to-a-new-person measurement, and it shows **no transfer penalty at all**:

| | seen F1 | unseen F1 |
|---|---|---|
| x03 (3 held out) | 0.0021 | **0.0032** |
| x06 (6 held out) | 0.0016 | **0.0031** |
| x09 (9 held out) | 0.0023 | **0.0023** |

`unseen ≥ seen` at every point. A model that never saw a person during training decodes
that person exactly as (un)successfully as the people it trained on. Two independent
read-offs — one aggregate and confounded-then-controlled, one direct and undiluted — both
land on the null, and both land slightly on the *wrong side* of it. That mutual
consistency is what makes the null trustworthy.

**3. Everything is inside the same floor band as E5 and E6.** The full 7-point spread is
0.0000–0.0040. E5's entire architecture grid spanned 0.0083; E6's entire 10× data curve
spanned 0.0083. Three orthogonal sweeps, one band, and it is the band of a handful of
lucky tuple matches against 1936 gold triplets. As in E5/E6, the runs whose best val F1
lands at epoch 1 (`x09`, `c06`) are degenerate; those that wander later (`k0` @39,
`x03` @43, `c09` @44) are "marginally less degenerate," not working.

> **CORRECTION (2026-07-12).** This paragraph originally added that the alignment loss
> "behaves sensibly — it degrades monotonically as training subjects are removed (2.8732
> at k0 → 3.1532 at x09), so the bridge is measurably learning *less* with fewer people."
> **That over-reads the number.** The pooled InfoNCE chance floor at `--batch_size 16` is
> **2.86** (measured; `model/align_chance_floor.py`), so *every* E8 run — k0's 2.8732
> included — sits **at or above chance**. There is no "learning less"; there is no
> learning at any point on this axis. The spread from 2.87 to 3.15 is movement *within*
> the chance band, not a degradation from a learned state. See
> `e7_perword_alignment_writeup.md`.

**4. RQ5 closes as a null: the ceiling is not a cross-subject-transfer wall.** The last
alternative explanation for the frozen ~0 ceiling is eliminated. Combined with the rest,
the failure is now characterized from five independent angles — E1 (fidelity cliff),
E3 (input-independent prediction collapse), E5 (architecture-independent), E6
(data-independent), E8 (subject-independent). Every lever that is *not* signal fidelity
has now been pulled, and none of them moves the outcome.

## What this does **not** show (the honest limit)

E8 is **underpowered by construction, and the direction of that weakness matters.** The
paired design can only detect subject-specificity if there is decodable signal available
to *lose* — and there isn't any. At the floor, "the code is subject-**shared**" and "the
code is **uniformly undecodable for everyone**" produce identical measurements, and E8
cannot separate them.

So the defensible claim is the **elimination**, not the positive: E8 rules out the
transfer-wall *explanation* for the ceiling. It does **not** establish that EEG→triplet
codes are shared across people. The canned read-off in `aggregate_e8.py` hedges this as
"codes are shared (or, at floor, equally undecodable)"; given E1/E3/E5/E6, the second
reading is the one the evidence supports, and the thesis should say so rather than quietly
banking the more attractive first one. A positive claim about shared codes would need a
setting where decoding works at all — which, on ZuCo-scale word-level EEG through a frozen
REBEL, is not this one.

## Caveats

- **Single seed (42)**, matching E5/E6 for point-to-point comparability. At floor a
  7-point single-seed set wobbles; the conclusion rests on the *magnitude* of the whole
  band (≤0.004) and on the agreement of two independent read-offs, not on any point-to-
  point ordering — which is exactly why the wrong-signed gaps are reported as noise rather
  than reinterpreted.
- **Held-out sets are a fixed alphabetical tail.** If a reviewer suspects the tail is
  idiosyncratic, rotating the held-out third at `x06`/`c06` is the cheap robustness check.
  Given `unseen ≥ seen` at all three levels, it is unlikely to change the story.
- **A beam-width bug was found and fixed while writing this up.**
  `cross_subject_eval.py` defaulted to `--beam_size 1` (greedy) while `train.py`'s test
  eval defaults to 4, so the first aggregation compared greedy seen/unseen F1 against
  beam-4 aggregate F1. It surfaced as `k0` reporting `seen_f1 = 0.0000` against
  `test_f1 = 0.0040` — impossible, since k0 excludes nobody and its "seen" partition *is*
  the whole test set. Fixed in `b3da206` (default → 4, pinned explicitly in the sbatch);
  all 7 points were re-scored at beam 4 by `exp_e8_rescore.sbatch`. The four runs that
  exclude no subjects (k0, c03, c06, c09) now show `seen F1 == test F1` to four decimals,
  which is the check that the two decoding paths are unified. **Aggregate test F1 was
  never affected** (it always came from `train.py` at beam 4), so the `ctrl − excl`
  read-off and the null result are unchanged by the fix; what the fix made quotable is
  `unseen F1`.
