# E8 — Cross-subject: shared vs subject-specific EEG codes (RQ5): results

Frozen BART, subject embedding **OFF**, temporal OFF, pooled align, seed 42 (k0 == E5 cell `s0t0aPL`). Paired design: the **excl** arm drops whole subjects from TRAIN; the **ctrl** arm drops a matched sample count but keeps all 18 subjects. `unseen F1` = test F1 on the held-out subjects only (via `cross_subject_eval.py`).

**Completed: 7/7 points.**

| TAG | arm | held-out | kept frac | test F1 | seen F1 | unseen F1 | best val F1 (ep) |
|---|---|---|---|---|---|---|---|
| k0 | baseline | — | 1.000 | 0.0040 | 0.0040 | — | 0.0015 (39) |
| x03 | excl | 3 | 0.827 | 0.0023 | 0.0021 | 0.0032 | 0.0006 (43) |
| x06 | excl | 6 | 0.664 | 0.0021 | 0.0016 | 0.0031 | 0.0013 (42) |
| x09 | excl | 9 | 0.491 | 0.0023 | 0.0023 | 0.0023 | 0.0000 (1) |
| c03 | ctrl | 3 | 0.827 | 0.0000 | 0.0000 | — | 0.0024 (20) |
| c06 | ctrl | 6 | 0.664 | 0.0000 | 0.0000 | — | 0.0000 (1) |
| c09 | ctrl | 9 | 0.491 | 0.0012 | 0.0012 | — | 0.0012 (44) |

## Read-off — subject-specificity penalty (ctrl − excl)

At matched kept-sample fraction, `F1(ctrl) − F1(excl)` isolates the cost of removing whole *subjects* (vs random samples). Positive & growing with held-out count ⇒ subject-**specific** codes; ≈0 ⇒ subject-**shared** (only sample count matters).

| kept frac | held-out | excl F1 | ctrl F1 | ctrl − excl | unseen F1 |
|---|---|---|---|---|---|
| 0.827 | 3 | 0.0023 | 0.0000 | -0.0023 | 0.0032 |
| 0.664 | 6 | 0.0021 | 0.0000 | -0.0021 | 0.0031 |
| 0.491 | 9 | 0.0023 | 0.0012 | -0.0011 | 0.0023 |

- Largest ctrl−excl gap: -0.0011. A gap near 0 across all levels says the frozen ~0 ceiling is **not** a cross-subject-transfer artifact — codes are shared (or, at floor, equally undecodable) and removing subjects costs no more than removing the same #samples. A gap that grows with held-out count says the codes are subject-specific: a model never trained on a person cannot decode them. Either way, with E5 (architecture) and E6 (data) this closes the RQ4/RQ5 arc.
- `unseen F1` (held-out subjects only) is the direct generalize-to-a-new-person number; compare it to `seen F1` at the same point — a seen≫unseen split is the sharpest subject-specificity signal, undiluted by the seen majority.
