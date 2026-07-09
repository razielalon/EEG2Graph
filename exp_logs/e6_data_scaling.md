# E6 — Frozen-BART data-scaling curve (RQ4 follow-up): results

Best E5 cell (#2: subject=64, temporal=0, align=pooled, frozen BART, seed 42) held fixed; the only moving part is `--train_sentence_frac` — the fraction of TRAIN *unique sentences* seen. val/test stay full.

**Completed: 4/4 points.**

| frac | test F1 | test P | test R | best val F1 (ep) | align final | n_train |
|---|---|---|---|---|---|---|
| 0.10 | 0.0006 | 0.0007 | 0.0005 | 0.0006 (16) | 3.4062 | — |
| 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 3.0738 | — |
| 0.50 | 0.0023 | 0.0026 | 0.0021 | 0.0000 (1) | 3.0019 | — |
| 1.00 | 0.0083 | 0.0089 | 0.0077 | 0.0014 (31) | 2.8803 | — |

## Read-off

- Test F1 at frac 0.10 → 1.00: 0.0006, 0.0000, 0.0023, 0.0083.
- Range across the curve: 0.0083 (min 0.0000, max 0.0083).
- Monotone non-decreasing in data: no.
- Slope of the top half (frac 0.5 → 1.0): +0.0060. A clear positive slope still climbing at frac=1.0 argues the frozen ceiling is a sample-size artifact (more data would help). A flat curve at ~0 argues the E1 fidelity cliff is the binding constraint at this data scale — architecture (E5) and data (E6) are both exhausted, and the bottleneck is EEG signal fidelity itself.

