# E5 — Frozen-BART component ablation grid (RQ4): results

Under frozen BART, which architectural component moves the needle? Each cell varies only the three axes; all other hyperparameters are fixed (seed 42, 60 epochs, patience 25, `--align_weight 1.0`, `--align_temp 0.07`).

**Completed: 8/8 cells.**

| # | subject | temporal | align | test F1 | test P | test R | best val F1 (ep) | align@best | align final |
|---|---|---|---|---|---|---|---|---|---|
| 1 | off | off | pooled | 0.0040 | 0.0038 | 0.0041 | 0.0015 (39) | 2.9140 | 2.8732 |
| 2 | on | off | pooled | 0.0083 | 0.0089 | 0.0077 | 0.0014 (31) | 2.9227 | 2.8803 |
| 3 | off | on | pooled | 0.0029 | 0.0027 | 0.0031 | 0.0000 (1) | 3.5312 | 2.8478 |
| 4 | on | on | pooled | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 3.2574 | 2.9092 |
| 5 | off | off | perword | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 9.1651 | 6.2145 |
| 6 | on | off | perword | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 9.0708 | 6.2140 |
| 7 | off | on | perword | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 7.6525 | 6.1751 |
| 8 | on | on | perword | 0.0000 | 0.0000 | 0.0000 | 0.0000 (1) | 9.6311 | 6.1555 |

## Read-off

- Best cell by test F1: **#2** (subject=on, temporal=off, align=pooled) → test F1 0.0083.
- Test-F1 spread across the grid: 0.0083.
- If the spread is ~0 the frozen-bridge ceiling is architecture-independent at this data scale (hands the story to E6 data-scaling, reinforces E1's fidelity cliff). If one axis lifts F1, that names what helps under frozen BART.

