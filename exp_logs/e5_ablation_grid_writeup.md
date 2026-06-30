# E5 — Frozen-BART component ablation (RQ4: does any architectural axis lift the frozen ceiling?)

**Question:** Under a frozen REBEL decoder — the regime where the EEG bridge is the
*only* path from brain signal to loss — does any architectural component move the
needle? We isolate three candidate levers and test every combination: a per-subject
embedding (does conditioning on who is reading help?), a temporal Transformer in the
bridge (does sequence modelling over words before REBEL help?), and per-word vs
pooled contrastive alignment (does a dense, position-wise InfoNCE target give the
bridge a better gradient than a single pooled one?).

**Method:** A 2×2×2 grid, one frozen-BART training run per cell. Axes:
`--n_subject_buckets {0,64}` (subject off/on), `--bridge_transformer_layers {0,2}`
(temporal off/on, `--bridge_nhead 8`), `--align_mode {pooled,perword}`. Everything
else is held fixed: `--freeze_bart`, `--bridge_layers 2`, `--align_weight 1.0`,
`--align_temp 0.07`, `--seed 42`, 60 epochs, patience 25. Each cell is launched by
`exp_e5_cell.sbatch` into `checkpoints_e5_<TAG>/` (TAG grammar `s{0,1}t{0,1}a{PL,PW}`)
and aggregated by `model/aggregate_e5.py`. Data: `e5_ablation_grid.json` /
`e5_ablation_grid.md`. Runs: 8/8 completed on the BGU cluster (jobs 18451298–305).

## Results

| # | subject | temporal | align | test F1 | best val F1 (ep) | align final |
|---|---|---|---|---|---|---|
| 1 | off | off | pooled  | 0.0040 | 0.0015 (39) | 2.8732 |
| 2 | on  | off | pooled  | **0.0083** | 0.0014 (31) | 2.8803 |
| 3 | off | on  | pooled  | 0.0029 | 0.0000 (1)  | 2.8478 |
| 4 | on  | on  | pooled  | 0.0000 | 0.0000 (1)  | 2.9092 |
| 5 | off | off | perword | 0.0000 | 0.0000 (1)  | 6.2145 |
| 6 | on  | off | perword | 0.0000 | 0.0000 (1)  | 6.2140 |
| 7 | off | on  | perword | 0.0000 | 0.0000 (1)  | 6.1751 |
| 8 | on  | on  | perword | 0.0000 | 0.0000 (1)  | 6.1555 |

## Three conclusions

**1. The frozen ceiling is architecture-independent.** Every cell lands at test
F1 ≈ 0; the best (cell 2, subject-on / pooled) is 0.0083 and the spread across the
entire grid is 0.0083 — i.e. the grid is flat at the floor. Neither subject
conditioning, nor a temporal bridge, nor per-word alignment, nor any combination,
produces a model that decodes triplets above noise. **No architectural axis moves
the needle under frozen BART.** This is a null result, and it is the answer to RQ4:
the bottleneck is not the bridge's capacity or its alignment objective.

**2. Per-word alignment optimizes its own loss without buying any F1.** The
per-word cells (5–8) drive their dense InfoNCE objective down (≈9.x → ≈6.2) — the
bridge *is* learning to match REBEL's per-position text encoding — yet every one of
them scores *exactly* 0.0 F1, with best-val at epoch 1 (no eval-time improvement
ever). Pooled alignment converges to a much lower absolute loss (≈2.87), but the two
modes are not directly comparable (per-word sums InfoNCE over many more positions).
The takeaway is qualitative: a better-aligned encoder representation does **not**
translate into decodable output. The cliff is downstream of alignment, in the frozen
decoder's cross-attention.

**3. The only non-zero cells are the longest survivors, and they are noise.** Cells
1 and 2 are the only runs whose best val epoch is past epoch 1 (39 and 31), and they
are the only ones with non-zero test F1 — but at 0.004–0.008 this is a handful of
lucky tuple matches over 1246 test samples, not signal. Read as "marginally less
degenerate," not "working."

## Where this points

E5 closes the architecture question for the frozen regime: you cannot engineer your
way past the ceiling with these components. Combined with **E1** (the noise-
interpolation fidelity cliff — output stays intact until EEG fidelity collapses past
a threshold) and **E3** (prediction collapse — ~0 F1 is input-independent
regurgitation), the picture is consistent: at this data scale the EEG→text bridge
cannot carry enough signal through a frozen REBEL for the decoder to act on, *and*
no architectural variant changes that. The remaining lever is **data scale, not
architecture** — which is exactly what **E6** tests via the `--train_sentence_frac`
curve already wired into this branch. If E6's curve is also flat, the fidelity cliff
(E1) is the binding constraint; if F1 climbs with more sentences, the frozen ceiling
is a sample-size artifact rather than a fundamental one.
