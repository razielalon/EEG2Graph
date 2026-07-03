# EEG2Graph — Experiment Index

Master index of every experiment, which branch it lives on, what it found, and a
deep-dive reading path for each. Generated 2026-06-27; updated 2026-07-03 (E5 grid
completed 8/8 cells; E6 scaffolding built).

> **Branch note:** the E-series writeups/data are **not on `main`**. E1–E4 live on
> `exp/tier1-conclusions`; E5 lives on `exp/tier2-grid`. Use `git checkout <branch>`
> or `git show <branch>:<path>` to read them. See "Family B" for the
> architecture-variant branches whose checkpoints the E-series analyzes.

Two families:
- **Family A — diagnostic / conclusion experiments (E1–E5):** the "why does it fail"
  work. They analyze existing checkpoints; they don't try to win.
- **Family B — architecture-variant runs:** the candidate model changes that
  produced the checkpoints Family A dissects. All fork from `4944849` ("Add
  text-encoder alignment loss").

---

## Family A — Diagnostic / conclusion experiments

| Exp | RQ | Branch | One-line finding |
|----|----|--------|------------------|
| **E1** Noise-interpolation / fidelity cliff | RQ2 | `exp/tier1-conclusions` | Generation needs the encoder to reproduce REBEL's *per-position* text states almost exactly (α≥0.9); the pooled alignment objective saturates at α≈0.5 → the bridge trains to a metric it can satisfy without ever generating correct triplets. |
| **E2** EEG-ablation (real/swap/zero) | RQ3 | writeup on `exp/tier1-conclusions`; eval code + checkpoints on `exp/decoder-noise`, `exp/subject-temporal` | The decoder *does* causally use the EEG (swap flips 56–93% of outputs, zero flips 54–100%), but the variation is non-functional — F1=0 in every condition. Bottleneck is EEG *misuse*, not EEG-blindness. |
| **E3** Prediction-collapse analysis | RQ3 | `exp/tier1-conclusions` | Quantifies the shortcut: `decoder_noise` emits one triplet for 86% of inputs, 100% memorized from train. `subject_temporal` is ~4× more diverse — architecture moves it off the constant-output mode (correlational). |
| **E4** EEG signal probes (REBEL-free) | RQ1 | `exp/tier1-conclusions` | Signal exists (~2.6× chance sentence retrieval, broadband, gamma-strongest) but it's *sentence-identity*, **not** the word-class/length structure the task needs. Bottleneck = representational alignment, not absence of signal. |
| **E5** Frozen-BART component ablation grid | RQ4 | `exp/tier2-grid` | **COMPLETE — 8/8 cells (jobs 18451298–305).** 2×2×2 grid (subject × temporal × align-mode) under frozen BART. **Null result: the frozen ceiling is architecture-independent** — every cell lands at test F1 ≈ 0, best is cell #2 (subject-on/pooled) at **0.0083**, total spread 0.0083. Per-word alignment drives its own loss down (≈9→6.2) but scores exactly 0.0 F1 — the cliff is downstream, in the frozen decoder's cross-attention. Hands the story to E6 (data scale). |

**E6 built, not yet launched** (`exp/tier2-grid`): data-scaling curve
(`--train_sentence_frac`) holding E5's best cell #2 fixed — scaffolding
(`exp_e6_scale.sbatch`, `model/aggregate_e6.py`) committed, 4 runs queued.

**Planned, not yet run** (flags already merged into `exp/tier2-grid`): **E8**
cross-subject (`--exclude_subjects`), **E10** difficulty ladder. **E7** = the
per-position InfoNCE, which *is* `exp/align-perword` (Family B) — and was also
folded into E5 as the `align_mode perword` axis (cells 5–8).

---

## Family B — Architecture-variant runs (candidate fixes)

All fork from `4944849` → `b090f55` (seeded baseline). These produced the
checkpoints the E-series dissects.

| Branch | What it changes | Status / result |
|--------|-----------------|-----------------|
| `exp/baseline` | Seeded control — bridge + REBEL, pooled alignment | early run (May-31 `.out` on main) |
| `exp/recon-head` | + sentence-reconstruction auxiliary head | early run |
| `exp/subject-cond` | + per-subject embedding conditioning | early run |
| `exp/temporal-bridge` | + temporal Transformer pre-encoder in the bridge | early run |
| `exp/subject-temporal` | **merge** of subject-cond + temporal-bridge | strongest checkpoint = E5 cell 4 (0.57%); analyzed by E2/E3 |
| `exp/align-perword` | per-position InfoNCE instead of pooled (**= E7**) | built + ran; the cliff fix E1 predicted |
| `exp/decoder-noise` | decoder-input token noise | built + ran; analyzed by E2/E3 |

---

## Deep-dive path for each experiment

For each, read **writeup → script → data → relevant model code**.

### E1 — fidelity cliff
1. `exp_logs/e1_noise_interpolation.md` (the two conclusions)
2. `model/noise_interpolation.py` — the `states(α)=α·text+(1−α)·noise` sweep; scores
   with **both** parsers (`extract` faithful vs `delinearize` train-eval)
3. `exp_logs/e1_noise_interpolation.json` + `.png`
4. `model/eeg_graph_model.py:encode_text` (text teacher being corrupted) +
   `train.py:contrastive_alignment_loss` (the pooled metric that saturates)

### E2 — causal EEG ablation
1. `exp_logs/e2_eeg_ablation.md` (esp. the "F1-delta is floored" caveat — evidence
   rests on `identical_to_real`, not F1)
2. `model/eeg_ablation_eval.py` — real/swap/zero derangement; `subject_idx` pinned to
   the output slot
3. `exp_logs/e2_eeg_ablation.json`
4. `inference.load_model` arch-flag forwarding (E2 needed the `bridge_layers`/subject/
   temporal fixes to load the checkpoints)

### E3 — prediction collapse
1. `exp_logs/e3_prediction_collapse.md`
2. `model/prediction_analysis.py` — pure post-hoc on `test_results.json`, no GPU; how
   `train_memorization`, `top1_share`, `pred_entropy_bits` are computed
3. `exp_logs/e3_prediction_collapse.json`
4. Read E2 next — E3 is correlational, E2 is its causal confirmation

### E4 — signal probes
1. `exp_logs/e4_signal_probes.md` (studies A–D)
2. `tests/eeg_signal_probe_extended.py` — REBEL-free linear probes; 840 = 8 bands ×
   105 channels band-major; chance = 10/67
3. `exp_logs/e4_signal_probes.json`
4. `tests/eeg_signal_probe.py` (the original probe E4 extends)

### E5 — component ablation grid (COMPLETE)
1. `exp_logs/e5_ablation_grid_writeup.md` (the null result + three conclusions) →
   `exp_logs/e5_ablation_grid.md` (8-cell table with P/R/align) + `.json`
2. `exp_logs/e5_ablation_grid_plan.md` (grid table + "Prerequisite — unified branch":
   why `tier2-grid` exists)
3. `exp/tier2-grid` `model/train.py` — merged `--n_subject_buckets`,
   `--bridge_transformer_layers`, `--align_mode` flags + `contrastive_alignment_perword`
4. `exp_e5_cell.sbatch` (one job per cell via `--export`, TAG grammar
   `s{0,1}t{0,1}a{PL,PW}`) + `model/aggregate_e5.py` (scans `checkpoints_e5_*/`)

### E6 — data-scaling curve (BUILT, not yet launched)
1. `exp_logs/e6_data_scaling_plan.md` (build & run plan + conclusion-either-way:
   rising curve ⇒ data-limited; flat curve ⇒ E1 fidelity cliff is fundamental)
2. Holds E5's best cell #2 fixed, sweeps `--train_sentence_frac {0.10,0.25,0.50,1.00}`
   (subsamples by unique sentence, so the split invariant holds; val/test stay full)
3. `exp_e6_scale.sbatch` (`FRAC`+`TAG` via `--export`) + `model/aggregate_e6.py`
   → `exp_logs/e6_data_scaling.{json,md}`

---

## Recommended cross-experiment narrative order

Read in **RQ order** (not numerical) — this is the actual argument:

> **E4** (RQ1: is there signal? → yes, but wrong kind) → **E1** (RQ2: how exact must
> the encoder be? → near-perfect, and the objective can't see it) → **E3** (RQ3
> correlational: how does it fail? → collapse to memorized output) → **E2** (RQ3
> causal: does it use EEG? → yes but non-functionally) → **E5** (RQ4: which component
> helps? → none; the frozen ceiling is architecture-independent) → **E6** (RQ4
> follow-up: does data scale lift it? → TBD).

The chain: *signal exists but is misaligned (E4) → alignment must be per-position,
pooled can't get there (E1) → so the model collapses (E3) → the collapse isn't
EEG-blindness, it's EEG-misuse (E2) → architecture can't fix it under frozen BART
(E5) → is it data-limited or fundamental? (E6)*.

---

## Caveats that apply to every number above

- **Eval-parser cap:** every reported F1 uses `delinearize`, which caps even perfect
  output at ~0.745 (E1, Conclusion 1). All these runs sit at F1≈0 where both parsers
  agree, so it changes no conclusion — but it matters once a run produces good output.
- **Conclusion branches are unmerged.** E1–E4 live only on `exp/tier1-conclusions`,
  E5 on `exp/tier2-grid`. Consider merging `exp/tier1-conclusions` → `main` before
  the final thesis writeup so the analysis isn't stranded.
