# EEG2Graph — Experiment Index

Master index of every experiment, which branch it lives on, what it found, and a
deep-dive reading path for each. Generated 2026-06-27; last updated 2026-07-18 —
**Tier 2 is finished on the cluster: E5 (8/8), E6 (4/4) and E8 (7/7) all completed,
E7 was answered without a new run, E9 (retrieval probe, 19/19 checkpoints) is done, and E10
(sentence-level baselines + ZuCo-1 replication) closes the RQ1 arc.**

> **Branch note:** the E-series writeups/data are **not on `main`**. E1–E4 live on
> `exp/tier1-conclusions`; E5–E8 live on `exp/tier2-grid`. Use `git checkout <branch>`
> or `git show <branch>:<path>` to read them. See "Family B" for the
> architecture-variant branches whose checkpoints the E-series analyzes.

Two families:
- **Family A — diagnostic / conclusion experiments (E1–E8):** the "why does it fail"
  work. They analyze existing checkpoints or run controlled sweeps; they don't try to win.
- **Family B — architecture-variant runs:** the candidate model changes that
  produced the checkpoints Family A dissects. All fork from `4944849` ("Add
  text-encoder alignment loss").

---

## Status at a glance

| Exp | RQ | Cluster status | Branch |
|----|----|----------------|--------|
| E1 noise interpolation | RQ2 | **done** (post-hoc, no cluster job) | `exp/tier1-conclusions` |
| E2 EEG ablation | RQ3 | **done** (`exp_e2_ablation.sbatch`) | `exp/tier1-conclusions` (+ eval code on `exp/decoder-noise`, `exp/subject-temporal`) |
| E3 prediction collapse | RQ3 | **done** (post-hoc, CPU) | `exp/tier1-conclusions` |
| E4 signal probes | RQ1 | **done** (post-hoc, CPU) | `exp/tier1-conclusions` |
| E5 ablation grid | RQ4 | **done — 8/8 cells** (jobs 18451298–305) | `exp/tier2-grid` |
| E6 data scaling | RQ4 | **done — 4/4 points** | `exp/tier2-grid` |
| E7 per-word alignment | RQ2 | **answered, no new run needed** (it *is* E5 cells 5–8) | `exp/tier2-grid` (+ original impl on `exp/align-perword`) |
| E8 cross-subject | RQ5 | **done — 7/7 points** | `exp/tier2-grid` |
| E9 retrieval eval | RQ1/RQ2 | **done — 19/19 checkpoints** (job 19431000) | `exp/e9-retrieval` |
| E10 sentence baselines + ZuCo-1 replication | RQ1 | **done** (post-hoc, CPU, both datasets) | `exp/e10-baselines` |
| E11 difficulty ladder | RQ4 | **planned, not built** (superseded by E9/E10) | `exp/tier2-grid` |

**Headline across the whole series (E7, 2026-07-12):** the alignment InfoNCE loss has a
**chance floor** (pooled 2.865, per-word 6.182 — measured by `model/align_chance_floor.py`),
and **no run in E5, E6 or E8 ever beats it**. Every "the alignment loss went down" claim
was being read against an implicit floor of zero. The bridge does not learn EEG codes that
the frozen decoder then ignores; **it learns nothing at all** — it cannot pick the correct
sentence's text embedding out of a batch of 16 better than chance. That reframes E5/E6/E8
from three unexplained nulls into one quantified, mechanistic negative result.

**E9 update (2026-07-18) — the signal is *preserved*, not destroyed.** E4's identical
retrieval probe, re-run on the trained representations, shows every bridge retrieves the
read sentence well above chance (bridge top-10 ≈ 0.33, encoder ≈ 0.20 vs chance 0.149;
raw 0.395, untrained bridge 0.404, text upper bound 0.930). So E7's "the bridge learns
nothing" must be **scoped**: it learns nothing that aligns to REBEL's text geometry, but it
does **not** destroy the sentence-level identity signal it starts with — that survives into
the very states the decoder reads, and F1 is still 0. The failure is **representational
mismatch on two axes** — wrong *kind* (identity, not word-level triplet structure; E4) and
wrong *geometry* (not text-aligned; E7) — not signal absence. E9 is the reconciliation that
makes E4 (signal present) and E7 (nothing text-aligned) mutually consistent.

**E10 update (2026-07-18) — what the signal is *worth*, and a second dataset.** Used
directly, an **oracle retrieve-and-copy baseline** (retrieve the sentence from EEG, emit its
gold triplets) scores triplet-F1 **0.067 (ZuCo 2.0) / 0.094 (ZuCo 1.0) — ~8–10× the
generative model's 0.008**: the word-level generative architecture wastes ~90% of the signal
it demonstrably has (E9). But the **deployable** version (unseen sentences → retrieve from
train) and the model's own collapse mode (most-frequent-train triplet) both score **0.000** —
on a sentence-disjoint split triplets never recur, so nothing not handed the answer can
produce them. Separately, E4's probe **replicates on ZuCo 1.0** (12 different subjects):
sentence-identity 0.420 (2.7× chance, gamma-strongest), word-structure at chance. RQ1 is now
cross-dataset, and the negative result is robust.

---

## Family A — Diagnostic / conclusion experiments

| Exp | RQ | Branch | One-line finding |
|----|----|--------|------------------|
| **E1** Noise-interpolation / fidelity cliff | RQ2 | `exp/tier1-conclusions` | Generation needs the encoder to reproduce REBEL's *per-position* text states almost exactly (α≥0.9); the pooled alignment objective saturates at α≈0.5 → the bridge trains to a metric it can satisfy without ever generating correct triplets. |
| **E2** EEG-ablation (real/swap/zero) | RQ3 | writeup on `exp/tier1-conclusions`; eval code + checkpoints on `exp/decoder-noise`, `exp/subject-temporal` | The decoder *does* causally use the EEG (swap flips 56–93% of outputs, zero flips 54–100%), but the variation is non-functional — F1=0 in every condition. Bottleneck is EEG *misuse*, not EEG-blindness. |
| **E3** Prediction-collapse analysis | RQ3 | `exp/tier1-conclusions` | Quantifies the shortcut: `decoder_noise` emits one triplet for 86% of inputs, 100% memorized from train. `subject_temporal` is ~4× more diverse — architecture moves it off the constant-output mode (correlational). |
| **E4** EEG signal probes (REBEL-free) | RQ1 | `exp/tier1-conclusions` | Signal exists (~2.6× chance sentence retrieval, broadband, gamma-strongest) but it's *sentence-identity*, **not** the word-class/length structure the task needs. Bottleneck = representational alignment, not absence of signal. |
| **E5** Frozen-BART component ablation grid | RQ4 | `exp/tier2-grid` | **COMPLETE — 8/8 cells.** 2×2×2 grid (subject × temporal × align-mode) under frozen BART. **Null: the frozen ceiling is architecture-independent** — every cell at test F1 ≈ 0; best is cell #2 (subject-on/pooled) at **0.0083**; total spread 0.0083. All four per-word cells score exactly **0.0000** (0/1936 correct triplets). |
| **E6** Data-scaling curve | RQ4 | `exp/tier2-grid` | **COMPLETE — 4/4 points.** `--train_sentence_frac` ∈ {0.10, 0.25, 0.50, 1.00} on E5's best cell. F1 = 0.0006 / 0.0000 / 0.0023 / 0.0083 — **non-monotone and flat at the floor**; a 10× data increase buys less than the E5 architecture spread. **More ZuCo-scale data would not help.** |
| **E7** Per-word alignment + **alignment chance floor** | RQ2 | `exp/tier2-grid` | **ANSWERED WITHOUT A NEW RUN** — per-word alignment already ran as E5 cells 5–8. It scores 0/1936 and lands *exactly on its own chance floor*. Measuring the floors (pooled 2.865, per-word 6.182) shows **every run in E5/E6/E8 sits at or above chance**: the bridge never learned anything. Also: the per-word objective is multi-positive, so a perfectly-aligned encoder scores *worse than chance* — it never asked for what it was meant to ask for. |
| **E8** Cross-subject (shared vs subject-specific codes) | RQ5 | `exp/tier2-grid` | **COMPLETE — 7/7 points.** Paired `excl` (drop whole subjects) vs `ctrl` (drop matched sample count) at kept-frac 0.83/0.66/0.49. The subject-specificity penalty (ctrl − excl) is ≈0 and has the *wrong sign* (−0.0023 … −0.0011). **The ceiling is not a cross-subject-transfer wall.** Honest limit: at the floor, "shared codes" and "uniformly undecodable" are indistinguishable. |
| **E9** Bridge/encoder retrieval probe | RQ1/RQ2 | `exp/e9-retrieval` | **COMPLETE — 19/19 checkpoints.** E4's identical retrieval probe on the trained representations: bridge top-10 ≈ 0.31–0.37, encoder ≈ 0.20 — **well above chance 0.149** (raw 0.395, untrained bridge 0.404, text upper bound 0.930). Signal is **preserved, not destroyed**; F1≈0 is representational mismatch (wrong *kind* + wrong *geometry*), reconciling E4 (signal present) with E7 (not text-aligned). Temporal-bridge cells alone collapse toward chance; per-word alignment alone keeps identity through the encoder. |
| **E10** Sentence-level baselines + ZuCo-1 replication | RQ1 | `exp/e10-baselines` | **DONE (both datasets).** Oracle EEG retrieve-and-copy F1 **0.067 / 0.094** (ZuCo 2.0 / 1.0) — ~8–10× the model — but deployable (unseen sentences) and most-frequent-train both **0.000**: the signal does not support triplets on unseen sentences. E4 replicates on ZuCo 1.0 (sentence-identity 0.420, 2.7× chance; word-structure at chance). Constructive capstone + external validity. |

### Not yet run

- **E11 — difficulty ladder** (RQ4). Flags merged into `exp/tier2-grid`; nothing built.
  Largely **superseded** — E9 (signal preserved) and E10 (retrieval baseline scores 0 on
  unseen sentences) already locate the bottleneck as representational, so making the task
  easier is unlikely to change the verdict.

---

## Family B — Architecture-variant runs (candidate fixes)

All fork from `4944849` → `b090f55` (seeded baseline). These produced the checkpoints the
E-series dissects; their Slurm `.out` logs are on `main` in `exp_logs/`.

| Branch | What it changes | Status / result |
|--------|-----------------|-----------------|
| `exp/baseline` | Seeded control — bridge + REBEL, pooled alignment | ran (job 17929081) |
| `exp/recon-head` | + sentence-reconstruction auxiliary head | ran (job 17929072) |
| `exp/subject-cond` | + per-subject embedding conditioning | ran (job 17929104) |
| `exp/temporal-bridge` | + temporal Transformer pre-encoder in the bridge | ran (job 17929128) |
| `exp/subject-temporal` | **merge** of subject-cond + temporal-bridge | ran (job 17986027); analyzed by E2/E3 |
| `exp/align-perword` | per-position InfoNCE instead of pooled (**= E7**) | ran (job 17995163); re-run inside E5 as `align_mode perword` (cells 5–8) → 0/1936 correct, at chance |
| `exp/decoder-noise` | decoder-input token noise | ran (job 18096362); analyzed by E2/E3 |

`exp/tier2-grid` is the unified branch: it merges the subject / temporal / align-mode /
data-scaling / exclude-subjects flags into one `train.py` so E5–E8 are byte-comparable.

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

### E5 — component ablation grid (COMPLETE, 8/8)
1. `exp_logs/e5_ablation_grid_writeup.md` (the null result + three conclusions) →
   `exp_logs/e5_ablation_grid.md` (8-cell table with P/R/align) + `.json`
2. `exp_logs/e5_ablation_grid_plan.md` (grid table + "Prerequisite — unified branch":
   why `tier2-grid` exists)
3. `exp/tier2-grid` `model/train.py` — merged `--n_subject_buckets`,
   `--bridge_transformer_layers`, `--align_mode` flags + `contrastive_alignment_perword`
4. `exp_e5_cell.sbatch` (one job per cell via `--export`, TAG grammar
   `s{0,1}t{0,1}a{PL,PW}`) + `model/aggregate_e5.py` (scans `checkpoints_e5_*/`)
5. **Read E7 immediately after** — it reinterprets the align-loss column of this table.

### E6 — data-scaling curve (COMPLETE, 4/4)
1. `exp_logs/e6_data_scaling_writeup.md` (three conclusions + an in-place **correction
   notice**: the falling align loss is descent *toward chance*, not learning) →
   `exp_logs/e6_data_scaling.md` + `.json`
2. `exp_logs/e6_data_scaling_plan.md` (conclusion-either-way design: rising curve ⇒
   data-limited; flat curve ⇒ the E1 fidelity cliff is fundamental — it came out flat)
3. Holds E5's best cell #2 fixed, sweeps `--train_sentence_frac {0.10,0.25,0.50,1.00}`
   (subsamples by unique sentence, so the split invariant holds; val/test stay full)
4. `exp_e6_scale.sbatch` (`FRAC`+`TAG` via `--export`) + `model/aggregate_e6.py`

### E7 — per-word alignment & the alignment chance floor (NO NEW RUNS)
1. `exp_logs/e7_perword_alignment_writeup.md` — **read this before trusting any align-loss
   number anywhere in the series**
2. `model/align_chance_floor.py` — pushes random EEG states through the exact loss
   functions `train.py` uses (B=16, d=1024, temp 0.07, 20 trials) to measure the floors
3. Data: E5 cells 5–8 in `exp_logs/e5_ablation_grid.json` (there is no separate E7 dataset)
4. `train.py:contrastive_alignment_perword` — note the multi-positive design flaw: an
   exactly-aligned encoder scores 13.12, *worse* than the 6.18 chance floor

### E8 — cross-subject (COMPLETE, 7/7)
1. `exp_logs/e8_cross_subject_writeup.md` (four conclusions + "what this does **not**
   show" — E8 is underpowered *by construction* at the floor) →
   `exp_logs/e8_cross_subject.md` + `.json`
2. `exp_logs/e8_cross_subject_plan.md` (the paired excl/ctrl design and why the ctrl arm is
   mandatory: dropping subjects also drops samples, a confound E6 already measured)
3. `model/cross_subject_eval.py` (seen/unseen F1 split) + `model/aggregate_e8.py`
4. `exp_e8_cross_subject.sbatch` and `exp_e8_rescore.sbatch` (rescore-only job, added when
   a seen/unseen beam mismatch was found — commit `b3da206`)

### E9 — bridge/encoder retrieval probe (COMPLETE, 19/19)
1. `exp_logs/e9_retrieval_writeup.md` (seven conclusions + the E4/E7 reconciliation +
   "what this does **not** show") → `exp_logs/e9_retrieval.md` + `.json`
2. `exp_logs/e9_retrieval_plan.md` (design + **pre-registered** decision rule
   preserved-vs-destroyed, and the numpy-only raw/random baselines locked before the run)
3. `model/bridge_retrieval_probe.py` — E4's `retrieval_topk` copied **verbatim**; only the
   input feature matrix varies (raw / random_bridge / bridge / encoder / text upper bound)
4. `exp_e9_probe.sbatch` (post-hoc, auto-discovers `checkpoints_e5/6/8_*`) + run log
   `exp_logs/eeg2graph-e9-19431000.out`

### E10 — sentence-level baselines & cross-dataset replication (COMPLETE)
1. `exp_logs/e10_sentence_baseline_writeup.md` (E10a retrieve-and-copy: two conclusions
   pulling opposite ways; E10b ZuCo-1 replication) →
   `exp_logs/e10_sentence_baseline_{zuco2,zuco1}.json`
2. `model/sentence_retrieval_baseline.py` — REBEL-free; E4/E9 retrieval front end + micro-F1
   matching `train.py:compute_triplet_f1`; conditions oracle / deployable / random / freq-train
3. `exp_logs/e4_signal_probes_zuco1.json` (E10b = E4's probe re-run on ZuCo 1.0)

---

## Recommended cross-experiment narrative order

Read in **RQ order** (not numerical) — this is the actual argument:

> **E4** (RQ1: is there signal? → yes, but the wrong kind) → **E1** (RQ2: how exact must
> the encoder be? → near-perfect, and the pooled objective can't see it) → **E3** (RQ3
> correlational: how does it fail? → collapse to a memorized output) → **E2** (RQ3 causal:
> does it use the EEG? → yes, but non-functionally) → **E5** (RQ4: which component helps?
> → none; the frozen ceiling is architecture-independent) → **E6** (RQ4: does data scale
> lift it? → no, the curve is flat at the floor) → **E8** (RQ5: is it a subject-transfer
> wall? → no) → **E7** (the reframe: none of those nulls needed explaining, because the
> bridge never beat chance in the first place) → **E9** (…and yet the signal is *preserved*:
> the trained representation retrieves the sentence above chance all the way to the decoder's
> input, so the wall is representational mismatch, not signal loss) → **E10** (the
> constructive check: an oracle sentence-retrieval baseline beats the model ~10×, but even it
> scores 0 on unseen sentences — the signal does not support the task; and E4 replicates on a
> second dataset).

The chain: *signal exists but is misaligned (E4) → alignment must be per-position, and
pooled can't get there (E1) → so the model collapses (E3) → the collapse isn't
EEG-blindness, it's EEG-misuse (E2) → architecture can't fix it under frozen BART (E5) →
neither can data (E6) → nor is it a cross-subject wall (E8) → because the bridge never
learned any EEG code that aligns to REBEL's text geometry: every alignment loss in the
series sits at chance (E7) → and yet the signal is not *gone* — the trained representation
still retrieves the sentence above chance, so F1≈0 is representational mismatch (wrong kind
+ wrong geometry), not absence of signal (E9).*

**That open question is now answered (E9):** the bridge encoding *does* contain recoverable
sentence information — bridge retrieval ≈ 0.33 vs chance 0.149, above chance even in the
encoder states the decoder reads. So the negative result is not "no signal": it is signal of
the wrong *kind* (sentence identity, not triplet structure) in the wrong *geometry* (not
text-aligned), through a frozen decoder that cannot convert it.

---

## Caveats that apply to every number above

- **Alignment losses must be read against their chance floor, not zero** (E7). Pooled floor
  = 2.865 ± 0.122 at B=16; per-word floor = 6.182 ± 0.005 at B=16, S_t=30. Every observed
  value in E5/E6/E8 is at or above these. The E6 and E8 writeups carry in-place correction
  notices where they originally misread a falling/rising loss as learning; neither
  correction changes those experiments' conclusions, only the reason they are null.
- **Eval-parser cap:** every reported F1 uses `delinearize`, which caps even perfect
  output at ~0.745 (E1, Conclusion 1). All these runs sit at F1≈0 where both parsers
  agree, so it changes no conclusion — but it matters once a run produces good output.
- **Single seed (42)** across E5, E6 and E8, chosen for point-to-point comparability. At
  the floor a single-seed point wobbles; every conclusion rests on the *magnitude of the
  whole band* (≤0.008 F1), never on point-to-point ordering.
- **Conclusion branches are unmerged.** E1–E4 live only on `exp/tier1-conclusions`, E5–E8
  only on `exp/tier2-grid`, E9 only on `exp/e9-retrieval`. Merge all three into `main`
  before the final thesis writeup so the analysis isn't stranded on side branches.
