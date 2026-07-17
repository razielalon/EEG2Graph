# E9 — Bridge/encoder retrieval probe (RQ1/RQ2): plan

**Status:** built, not yet run on the cluster. Baselines (raw, random_bridge)
reproduced locally. Branch: `exp/e9-retrieval` (forks `exp/tier2-grid`).

## The question

E7 left one question with actual dynamic range: does the trained bridge encoding
contain *any* recoverable sentence information? E4 says the **raw** EEG does
(linear probe → ~2.6× chance sentence retrieval, top-10 0.394). E7 says the
**alignment loss** never beats its chance floor. E9 asks the question those two
straddle: **once the EEG passes through the trained bridge, is the E4 signal still
there, or did training remove it?**

This directly resolves the apparent E4↔E7 contradiction the thesis has to defend.

## Method

Run E4's retrieval probe (`retrieval_topk`, copied **verbatim** from
`tests/eeg_signal_probe_extended.py`: StandardScaler → Ridge → TF-IDF/SVD sentence
target → nearest-neighbour retrieval among the 67 unique test sentences, chance
top-10 = 10/67 = 0.149) — changing **only** the input feature matrix. Five sources,
each mean-pooled to one vector per sentence over fixated words (text over text
tokens):

| source | features | role |
|---|---|---|
| `raw` | mean-pooled raw 840-dim EEG | reference (must reproduce E4's 0.394) |
| `random_bridge` | untrained `Linear(840→1024)+LayerNorm`, 3 seeds | control: a generic projection preserves the signal (≈ raw) |
| `bridge` | trained checkpoint's `_bridge_forward` output (1024-dim) | **verdict**: did the trainable map keep the signal? |
| `encoder` | REBEL encoder states over the bridge output | **verdict**: is the signal in what the decoder reads? |
| `text` | REBEL's own text-encoder states for the gold sentence | upper bound (should be ≈1.0, reproducing E1's α=1) |

Runs on the 19 checkpoints already in hand (E5 8 + E6 4 + E8 7); `load_model`
forwards each checkpoint's `n_subject_buckets` / `bridge_transformer_layers` so
subject- and temporal-variant bridges are reconstructed correctly. Script:
`model/bridge_retrieval_probe.py`. Job: `exp_e9_probe.sbatch` (post-hoc, ~4h, 1 GPU,
no training). Output: `exp_logs/e9_retrieval.json` / `.md`.

## Local pre-registration (already measured, numpy-only)

| source | top1 | top10 | median rank |
|---|---|---|---|
| raw EEG (E4 reproduction) | 0.065 | **0.394** | 16 |
| chance | 0.015 | 0.149 | 34 |
| random_bridge (3 seeds) | 0.069 | 0.404 ± 0.002 | 15 |

So the reference band is locked: **raw ≈ random_bridge ≈ 0.39–0.40**. The only
unknown is where the *trained* bridge/encoder land relative to that band and the
0.149 chance line.

## Decision rule (conclusion either way)

- **`bridge`/`encoder` top-10 ≈ 0.39–0.40** (in the raw/random band) → training
  **preserved** the signal. F1 ≈ 0 is then a **decoder-usability** failure, not an
  absence of signal: the sentence identity survives into the very states the
  decoder cross-attends to, but the frozen REBEL cross-attention (tuned for text
  states) cannot turn it into triplets. This *softens* E6/E7's "fundamental at
  word granularity" into "present but unusable through a frozen decoder" — a
  different, more fixable thesis (fine-tune cross-attention / a learned adapter),
  and it flags the E4↔E7 gap as a target-mismatch artifact (InfoNCE-to-REBEL-text
  vs. Ridge-to-TF-IDF), not a real contradiction.
- **`bridge`/`encoder` top-10 ≈ 0.149** (at chance) → training **destroyed** the
  signal that `random_bridge` shows was there at init. The InfoNCE-through-frozen-
  decoder objective actively collapses the bridge onto a degenerate manifold. This
  *hardens* the negative result and pins the failure on the objective/optimization,
  and it fully reconciles E4 (signal in the input) with E7 (nothing in the trained
  representation).
- **Intermediate** (between 0.149 and 0.39) → partial degradation; report the
  fraction of E4's signal retained, per architecture cell (does subject/temporal/
  per-word help *retention* even though it never helped F1?).

## What E9 does **not** show

Retrieval into a TF-IDF/SVD text space is a *sentence-identity* probe (same as E4),
not a word-level triplet-structure probe — E4 already showed the word-level
structure the task needs is absent. So a high `bridge` retrieval would show the
bridge preserves *what little signal exists*, not that triplets are decodable. E9
adjudicates "preserved vs destroyed," which is the open question; it does not
reopen "is sentence identity enough for triplets" (E4/E1 answer: no).

## Sanity checks baked in

- `raw` must land at 0.394 (locks the harness to E4).
- `text` must land near 1.0 (locks the pooling/encoder path; if it doesn't,
  masked mean-pooling of encoder states is misconfigured and the `encoder` numbers
  are not trustworthy).
- `k0`/`f100` bridges should match their E5 twins (`s0t0aPL`/`s1t0aPL`).
