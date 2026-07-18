# E9 — Bridge/encoder retrieval probe: the signal is preserved, not destroyed (RQ1/RQ2)

**Status: done — 19/19 checkpoints, one post-hoc job (19431000).** This is the one
experiment in the series whose metric has real dynamic range, and it settles the
E4↔E7 tension the thesis has to defend.

## The question

E4 found a linear probe on the **raw** EEG retrieves the read sentence at ~2.6×
chance (top-10 0.394). E7 found the **trained bridge**'s InfoNCE alignment never
beats its chance floor. Read naively those contradict each other ("signal exists"
vs "bridge learns nothing"). E9 removes the ambiguity by running E4's *identical*
retrieval probe (`retrieval_topk`, copied verbatim: StandardScaler → Ridge →
TF-IDF/SVD sentence target → nearest-neighbour among the 67 unique test sentences,
chance top-10 = 10/67 = 0.149) and changing **only the input feature matrix** —
from raw EEG to the trained model's own representations. Decision rule (pre-
registered in `e9_retrieval_plan.md`): retrieval **≈ raw** ⇒ training *preserved*
the signal (wall is downstream); retrieval **≈ chance** ⇒ training *destroyed* it.

## Results

**Baselines (shared, computed once):**

| source | top-10 | top-1 | median rank |
|---|---|---|---|
| raw EEG (E4 reproduction) | **0.395** | 0.065 | 16 |
| random bridge — untrained Linear+LN, 3 seeds | 0.404 ± 0.002 | 0.069 | 15 |
| text encoder — REBEL, upper bound | **0.930** | 0.639 | 1 |
| *chance* | 0.149 | 0.015 | 34 |

`raw` reproduces E4 to the third decimal (harness locked); `text` upper bound is at
median rank 1 (encoder-pooling path locked). Full 19-row table:
`exp_logs/e9_retrieval.md` / `.json`.

**Trained representations (bridge output / encoder states), grouped:**

| group | config | bridge top-10 | encoder top-10 | val F1 |
|---|---|---|---|---|
| E5 pointwise, pooled (`s0t0aPL`,`s1t0aPL`) | temporal off, pooled | 0.326–0.339 | 0.225–0.246 | ~0 |
| E5 pointwise, **per-word** (`s0t0aPW`,`s1t0aPW`) | temporal off, per-word | 0.343–0.348 | **0.326–0.357** | 0 |
| E5 **temporal** (`s*t1a*`) | temporal **ON** | **0.157–0.204** | 0.202–0.230 | 0 |
| E6 data curve (`f010`→`f100`) | temporal off, pooled | 0.320–0.366 | 0.177–0.246 | ~0 |
| E8 excl/ctrl (`x*`,`c*`) | temporal off, pooled | 0.307–0.323 | 0.168–0.221 | ~0 |

Reference points: chance 0.149, raw 0.395, untrained bridge 0.404, text 0.930. At
n=1246 test samples the standard error of a top-10 proportion is ≈0.010, so bridge
≈0.33 is ~18 SE above chance and encoder ≈0.20 is ~5 SE above; even the weakest
unseen-subject encoder cell (0.168) is ~2 SE above.

## Conclusions

**1. Verdict: the signal is preserved, not destroyed.** Every trained bridge
retrieves the read sentence far above chance. The pointwise bridges cluster at
top-10 0.31–0.37 (mean ≈0.33) — retaining **~70% of the untrained bridge's**
above-chance retrieval and ~73% of raw EEG's. Training does not collapse the bridge
to chance. The "destroyed" branch of the decision rule is **ruled out**; the outcome
is the "present but unusable" branch.

**2. The signal survives into the states the decoder reads — attenuated but still
above chance.** REBEL's frozen encoder roughly halves the pooled-cell retrieval
(bridge ≈0.33 → encoder ≈0.20): a text-tuned encoder partially washes out EEG-side
identity. But the encoder states the decoder actually cross-attends to are still
above chance (~5 SE), and F1 is nonetheless **exactly 0**. So the decoder is handed
above-chance sentence information and still emits no correct triplets.

**3. This reconciles E4 and E7 without overturning either.** E4 (raw, fresh probe) =
0.395; E7 (trained bridge vs REBEL text states, InfoNCE) = chance; E9 (trained
representation, fresh probe) = 0.33 bridge / 0.20 encoder. The three agree once two
questions the earlier work conflated are separated:
- *Is sentence identity linearly recoverable from the representation?* — **yes**, at
  every stage (E4, E9).
- *Is the representation aligned to REBEL's text-embedding geometry that the frozen
  cross-attention consumes?* — **no** (E7).

E7's "the bridge learns nothing" is exact but must be **scoped**: the bridge learns
nothing that matches REBEL's text geometry; it does **not** destroy the (weak,
sentence-level) identity signal it starts with — that survives training into the
decoder's input. Signal present ≠ signal in the usable coordinate system.

**4. The failure is representational mismatch on two axes, not signal absence.**
Combining E4 + E7 + E9, F1≈0 has a precise mechanistic cause: the information
reaching the decoder is **(a) the wrong *kind*** — sentence identity, not the
word-level triplet structure the task requires (E4) — and **(b) in the wrong
*geometry*** — not aligned to the text-state manifold REBEL's cross-attention was
built to read (E7). E9 closes the last loophole ("maybe training just deletes the
signal"): the signal is measurably there. The negative result sharpens from *"no
signal"* to *"signal present, wrong kind, wrong geometry, through a frozen decoder
that cannot convert it."*

**5. The temporal bridge is the one component that destroys signal.** The
temporal-Transformer cells collapse bridge retrieval to 0.157–0.204 (mean 0.177) —
the closest any trained representation gets to chance. The sequence-mixing
pre-encoder scrambles the per-word identity fingerprint before REBEL sees it. This
sharpens E5's null: the temporal bridge doesn't merely fail to lift F1, it *actively
removes* the little recoverable signal. (REBEL's encoder partially re-lifts these
cells to ~0.22, but they never rejoin the pointwise band.)

**6. Per-word alignment is the one objective that keeps identity through the
encoder.** The per-word cells (`s0t0aPW`,`s1t0aPW`) are the only trained models whose
*encoder* retrieval (0.33–0.36) matches their bridge and rivals raw EEG — pooled
alignment lets REBEL's encoder wash identity out to ~0.20, per-word alignment holds
it in. Yet both score 0 F1. So per-word alignment demonstrably *shapes* the encoder
representation (consistent with E7's finding that it optimizes *a* loss) — it just
shapes it toward better sentence identity, which is still the wrong target for
triplets.

**7. Retrievability is architecture/data/subject-flat, exactly like F1.** Within the
pointwise-pooled family the E6 data curve (bridge 0.366 → 0.334 → 0.320 → 0.326) and
the E8 excl/ctrl arms (0.307–0.323, with unseen ≈ seen) sit in one band —
retrievability moves as little as F1 did across E5/E6/E8. E8's unseen-subject bridges
retain the same ~0.31 as seen, an independent echo of "not a cross-subject-transfer
wall."

## What E9 does **not** show (honest limits)

- It measures **sentence identity, not triplet structure**. Above-chance identity in
  the decoder's input does *not* imply the triplets are recoverable — E4 already
  showed the word-level structure the task needs is absent. "Present but unusable" is
  a claim about signal *kind and geometry*, not a promise that a better decoder would
  succeed.
- The "present but unusable" reading suggests a fixable direction — unfreeze/adapt
  REBEL's cross-attention, or learn a text-space adapter so the decoder can consume
  the EEG geometry — but E9 supports that as a **hypothesis**, not a proof.
- **Single seed, single probe.** The above-chance margins are large for the bridge
  (~18 SE) and modest for the encoder (~2–5 SE); the ordering *within* the 0.31–0.37
  band should not be over-read.
- The text upper bound is **0.93 under this probe**, not 1.0 — a harness cap
  (Ridge→TF-IDF), analogous to E1's delinearize cap, not a limit of the text states
  themselves (which sit at median rank 1).

## Where this points

E9 is the capstone of the RQ1/RQ2 arc. It converts the series' central claim from
"the bridge never learned anything" into the stronger, two-axis diagnosis of
Conclusion 4, and it is the reconciliation that makes **E4** (signal in the input)
and **E7** (nothing in the trained *alignment*) mutually consistent. Narrative
order for the thesis is now:
**E4** (signal exists, wrong kind) → **E1** (fidelity cliff) → **E3/E2** (collapse,
EEG-misuse) → **E5/E6/E8** (architecture/data/subject-independent) → **E7** (bridge
never text-aligns) → **E9** (…but the signal is *preserved*, above chance, all the
way to the decoder's input — so the wall is representational mismatch, not signal
loss).

## Reproduction

`model/bridge_retrieval_probe.py` (numpy baselines + lazy-torch trained sources),
`exp_e9_probe.sbatch` (post-hoc, auto-discovers `checkpoints_e5/6/8_*`), results in
`exp_logs/e9_retrieval.{json,md}`. Branch `exp/e9-retrieval` (off `exp/tier2-grid`).
