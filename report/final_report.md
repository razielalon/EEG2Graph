# Reconstructing Knowledge Graphs from Reading EEG: An Anatomy of a Negative Result

**EEG2Graph — Final Project Report**

*Authors: [names] · Supervisor: [name] · [Department], Ben-Gurion University · [date]*

> **Draft status.** This is a complete first draft assembled from the project's ten
> experiments (E1–E10; see `exp_logs/EXPERIMENTS.md` for the machine-readable index and
> per-experiment writeups). Placeholders in [brackets] and the reference list need to be
> filled in / formatted to the department's style.

---

## Abstract

We set out to translate word-level EEG recorded during natural reading (the ZuCo 1.0 and
2.0 corpora) into linearized knowledge-graph triplets, using a thin trainable *bridge* that
projects EEG features into the embedding space of REBEL, a BART-large model pre-fine-tuned
for relation extraction. The system does not work: across every configuration we trained,
test-set triplet-F1 never rises above ~0.008, statistically indistinguishable from zero.
Rather than report this as an engineering failure, we treat it as the object of study and
ask *why* it fails, through ten controlled experiments organized around five research
questions. Our central finding is that the failure is **not** an absence of signal, an
architecture defect, a data-scale limitation, or a cross-subject transfer wall — each of
which we test and rule out — but a **representational mismatch on two axes**. The EEG carries
a weak but real *sentence-identity* signal (≈2.6–2.7× chance sentence retrieval, replicated
across both datasets) that survives training into the very decoder states the model reads;
yet it is (i) the wrong *kind* of information — sentence identity, not the word-level
relational structure triplet generation requires — and (ii) in the wrong *geometry* — not
aligned to the text-embedding manifold REBEL's frozen cross-attention consumes. We further
show, constructively, that using the signal at the granularity where it actually lives (an
oracle sentence-retrieval baseline) beats the full generative model by ~8–10×, while still
failing to generalize to unseen sentences. We contribute a quantified, mechanistic account
of the failure and two reusable diagnostics: an InfoNCE *chance-floor* analysis and a
*representation-retrieval* probe.

**Keywords:** EEG decoding, knowledge graphs, relation extraction, brain-to-text, negative
results, representational alignment, ZuCo, REBEL.

---

## 1. Introduction

### 1.1 Problem and motivation

Reading comprehension leaves measurable traces in the brain. If those traces could be
decoded into structured meaning — not just words, but *relations* between entities — it would
be a step toward brain-computer interfaces that operate at the level of propositions rather
than characters. We formalize this as a sequence-to-sequence problem: given the EEG recorded
while a person reads a sentence, emit the set of `(subject, relation, object)` triplets that
the sentence expresses, i.e. its knowledge-graph representation.

```
Input:  EEG recorded while reading "Barack Obama was born in Hawaii."
Output: (Barack Obama, place of birth, Hawaii)
```

The obvious obstacle is data scale. Relation extraction models are trained on hundreds of
thousands of sentences; the ZuCo EEG corpora provide on the order of ten thousand word-level
samples. Our design answers this with transfer: instead of teaching a decoder the triplet
grammar from scratch, we reuse **REBEL** [Huguet Cabot & Navigli, 2021], a BART-large
[Lewis et al., 2020] checkpoint already fine-tuned to emit the exact linearized-triplet
format we want. The only component that must learn from EEG is a small **bridge** that maps
neural features into REBEL's embedding space. In principle the gradient then flows into the
EEG→text mapping rather than into relearning language.

### 1.2 The pivot: from "make it work" to "explain why it doesn't"

The system did not learn to decode triplets. After exhausting the standard levers —
architecture variants, a contrastive alignment objective, regularization, longer training —
test-set triplet-F1 remained pinned at essentially zero. At that point we changed the goal.
A model that outputs nothing useful is uninformative; a *well-characterized* failure is a
result. We therefore reframed the project as a diagnostic study: to determine, rigorously and
from multiple independent angles, **what specifically prevents word-level ZuCo EEG from being
decoded into knowledge-graph triplets through this architecture.** This report is that study.

### 1.3 Research questions

- **RQ1 — Signal.** Is there decodable linguistic signal in the EEG at all, and of what
  kind? (Experiments E4, E9, E10.)
- **RQ2 — Fidelity & alignment.** How faithfully must the bridge reproduce the decoder's
  expected input, and does the training objective actually drive it there? (E1, E7.)
- **RQ3 — Failure mode.** When a run scores ~0, *how* does it fail — varied-but-wrong output,
  or collapse to an input-independent answer? (E3, E2.)
- **RQ4 — Architecture & data.** Under a controlled (frozen-decoder) regime, does any
  architectural component or more data lift performance off the floor? (E5, E6.)
- **RQ5 — Subject transfer.** Is the ceiling a cross-subject generalization wall? (E8.)

### 1.4 Contributions

1. A **quantified, mechanistic negative result**: EEG→triplet failure on ZuCo is
   representational mismatch (wrong *kind* and wrong *geometry* of signal), not signal
   absence, architecture, data scale, or subject transfer — each ruled out with a dedicated
   experiment.
2. Two reusable **diagnostics**: (a) an InfoNCE **chance-floor** calibration that turns
   "the alignment loss went down" into a testable claim (E7), and (b) a
   **representation-retrieval probe** that measures whether signal is preserved through a
   trained network (E9).
3. A **constructive baseline** showing that a trivial sentence-retrieval approach, working at
   the granularity where the signal exists, outperforms the full generative pipeline by
   ~8–10× — while still failing to generalize, which sharpens rather than softens the
   negative result (E10).
4. **Cross-dataset external validity**: the core signal finding replicates on ZuCo 1.0 and
   2.0 (independent subjects and tasks).

---

## 2. Background and related work

**EEG during reading.** The ZuCo corpora [Hollenstein et al., 2018; 2020] record EEG and eye
tracking simultaneously during natural sentence reading. We use the eye-tracking fixations to
align brain activity to individual words: for each fixated word we extract the EEG of its
first gaze-duration window, yielding a fixed-length feature vector per word. EEG is a
low-signal-to-noise, non-invasive modality; prior brain-to-text work typically operates at
the level of coarse semantic categories or whole-sentence classification rather than
fine-grained structure.

**Relation extraction and REBEL.** Extracting `(subject, relation, object)` triplets from
text is a mature NLP task. REBEL [Huguet Cabot & Navigli, 2021] casts it as autoregressive
generation: a BART-large encoder–decoder emits a linearized string with special markers
(`<triplet>`, `<subj>`, `<obj>`) in subject–object–relation order. Because the markers are
native vocabulary tokens and the decoder is already fluent in the grammar, we can reuse the
pretrained weights directly and avoid teaching the format from our small EEG dataset.

**Bridging modalities and contrastive alignment.** Mapping one modality into a frozen model's
input space is a common transfer pattern. Because our decoder is teacher-forced during
training — it sees the gold previous tokens and can lower cross-entropy without attending to
the EEG — we add a contrastive **InfoNCE** [van den Oord et al., 2018] auxiliary loss that
pulls the bridge's encoder states toward REBEL's own encoder states for the gold sentence
text, providing a dense gradient into the bridge that does not route through the decoder. The
adequacy of this objective is itself one of our research questions (RQ2).

---

## 3. Data

**Corpora.** ZuCo 2.0 (18 subjects; normal-reading and task-specific-reading tasks) is our
primary dataset; ZuCo 1.0 (12 subjects) is used for external-validity replication. ZuCo 1.0
ships as MATLAB v5 and 2.0 as MATLAB v7.3 (HDF5); our preprocessing auto-detects and
dispatches accordingly.

**Features.** Each word is represented by a **840-dimensional** vector: 8 frequency bands ×
105 EEG channels, from the word's first gaze-duration fixation. Words that were skipped during
reading are zero-padded and flagged by a `has_fixation` mask. Features are z-scored per
subject, computed over fixated words only. A sentence is thus a variable-length sequence of
840-dim word vectors.

**Targets.** For each unique sentence *text*, we run REBEL itself to produce the gold
triplets, so the training targets are in exactly the format the decoder was pretrained to
emit. This yields `sentence_triplets.json`, keyed by sentence text.

**Splits.** We split 80/10/10 **by sentence text**, not by sample, so that a sentence read by
multiple subjects stays entirely within one split. This is essential and has a consequence we
return to in §6.1: **test sentences are disjoint from training sentences**, so any correct
output must generalize to unseen sentences, not memorize. For ZuCo 2.0 the test split
contains 1,246 samples spanning 67 unique sentences and 1,936 gold triplets.

---

## 4. Model and methods

### 4.1 Architecture

```
Word-level EEG  (B, S, 840)
        │
   Bridge  (trainable):  Linear 840→1024 · LayerNorm · Dropout      (~0.9M params)
        │  inputs_embeds
   REBEL = BART-large fine-tuned for relation extraction  (~406M params; low-LR or frozen)
        │
   Linearized triplets:  "<triplet> Barack Obama <subj> Hawaii <obj> place of birth"
```

The bridge is the only component that sees EEG; its output replaces REBEL's encoder token
embeddings via `inputs_embeds`. The output dimension is 1024 (BART-large's `d_model`). We
deliberately keep the bridge minimal (no nonlinearity before REBEL's first Transformer block,
which hurt early training). Optional variants add a per-subject embedding, a temporal
Transformer over word positions, and layer configurations, all evaluated in §6.4.

### 4.2 Training

We train with label-smoothed cross-entropy plus the InfoNCE alignment loss (§2), using
differential learning rates (a high rate for the bridge, a low rate for REBEL) with
layer-wise decay, and a warm-up when REBEL is unfrozen. Two regimes matter:

- **Unfrozen REBEL.** The full model fine-tunes. This is the natural setting but exposes the
  teacher-forcing shortcut (§6.3).
- **Frozen REBEL** (`--freeze_bart`). Only the bridge trains, so the bridge is the *only* path
  from EEG to loss. This is the clean regime for asking whether the bridge can carry signal at
  all (§6.4), and is used for E5/E6/E8/E9.

### 4.3 Evaluation

Predictions are generated (beam search), delinearized back into triplet tuples, and scored
with **micro-averaged exact-match F1** on lowercased `(subject, relation, object)` tuples.
Exact match is strict but unambiguous. One caveat (established in §6.2): the training-time
parser caps even *perfect* output at F1 ≈ 0.745, so all F1 numbers are read against that
ceiling; since our runs sit at ~0 this changes no conclusion.

---

## 5. Experimental design

The ten experiments fall into two families. **Family A** (E1–E10, diagnostic) analyzes
existing checkpoints or runs controlled sweeps to explain the failure. **Family B**
(architecture-variant training runs) produced the checkpoints Family A dissects. We report by
research question rather than by number, because that is the actual line of argument. Two
methodological choices recur:

- **Chance floors.** A contrastive loss decreasing is meaningless without knowing the value a
  model that has learned *nothing* would attain (`ln N` for N candidates). We measure these
  floors explicitly (E7) and read every alignment number against them.
- **Single seed (42)** across the frozen-regime sweeps, chosen for point-to-point
  comparability. Because every conclusion rests on the *magnitude of a whole band* (F1 ≤
  0.008 across dozens of cells), not on point ordering, single-seed noise does not threaten
  the results; we flag this as a limitation in §8.

---

## 6. Results

### 6.1 RQ1 — Is there signal, and of what kind?

**There is signal, but it is sentence identity, not word structure (E4).** Stripping the
model away entirely, we train linear probes directly on the 840-dim features. A probe can
retrieve *which* of the 67 test sentences a subject read at **top-10 = 0.394 (≈2.6× the 0.149
chance rate)**, median rank 16/67. The signal is broadband and strengthens with frequency
(gamma bands strongest), and it is spatially distributed (no single channel carries it). But
probes for *word-level* properties fail: content-vs-function word classification reaches
0.696 against a 0.699 majority baseline (i.e. chance), and word-length regression gives
R² = −0.018 (no signal). **The decodable signal is a sentence-level fingerprint; the
word-level linguistic structure the triplet task needs is not linearly present.**

**This replicates on a second dataset (E10b).** Re-running the identical probe on ZuCo 1.0
(12 different subjects, different task pairing) reproduces the picture: sentence retrieval
top-10 = 0.420 (≈2.7× chance), gamma-strongest, while content/function (0.737 vs 0.738
majority) and word length (R² = −0.017) sit at chance. RQ1's answer is therefore not a
ZuCo-2 artifact.

**The signal survives training into the model (E9).** Does the trained bridge preserve this
signal or destroy it? We re-run E4's identical retrieval probe on the model's *own*
representations, changing only the input features. Against a raw-EEG reference of 0.395 and a
0.149 chance line, an *untrained* random projection retrieves at 0.404 (a generic linear map
preserves the signal), and the **trained bridge retrieves at 0.31–0.37**, its output states
still carrying most of the sentence identity. Passing those through REBEL's frozen encoder —
the states the decoder actually reads — attenuates but does not erase it (**≈0.20, still above
chance**), while REBEL's own text-encoder states retrieve near-perfectly (0.93, upper bound).
So training does **not** destroy the signal; it is present all the way to the decoder's input.

**What that signal is worth, used directly (E10a).** If the model's problem were merely
extraction, we should be able to use the sentence-identity signal in the most direct way:
retrieve the nearest sentence from the EEG and emit *its* gold triplets. An **oracle**
retrieve-and-copy baseline (gallery = the test sentences) scores **F1 = 0.067 (ZuCo 2.0) /
0.094 (ZuCo 1.0)** — roughly **8–10× the generative model's best 0.008**, and ~5× a
random-sentence-copy floor (0.014/0.017). The generative architecture extracts about a tenth
of what a trivial look-up does from the same signal.

But two facts keep this from being a fix, and they are the crux of the negative result: the
**deployable** version — retrieve the nearest *training* sentence, since real test sentences
are unseen — scores **0.000**, as does the model's own collapse mode (emit the
most-frequent-train triplet, §6.3). Because splits are by sentence, triplets essentially never
recur across sentences; nothing that is not already handed the candidate answers can produce
them. **The sentence-identity signal is real, and it reaches the decoder, but it does not
transfer to triplet generation on unseen sentences — because the thing that would transfer is
word-level relational structure, and that is exactly what the EEG lacks.**

### 6.2 RQ2 — How faithful must the bridge be, and does the objective get it there?

**A fidelity cliff the objective cannot see (E1).** Taking REBEL's own encoder states for the
test sentences and interpolating them toward matched noise, `states(α) = α·text + (1−α)·noise`,
we find generation is extremely fidelity-sensitive: F1 falls from 1.0 to 0 as α drops from
1.0 to ~0.55, needing α ≥ 0.9 for usable output. **But the pooled InfoNCE retrieval the
bridge optimizes stays at 1.0 across that entire collapse (still 0.985 at α = 0.5, where F1
is already 0).** The objective is satisfied long before the per-position fidelity generation
requires — so an encoder can look perfectly aligned by the contrastive metric and still
produce zero correct triplets. (This experiment also establishes the ≈0.745 eval-parser
ceiling noted in §4.3.)

**The alignment loss never beats chance (E7).** InfoNCE has a chance floor of `ln N`; nobody
had computed it for our objectives, so "the alignment loss decreased" had been read against an
implicit zero. Pushing random EEG states through the exact loss functions gives floors of
**2.865 (pooled)** and **6.182 (per-word)**. *Every* pooled run across E5/E6/E8 sits at or
above 2.865; the per-word runs sit at 6.16–6.21, exactly their floor. **The bridge does not
learn a representation that aligns to REBEL's text geometry — it captures essentially none of
the available headroom.** (We also found the per-word objective mis-specified: being
multi-positive, a perfectly aligned encoder scores 13.12, *worse* than its 6.18 floor; it
never asked for what it was meant to.) Read together with E9, the reconciliation is precise:
the sentence signal is *linearly recoverable by a fresh probe* (E4/E9) but is *not in REBEL's
text-embedding coordinate system* (E7) that the frozen cross-attention consumes.

### 6.3 RQ3 — How do the failing runs fail?

**Collapse to a memorized, input-independent answer (E3).** Analyzing saved predictions, the
representative unfrozen run emits the *same single triplet* for 86% of test inputs (prediction
entropy 0.71 bits, ~3 distinct outputs over 1,246 inputs), and **100% of what it emits appears
verbatim in the training set**. The ~0 F1 is not "varied but wrong"; it is input-independent
regurgitation. A stronger architecture (subject + temporal) is measurably less collapsed (33%
top-1 share, 2.94 bits, 19% train-memorized) but no more accurate.

**Yet the decoder does condition on the EEG (E2).** Regenerating each checkpoint's predictions
under real, swapped (deranged), and zeroed EEG, the outputs *move*: swapping flips 56–93% of
predictions and zeroing 54–100%. So the failure is **not** a decoder ignoring the EEG; it is
that the EEG-driven variation is **non-functional** — F1 is 0 under every condition. The
bottleneck is EEG *misuse*, not EEG-blindness. (Because real-EEG F1 is already 0, this causal
evidence rests on the sequence-change rates, not on an F1 delta — a caveat we state
explicitly.)

### 6.4 RQ4 — Can architecture or data lift the ceiling?

**No architectural component helps (E5).** Under frozen REBEL — where the bridge is the only
path to loss — we run a 2×2×2 grid over subject conditioning, a temporal Transformer, and
pooled-vs-per-word alignment. Every one of the 8 cells lands at test F1 ≈ 0; the best is 0.0083
and the entire grid spans 0.0083. The frozen ceiling is **architecture-independent**.

**Neither does data scale (E6).** Holding the best cell fixed and sweeping training-sentence
fraction ∈ {0.10, 0.25, 0.50, 1.00}, F1 is 0.0006 / 0.0000 / 0.0023 / 0.0083 — **flat at the
floor and non-monotone**. A 10× data increase buys the same ~0.008 as the entire architecture
grid. More ZuCo-scale data would not help. (E9 adds nuance: the temporal-Transformer cells are
in fact the *only* ones whose retrievable signal collapses toward chance — sequence-mixing
scrambles the per-word identity fingerprint — so that component is actively harmful, not
merely unhelpful.)

### 6.5 RQ5 — Is it a cross-subject transfer wall?

**No (E8).** A paired design drops either whole subjects (`excl`) or a matched number of
random samples (`ctrl`) from training, scoring held-out subjects at test. Subject-specific
codes would predict `ctrl ≫ excl`, growing with the number held out. Observed: the gap is ≈0
and has the *wrong sign* (−0.0023 … −0.0011), and held-out ("unseen") subjects are decoded no
worse than seen ones. The ceiling is not a subject-transfer artifact. (Honest limit: at the
floor, "subject-shared codes" and "uniformly undecodable" are indistinguishable; E8 rules out
the transfer-wall *explanation*, it does not establish shared codes.)

---

## 7. Discussion

Every benign explanation for F1 ≈ 0 has an experiment that eliminates it: it is not signal
absence (E4, E9, E10 — signal exists, survives training, and is worth ~10× the model if used
directly), not the failure being EEG-blindness (E2 — the decoder does use the EEG), not the
architecture (E5), not the data scale (E6), and not subject transfer (E8). What remains is a
**two-axis representational mismatch**:

1. **Wrong kind.** The only decodable EEG signal is sentence identity; the task needs
   word-level relational structure, which is not linearly present in the features (E4,
   replicated E10b) and does not transfer across sentences (E10a's deployable 0.000).
2. **Wrong geometry.** The sentence signal the bridge does preserve (E9) is not aligned to
   REBEL's text-embedding manifold that the frozen decoder cross-attends to (E7), so the
   decoder cannot convert it — and when the decoder is unfrozen to adapt, it takes a
   memorization shortcut instead (E2/E3).

This reframes a pile of null F1 numbers into a single coherent mechanism, and it is
falsifiable: it predicts (correctly) that alignment losses sit at chance, that architecture
and data do not move F1, that the signal is retrievable but only as sentence identity, and
that an oracle retrieval baseline beats the model but does not deploy. The constructive
counterpart — retrieval at the right granularity outperforming the model tenfold — is what
turns "we couldn't get it to work" into "here is precisely what the signal supports and what
it does not."

---

## 8. Limitations

- **Single seed (42)** across the frozen-regime sweeps. The nulls are over-determined (dozens
  of cells within one narrow band), but 2–3 seeds at key points would tighten the error bars.
- **Eval-parser ceiling.** The training parser caps perfect output at F1 ≈ 0.745; immaterial
  at ~0 but relevant if a future run produces good output.
- **Per-word objective mis-specification** (E7): the multi-positive formulation's optimum is
  weaker than intended, so per-word alignment was never a fair test of denser supervision. Its
  conclusion is unchanged (it also fails at chance), but this should be disclosed.
- **Oracle retrieval is transductive** (E10a): 0.067/0.094 is a diagnostic ceiling, handed the
  candidate set; the deployable and generalization-relevant number is 0.000.
- **Modality and scale scope.** Our claims are about *ZuCo-scale, word-level, reading EEG*
  through this bridge-plus-frozen-decoder design. They do not imply EEG in general, or
  higher-SNR modalities (MEG, intracranial), cannot carry word-level structure.

---

## 9. Conclusion and future work

We aimed to decode knowledge-graph triplets from reading EEG and instead produced a
thoroughly characterized negative result. Word-level ZuCo EEG carries a weak sentence-identity
signal that reaches the decoder intact but is the wrong kind of information in the wrong
representational geometry for triplet generation, and no architecture, data scale, or
subject-transfer fix changes that. A trivial retrieval baseline at the correct granularity
outperforms the full model by an order of magnitude yet still cannot generalize to unseen
sentences — the sharpest statement of what the signal can and cannot support.

Directions that follow directly from the diagnosis:

- **Match granularity to signal.** Since the decodable content is sentence-level, a
  sentence-level formulation (retrieval-augmented decoding, or classification over a closed
  proposition set) is where any real performance would come from — with the caveat that
  unseen-sentence generalization remains fundamentally limited by the absent word-structure.
- **Close the geometry gap deliberately.** A supervised map (e.g. Procrustes/ridge) from EEG
  states into REBEL's text-state manifold, or a lightly-tuned cross-attention adapter, would
  test whether the *geometry* axis alone is fixable, isolated from the *kind* axis.
- **Higher-SNR or richer inputs.** Whether the word-level structure is decodable at all is a
  modality/SNR question our data cannot settle; MEG or intracranial recordings would.

Beyond the specific finding, the project contributes a transferable methodology for diagnosing
brain-decoding pipelines: calibrate contrastive objectives against their chance floors, and
probe whether signal is *preserved through* a trained network rather than only whether the
end-task metric moves.

---

## References

*(Well-known works cited above; verify exact details and format to the department's citation
style.)*

- Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018).
  *ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.* Scientific
  Data.
- Hollenstein, N., Troendle, M., Zhang, C., & Langer, N. (2020). *ZuCo 2.0: A Dataset of
  Physiological Recordings During Natural Reading and Annotation.* LREC.
- Huguet Cabot, P.-L., & Navigli, R. (2021). *REBEL: Relation Extraction By End-to-end
  Language generation.* Findings of EMNLP.
- Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural
  Language Generation, Translation, and Comprehension.* ACL.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive
  Predictive Coding.* arXiv:1807.03748.

---

## Appendix A — Experiment index

| Exp | RQ | What it establishes |
|---|---|---|
| E1 | RQ2 | Fidelity cliff; the pooled objective saturates before generation-usable fidelity. |
| E2 | RQ3 | Decoder is EEG-sensitive (swap/zero move output) but non-functionally. |
| E3 | RQ3 | Failing runs collapse to memorized, input-independent output. |
| E4 | RQ1 | Signal is sentence identity (~2.6× chance), not word-level structure. |
| E5 | RQ4 | Frozen ceiling is architecture-independent (8-cell grid, spread 0.0083). |
| E6 | RQ4 | Frozen ceiling is data-independent (flat, non-monotone curve). |
| E7 | RQ2 | Alignment losses sit at their chance floor — no text-geometry alignment learned. |
| E8 | RQ5 | Not a cross-subject transfer wall (penalty ≈0, wrong sign). |
| E9 | RQ1/2 | Signal is preserved through the trained bridge/encoder (0.31–0.37 / 0.20 vs 0.149). |
| E10 | RQ1 | Oracle retrieval beats model ~10× but deployable=0; RQ1 replicates on ZuCo 1.0. |

Full writeups, data, and reproduction commands are in `exp_logs/` (see `EXPERIMENTS.md`).
Diagnostic branches: `exp/tier1-conclusions` (E1–E4), `exp/tier2-grid` (E5–E8),
`exp/e9-retrieval` (E9), `exp/e10-baselines` (E10).

## Appendix B — Key numbers at a glance

| Reference (ZuCo 2.0 test) | value |
|---|---|
| chance sentence retrieval (top-10) | 0.149 |
| raw-EEG sentence retrieval (E4) | 0.394 |
| trained bridge retrieval (E9) | 0.31–0.37 |
| encoder-state retrieval (E9) | ≈0.20 |
| text-encoder upper bound (E9) | 0.930 |
| generative model test F1 (best, E5) | 0.0083 |
| oracle retrieve-and-copy F1 (E10) | 0.067 |
| deployable retrieve-and-copy F1 (E10) | 0.000 |
| alignment InfoNCE chance floor, pooled (E7) | 2.865 |
