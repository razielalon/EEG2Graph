# E7 — Per-word alignment, and the alignment chance floor: the bridge never learns anything

**Status: no new runs.** E7's hypothesis was already tested — the per-word alignment
implementation from `exp/align-perword` was ported into `exp/tier2-grid` as
`--align_mode perword`, so **E5 cells 5–8 *are* E7**. This writeup pulls those four cells
out and, in checking them, establishes a result that reframes E5, E6 and E8. The E5
writeup and its data are left untouched.

**The original question (RQ2 follow-up):** E1 located a fidelity cliff and E3 found
input-independent prediction collapse. The proposed fix was denser alignment supervision:
the pooled InfoNCE gives the bridge one gradient per *sentence*, so maybe the bridge is
simply starved of signal. Per-position InfoNCE
(`contrastive_alignment_perword`, train.py:133) contrasts every EEG word position against
every text token position — a far denser objective. If the bridge was signal-starved,
this should show up as a better alignment and, downstream, some decodable output.

## The result that was already sitting in E5

| E5 cell | subject | temporal | align | test F1 | n_correct / n_gold | align: start → final |
|---|---|---|---|---|---|---|
| 5 (`s0t0aPW`) | off | off | perword | 0.0000 | **0 / 1936** | 9.17 → 6.2145 |
| 6 (`s1t0aPW`) | on  | off | perword | 0.0000 | **0 / 1936** | 9.07 → 6.2140 |
| 7 (`s0t1aPW`) | off | on  | perword | 0.0000 | **0 / 1936** | 7.65 → 6.1751 |
| 8 (`s1t1aPW`) | on  | on  | perword | 0.0000 | **0 / 1936** | 9.63 → 6.1555 |

All four per-word cells produce **exactly zero correct triplets** — not "near floor," a
clean zero, and strictly worse than the pooled cells (which manage 6–15 correct). The
models are not silent: they emit 1375–2315 predicted triplets. They are simply never
right.

The tempting read is: *"the alignment loss falls 9.17 → 6.21, so the objective is being
optimized — the bridge learns a well-aligned representation and the frozen decoder
ignores it."* **That read is wrong, and checking why is the actual finding.**

## The alignment losses never beat chance — in any run, in any experiment

InfoNCE has a **chance floor**: the loss produced by a model that has learned *nothing* is
not 0, it is `ln(N)` over N candidates. Nobody had ever computed it for these objectives,
so every "the align loss went down" claim in E5/E6/E8 was being read against an implicit
floor of zero. `model/align_chance_floor.py` measures the real floors by pushing random,
uninformative EEG states through the exact loss functions `train.py` uses (B=16, d=1024,
temp 0.07, 20 trials):

| objective | chance (random EEG) | attainable optimum | observed in E5/E6/E8 |
|---|---|---|---|
| **pooled** (`contrastive_alignment_loss`) | **2.865 ± 0.122** | **0.000** (EEG == text) | 2.848 – 3.406 |
| **per-word** (`contrastive_alignment_perword`) | **6.182 ± 0.005** | **4.137** (sentence-mean); ln(B)=2.77 analytic | 6.156 – 6.215 |

**Every pooled run in the entire series sits at or above its chance floor.** All four E5
pooled cells, all four E6 data-scaling points, all seven E8 cross-subject runs — 15 runs,
none below 2.865 by more than noise (the single nominal exception, E5 `s0t1aPL` at 2.8478,
is 0.1 sd below chance). The pooled objective's optimum is **0.0**, so the bridge captures
**none** of the available headroom. It cannot pick the correct sentence's text embedding
out of a batch of 16 better than a coin flip.

**Per-word is the same story.** Its floor is 6.182 and the four cells land at 6.2145,
6.2140, 6.1751, 6.1555 — i.e. *at chance*. So the celebrated "9.17 → 6.21 descent" is a
model falling from random-initialization garbage down to the chance floor and stopping
dead. Cell 8 (`s1t1aPW`, 6.1555) is 5.9 sd below chance and so is technically
distinguishable from it — but the per-word loss averages over hundreds of anchors, so its
sd is tiny (0.005) and significance is cheap. In magnitude it captures **0.026 nats of a
~2.0-nat headroom, about 1%** — and it still decodes **0 / 1936** triplets.

*(Aside, and a real design flaw in the per-word objective: because it is **multi-positive**
— every anchor must spread softmax mass across all ~30 token positions of its own sentence
— an exactly-aligned EEG state, copied verbatim from the text state, scores **13.12, worse
than chance**. It puts all its mass on the one twin token and the other 29 positives punish
it. The objective's optimum is not "match your word"; it is "be equally similar to every
word in your sentence," which is a *weaker* target than the one E7 set out to optimize. So
per-word alignment does not even ask for what it was supposed to ask for.)*

## What this means

**1. The failure is upstream, not downstream.** The story E5/E6/E8 told was: *the bridge
learns EEG codes; the frozen decoder won't act on them.* The truth is that **there are no
codes.** The EEG encoding carries no recoverable sentence identity at all — not enough to
win a 1-in-16 discrimination against the very text encoder it is being trained to match.
F1 ≈ 0 downstream is then not a mystery to be explained by architecture, data scale, or
subject transfer; it is the expected consequence of a bridge that never learned anything.

**2. E7's hypothesis is dead, and it died informatively.** Denser supervision was the
proposed fix for the E1 cliff. It was implemented, it was run four ways, and it lands
precisely on its own chance floor while producing strictly zero correct triplets. The
bridge is not signal-*starved*; there is no signal to be dense about. **Do not run E7 as a
separate experiment** — it is already answered, and the answer is negative.

**3. It rehabilitates E5/E6/E8 as a *measurement*, not just three null sweeps.** The fair
criticism of those experiments is that all arms read ~0 test F1, so they have no
resolution — "this lever doesn't matter" is indistinguishable from "the instrument sees
nothing." The chance-floor analysis converts that weakness into the point: the instrument
sees nothing **because the bridge learned nothing**, and now we can say so against a hard
reference value rather than by staring at an F1 of 0.008. This is a *quantified,
mechanistic* negative result, and unlike the F1 sweeps it has a well-defined scale.

**4. It converges with E1.** A bridge that cannot beat chance at 1-in-16 sentence
discrimination is exactly what E1's noise-interpolation fidelity cliff predicts. Two
independent measurements, one on the input side and one on the representation side, agree
that word-level ZuCo EEG does not carry recoverable sentence-level content through this
bridge.

## Corrections issued

This finding invalidates a claim made in two already-committed writeups, both of which
read a falling alignment loss as learning without checking the floor. Both have been
corrected in place:

- **`e6_data_scaling_writeup.md`** — claimed the loss falling 3.41 → 2.88 with more data
  showed "the bridge is demonstrably learning a better-aligned encoder representation."
  It shows the loss converging *toward chance* (2.865) from above.
- **`e8_cross_subject_writeup.md`** — claimed the loss rising 2.87 → 3.15 as subjects were
  removed showed "the bridge is measurably learning *less* with fewer people." Every one
  of those values is at or above chance; there was no learned state to degrade from.

Neither correction changes either experiment's conclusion (both were null results and
remain null); what changes is *why* they are null, and the corrected reason is stronger.

## Where this points

The open question is now sharp and, for the first time in this series, **has a metric with
dynamic range**: does the bridge encoding contain *any* recoverable information about the
sentence? The alignment loss says no at batch-16 discrimination. The direct test is a
**retrieval eval** — rank the gold sentence among N candidates by bridge-encoding
similarity, sweep N, and report top-k accuracy against the 1/N chance line. It runs on the
checkpoints we already have (E5, E6, E8: 19 of them), needs no retraining and no cluster
time, and unlike triplet F1 it cannot be pinned at zero by the exact-match requirement. If
retrieval is at chance too, the negative result is complete and airtight. If it is
*above* chance, then information is present but unusable by the decoder — and that is a
different thesis, with a different fix.

## Caveats

- The chance floors are measured with **random EEG states** against real text states,
  which is the right reference for "learned nothing." They are reported with the sd across
  20 trials; the pooled sd (0.122) is much larger than the per-word sd (0.005) because the
  pooled loss averages over only B=16 anchors per batch versus hundreds for per-word.
  Statistical significance below the per-word floor is therefore cheap, and magnitude
  (fraction of headroom captured) is the honest yardstick — hence the "1%" framing above.
- `n_text_tokens` is set to 30 in the floor measurement (a typical ZuCo sentence's BPE
  length). The per-word floor is `ln(B · S_t)` and so depends on it: the measured 6.182 at
  S_t=30 brackets the observed 6.156–6.215 tightly, which is itself corroboration that the
  runs are sitting at the floor rather than near it by coincidence.
- All observed values are `align_final` (last epoch) and `align_at_best` from the E5/E6/E8
  aggregates. A run could in principle dip below chance mid-training and return; the
  per-epoch `history.json` on the cluster would settle that, and is worth a look if anyone
  wants to contest the "never" in this writeup's title.
