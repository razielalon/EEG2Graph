# E1 — Text-teacher upper bound + noise-interpolation curve (RQ2)

**Question:** How faithfully would an EEG encoder have to reproduce REBEL's text
representation for the decoder to emit correct triplets?

**Method:** Take REBEL's own encoder states for the 67 unique test sentences and
interpolate toward matched Gaussian noise, `states(α) = α·text + (1−α)·noise`,
decoding triplets at each α with the gold `GEN_KWARGS` (beams=5, len=256).
Score with both the gold-label parser (`extract`, faithful) and the train-eval
parser (`delinearize`). Alignment is the InfoNCE retrieval@1 the bridge optimizes.
Script: `model/noise_interpolation.py`. Data: `e1_noise_interpolation.json`,
figure `e1_noise_interpolation.png`.

## Results

| α | retrieval@1 | pooled cos | F1 (faithful) | F1 (train-eval) |
|----|----|----|----|----|
| 1.00 | 1.000 | +1.000 | **1.000** | 0.745 |
| 0.95 | 1.000 | +0.999 | 0.934 | 0.698 |
| 0.90 | 1.000 | +0.998 | 0.768 | 0.569 |
| 0.80 | 1.000 | +0.990 | 0.624 | 0.488 |
| 0.70 | 1.000 | +0.975 | 0.475 | 0.386 |
| 0.65 | 1.000 | +0.965 | 0.270 | 0.240 |
| 0.60 | 1.000 | +0.953 | 0.068 | 0.068 |
| 0.50 | 0.985 | +0.923 | **0.000** | 0.000 |
| 0.30 | 0.851 | +0.841 | 0.000 | 0.000 |
| 0.00 | 0.015 | +0.677 | 0.000 | 0.000 |

## Two conclusions

**1. The eval harness has a measurement ceiling of ~0.75, not 1.0.** Perfect
text states (α=1) score F1=1.0 under the parser that *made* the gold labels, but
only **0.745** under `delinearize`, the parser `train.py:evaluate()` grades every
experiment with. `delinearize` leaks structural tokens into the relation field on
nested/duplicate triplets. Every reported experiment F1 is therefore measured
against a harness that cannot score even flawless output above ~0.75 — runs are
undervalued by this gap. (Generation itself is faithful: with the gold GEN_KWARGS,
output reproduces the stored `raw_rebel_output` byte-for-byte.)

**2. The fidelity cliff — and why the bridge's training objective can't see it.**
F1 falls from 1.0 → 0 as α goes 1.0 → ~0.55, i.e. generation needs the encoder
to reproduce REBEL's per-position states almost exactly (α ≥ 0.9 for F1 > 0.75).
**But the pooled InfoNCE retrieval@1 the alignment loss optimizes stays at 1.0
across that entire collapse, and is still 0.985 at α=0.5 where F1 is already 0.**
The objective the bridge is trained on (pooled-direction alignment) is satisfied
long before the per-position fidelity generation actually requires. An EEG encoder
can therefore look perfectly aligned by the contrastive metric and still produce
zero correct triplets. This quantitatively explains the ~0 F1 of every EEG run and
predicts that *strengthening pooled alignment cannot fix it* — the alignment target
must become per-position (cf. E7 `exp/align-perword`) or the task must be made
easier (E10 difficulty ladder).
