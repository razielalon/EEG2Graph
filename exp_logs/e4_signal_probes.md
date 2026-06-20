# E4 — Extended EEG signal probes (RQ1: where does the decodable signal live?)

**Question:** Is there decodable linguistic signal in the EEG at all, and if so
*where* does it live — which frequency bands, which channels, and does it expose
word-level linguistic structure or only sentence identity?

**Method:** REBEL-free linear probes (logistic / ridge regression on standardized
features), trained on train and evaluated on the held-out test split. Four
localisation studies on the 840-dim features (8 bands × 105 channels, band-major,
bands `t1 t2 a1 a2 b1 b2 g1 g2`):
**(A)** per-band sentence retrieval, **(B)** per-channel retrieval, **(C)** word-class
probes (content-vs-function, word-length), **(D)** per-subject retrieval.
Retrieval gallery = 67 unique test sentences; `topk` = top-10 hit-rate, **chance =
10/67 = 0.149**. Script: `tests/eeg_signal_probe_extended.py`. Data:
`e4_signal_probes.json`.

## A. Frequency bands — signal is broadband, rising toward high frequency

| Band | top1 | top10 | median rank (/67) |
|---|---|---|---|
| t1 (theta) | 0.039 | 0.278 | 22 |
| t2 | 0.038 | 0.278 | 23 |
| a1 (alpha) | 0.044 | 0.274 | 23 |
| a2 | 0.038 | 0.287 | 22 |
| b1 (beta) | 0.047 | 0.295 | 22 |
| b2 | 0.046 | 0.311 | 20 |
| g1 (gamma) | 0.044 | 0.323 | 20 |
| g2 | 0.055 | 0.333 | 19 |
| **ALL** | **0.065** | **0.394** | **16** |
| *chance* | 0.015 | 0.149 | 34 |

Every band beats chance, and **all 8 bands together (top10 0.394, ~2.6× chance)
beat any single band** — the signal is additive/distributed across bands, not
localised to one. Discriminability rises monotonically with frequency
(theta ≈ 0.278 → gamma2 0.333), consistent with gamma carrying the most
sentence-level information.

## B. Channels — no single channel carries it

Best single channel (top10): ch5 = 0.170, ch99 = 0.160, ch100 = 0.158 — only just
above chance (0.149), and the other ~100 channels sit at chance (0.137–0.157).
No electrode dominates; the retrievable signal is **spatially distributed**, which
is why the full 105-channel probe far exceeds any one channel. (ZuCo ships no
montage here, so channels are by index, not scalp region.)

## C. Word class — the signal is NOT linguistic structure

| Probe | result | baseline | verdict |
|---|---|---|---|
| content vs function | acc **0.696** | majority 0.699 | **at/below chance** — no signal |
| word length (R²) | **−0.018** | 0 | negative — no signal |

The probes that decode sentence identity **cannot** decode content-vs-function or
word length above the trivial baseline. So the decodable EEG signal is a
**sentence-level fingerprint, not the word-level linguistic structure** the
triplet task actually needs.

## D. Subjects — large inter-subject variance

Per-subject top10 ranges **0.31 (YHS) → 0.56 (YRK)** — best subjects
(YRK 0.557, YAC 0.532, YMD 0.472) are ~1.8× the weakest. All 18 subjects beat
chance, but signal quality is highly subject-dependent. **Motivates E8**
(cross-subject / held-out-subject study).

## Conclusion

EEG2Graph's input is **not noise**: sentence retrieval reaches ~2.6× chance
(median rank 16/67), broadband and spatially distributed, strongest in gamma.
**But** that signal is sentence-identity, not the word-class / word-length
structure the task requires — and it varies ~1.8× across subjects. This is the
complement to E1's fidelity cliff: signal genuinely exists, just not in the
linearly-accessible, per-position form the decoder needs. The bottleneck is
representational alignment, not absence of signal.
