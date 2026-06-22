# E2 — EEG-ablation eval (RQ3: does the decoder causally use the EEG?)

**Question:** E3 showed *correlationally* that the failing runs emit low-diversity,
train-memorised output. It could not say whether the decoder actually conditions on
the EEG. E2 is the causal test: regenerate each checkpoint's test predictions under
three input conditions and measure how much the output moves.

- **real** — the true EEG for each sentence.
- **swap** — each sentence decoded from *another* sentence's real EEG (a derangement, no fixed points).
- **zero** — all-zero EEG (attention mask unchanged).

If the decoder ignores the EEG, swap/zero leave the output unchanged
(`identical_to_real ≈ 1`) and F1 doesn't move. If it genuinely routes EEG through
cross-attention, swap/zero *change* the output. For the subject-conditioned model,
`subject_idx` is held to the **output slot** in all three conditions, so only the
EEG varies — the subject embedding is never the thing being ablated.

**Method:** inference-only, greedy (`beam_size=1`), n=1246 test samples, on each
checkpoint's own branch (`exp/decoder-noise`, `exp/subject-temporal`).
Reported per condition: triplet F1 and `identical_to_real` = fraction of generated
token sequences byte-identical to the real-EEG run. Script:
`model/eeg_ablation_eval.py`. Data: `e2_eeg_ablation.json`.

## Results

| Run | cond | F1 | n_pred | identical_to_real | **output changed** |
|---|---|---|---|---|---|
| decoder_noise | real | 0.000 | 1246 | 1.000 | — |
| decoder_noise | swap | 0.000 | 1246 | 0.435 | **56.5%** |
| decoder_noise | zero | 0.000 | 1246 | 0.461 | **53.9%** |
| subject_temporal | real | 0.000 | 1566 | 1.000 | — |
| subject_temporal | swap | 0.000 | 1559 | 0.072 | **92.8%** |
| subject_temporal | zero | 0.000 | 1246 | 0.000 | **100%** |

`n_correct = 0` in all six conditions; `n_gold = 1936`.

## Three conclusions

**1. Neither model is taking the naive "ignore the EEG" shortcut.** Both are
EEG-sensitive: swapping the EEG flips 56.5% of `decoder_noise`'s outputs and 92.8%
of `subject_temporal`'s; zeroing flips 53.9% and 100%. The decoder *does* condition
on the EEG — strongly, for the subject+temporal architecture. This sharpens E3's
"constant, input-independent output" reading: the output space is low-diversity
(E3), but *which* of the few sequences is emitted is genuinely EEG-selected.

**2. The EEG-sensitivity is non-functional.** Despite all that input-dependence,
**F1 is exactly 0 in every condition** — real EEG produces no more correct triplets
than random (swap) or absent (zero) EEG. The model never lands on a correct triplet
under any input. So the bottleneck is **not** a decoder that ignores the EEG; it is
that the EEG-driven variation is **not aligned with the correct triplets**. More
EEG-sensitivity (subject_temporal) buys exactly zero accuracy. This complements E4
(the decodable EEG signal is sentence-identity, not the word-level structure the
task needs) and E1 (the fidelity cliff).

**3. E2 resolves E3's open caveat about subject_temporal.** E3 warned that
subject_temporal's higher prediction diversity was *not* proof it reads the EEG — it
could be a higher-entropy input-independent prior. E2 settles it causally: 92.8% of
its outputs change under EEG swap (vs 56.5% for decoder_noise), and **100% change
under zero**, where it collapses to exactly one triplet per sample (`n_pred` 1566 →
1246). The subject+temporal bridge makes the model strongly EEG-coupled — its
diversity is genuinely EEG-driven, just not toward correct answers.

## Caveat — the F1-delta test is floored

E2 was designed assuming real F1 > 0, so swap/zero could *drop* it. Both checkpoints
score real F1 = 0 (and trained to `val_f1 = 0.0`), so the F1 comparison is
uninformative — it cannot go below zero. **The causal evidence here rests entirely on
`identical_to_real` (sequence-change rate) and `n_pred` (output shape), not on F1.**
A future E2 on a checkpoint with non-zero real F1 (if one is ever obtained) would let
the F1-delta carry the conclusion directly. Greedy decoding (`beam_size=1`) is used
here; E3's diversity stats came from the training-time `test_results.json`
(`beam_size=4`), so absolute `n_pred`/diversity figures are not directly comparable
across E2 and E3 — only the qualitative contrast is.
