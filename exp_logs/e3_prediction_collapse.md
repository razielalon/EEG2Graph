# E3 — Prediction-collapse analysis (RQ3: anatomy of the decoder shortcut)

**Question:** When an EEG run scores ~0, *how* is it failing? Is the decoder
emitting varied-but-wrong triplets, or has it collapsed to a (near-)constant,
input-independent answer that ignores the EEG and parrots the training labels?

**Method:** Post-hoc analysis of each run's saved `test_results.json`
(`[{"gold": [...], "pred": [...]}, ...]`) — no GPU, no model load. For every run
we measure prediction diversity (`unique_pred_rate`, `top1_share`,
`pred_entropy_bits`), output shape (`empty_rate`, `mean_pred_triplets`), relation
token diversity (`distinct1/2`), and `train_memorization` = fraction of predicted
triplet-tuples that appear verbatim in the TRAIN gold set.
Script: `model/prediction_analysis.py`. Data: `e3_prediction_collapse.json`.
Runs scored: the two with completed test predictions on the cluster
(`decoder_noise`, `subject_temporal`); n=1246 test samples each.

## Results

| Metric | decoder_noise | subject_temporal | reading |
|---|---|---|---|
| `unique_pred_rate`  | **0.0024** | 0.0177 | ~3 vs ~22 distinct outputs over 1246 inputs |
| `top1_share`        | **0.860**  | 0.326  | one triplet = 86% of all predictions vs 33% |
| `pred_entropy_bits` | **0.706**  | 2.941  | near-degenerate vs ~4× more diverse |
| `empty_rate`        | 0.000      | 0.000  | both always emit something |
| `mean_pred_triplets`| 1.098      | 1.805  | — |
| `distinct1`         | 0.0024     | 0.0036 | — |
| `distinct2`         | 0.0018     | 0.0046 | — |
| `train_memorization`| **1.000**  | 0.185  | 100% vs 19% of output seen in TRAIN gold |

## Two conclusions

**1. `decoder_noise` is the decoder shortcut, fully quantified.** 86% of all 1246
test predictions are the *same single triplet*, only ~3 distinct outputs exist
across the entire test set (entropy 0.71 bits), and **100% of what it emits is a
triplet that appears verbatim in the training set**. This is the textbook
shortcut — the decoder has stopped conditioning on the EEG and emits one constant,
memorised-from-train sequence. The ~0 F1 isn't "varied but wrong"; it's
input-independent regurgitation.

**2. The subject+temporal architecture is markedly less collapsed.** Same dataset,
same test set, but `subject_temporal` is ~4× more diverse (entropy 2.94 vs 0.71),
its top output covers only a third of samples (0.33 vs 0.86), and only 19% of its
output echoes train (vs 100%). The architecture change measurably moves the model
*away* from the constant-output failure mode — a real, reportable contrast.

## Caveat — E3 is correlational, E2 is the causal test

Higher diversity is **not** proof that `subject_temporal` reads the EEG: it could
just have a higher-entropy input-independent prior. The causal confirmation is
**E2** (`model/eeg_ablation_eval.py`) — real vs swapped vs zeroed EEG. If swap/zero
leave the output unchanged, the decoder ignores the EEG *regardless* of how diverse
its predictions look. `decoder_noise`'s 100% train-memorization + 86% top1-share
already predicts real ≈ swap ≈ zero for that run; E2 must confirm whether
`subject_temporal`'s extra diversity is actually EEG-driven.
