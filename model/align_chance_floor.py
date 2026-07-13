#!/usr/bin/env python
"""
Measure the CHANCE FLOOR of the two alignment objectives.

Why this exists: the alignment loss values reported by E5/E6/E8 (pooled ~2.85-3.15,
per-word ~6.16-6.21) were read as "the bridge is learning a better-aligned
representation." That reading is only meaningful against the loss's floor -- the
value a model that has learned NOTHING produces. InfoNCE over N candidates has a
chance floor of ln(N), which is not 0.

This script measures both floors directly by feeding UNINFORMATIVE representations
(random EEG states, independent of the text states) through the exact loss functions
train.py uses, and also measures the ceiling (perfectly aligned states) to confirm
the metric has dynamic range at all.

Read-off:
    chance  = loss for random/independent EEG states  (~ln(B) pooled, ~ln(N_t) per-word)
    perfect = loss for EEG states == text states       (~0)
    A trained run whose align loss sits AT chance has learned nothing, however far
    it fell from its random-init starting point.

Usage:
    python align_chance_floor.py                     # defaults match E5/E6/E8
    python align_chance_floor.py --batch_size 16 --trials 20
"""
import argparse

import torch

from train import contrastive_alignment_loss, contrastive_alignment_perword

# Observed align_final across the experiment series, for the read-off table.
OBSERVED = {
    "pooled": [
        ("E5 s0t0aPL", 2.8732), ("E5 s1t0aPL", 2.8803),
        ("E5 s0t1aPL", 2.8478), ("E5 s1t1aPL", 2.9092),
        ("E6 frac 0.10", 3.4062), ("E6 frac 0.25", 3.0738),
        ("E6 frac 0.50", 3.0019), ("E6 frac 1.00", 2.8803),
        ("E8 k0", 2.8732), ("E8 x03", 3.0448), ("E8 x06", 2.9810),
        ("E8 x09", 3.1532), ("E8 c03", 2.9487), ("E8 c06", 3.0219),
        ("E8 c09", 3.0820),
    ],
    "perword": [
        ("E5 s0t0aPW", 6.2145), ("E5 s1t0aPW", 6.2140),
        ("E5 s0t1aPW", 6.1751), ("E5 s1t1aPW", 6.1555),
    ],
}


def run(args):
    torch.manual_seed(args.seed)
    B, D = args.batch_size, args.d_model
    S_e, S_t = args.n_words, args.n_tokens
    dev = torch.device("cpu")

    # Distinct sentence texts => no same-text masking, the clean reference case.
    texts = [f"sentence number {i}" for i in range(B)]
    eeg_mask = torch.ones(B, S_e, dtype=torch.long)
    text_mask = torch.ones(B, S_t, dtype=torch.long)

    pooled_chance, perword_chance = [], []
    pooled_perfect, perword_perfect = [], []

    for _ in range(args.trials):
        h_text = torch.randn(B, S_t, D, device=dev)
        # CHANCE: EEG states drawn independently of the text states -- i.e. a bridge
        # that has learned nothing about this sentence.
        h_eeg_rand = torch.randn(B, S_e, D, device=dev)
        pooled_chance.append(
            contrastive_alignment_loss(h_eeg_rand, eeg_mask, h_text, text_mask,
                                       texts, args.temperature).item())
        perword_chance.append(
            contrastive_alignment_perword(h_eeg_rand, eeg_mask, h_text, text_mask,
                                          texts, args.temperature).item())

        # BEST-CASE for POOLED: EEG states copied from the text states. The pooled
        # loss is single-positive, so an exact copy IS its optimum (-> ~0).
        reps = (S_e + S_t - 1) // S_t
        h_eeg_perf = h_text.repeat(1, reps, 1)[:, :S_e, :]
        pooled_perfect.append(
            contrastive_alignment_loss(h_eeg_perf, eeg_mask, h_text, text_mask,
                                       texts, args.temperature).item())

        # BEST-CASE for PER-WORD is NOT the exact copy. The per-word loss is
        # MULTI-positive: every anchor must spread softmax mass over ALL S_t tokens
        # of its sentence. An exact copy puts ~all mass on the one twin token, which
        # the other S_t-1 positives are then penalized for -- so copying scores WORSE
        # than chance. The optimum is a representation equally similar to all of its
        # own sentence's tokens and dissimilar to every other sentence's: i.e. every
        # EEG position of sentence i collapses onto that sentence's mean text vector.
        # Its analytic value is ln(N_t / n_pos) = ln(B*S_t / S_t) = ln(B).
        sent_mean = torch.nn.functional.normalize(h_text.mean(dim=1), dim=-1)
        h_eeg_opt = sent_mean.unsqueeze(1).expand(B, S_e, D).contiguous()
        perword_perfect.append(
            contrastive_alignment_perword(h_eeg_opt, eeg_mask, h_text, text_mask,
                                          texts, args.temperature).item())

    import math
    import statistics as st
    mean = lambda xs: sum(xs) / len(xs)
    ln_B = math.log(B)
    ln_Nt = math.log(B * S_t)

    p_chance, p_sd = mean(pooled_chance), st.pstdev(pooled_chance)
    w_chance, w_sd = mean(perword_chance), st.pstdev(perword_chance)

    print(f"Config: B={B}  d_model={D}  n_words={S_e}  n_text_tokens={S_t}  "
          f"temp={args.temperature}  trials={args.trials}\n")

    print("POOLED InfoNCE  (contrastive_alignment_loss)  -- single-positive")
    print(f"  analytic chance   ln(B)           = {ln_B:.4f}")
    print(f"  MEASURED chance (random EEG)      = {p_chance:.4f} +/- {p_sd:.4f}")
    print(f"  MEASURED optimum (EEG == text)    = {mean(pooled_perfect):.4f}")

    print("\nPER-WORD InfoNCE  (contrastive_alignment_perword)  -- MULTI-positive")
    print(f"  analytic chance   ln(B*S_t)       = {ln_Nt:.4f}")
    print(f"  MEASURED chance (random EEG)      = {w_chance:.4f} +/- {w_sd:.4f}")
    print(f"  analytic optimum  ln(B)           = {ln_B:.4f}   "
          f"(spread mass over all S_t own-sentence positives)")
    print(f"  MEASURED optimum (sentence-mean)  = {mean(perword_perfect):.4f}")
    print("  NOTE: an exact EEG==text copy scores WORSE than chance here -- it puts")
    print("  all mass on one twin token, and the other S_t-1 positives punish that.")

    print("\n" + "=" * 68)
    print("READ-OFF vs the experiment series (align_final):")
    for mode, floor, sd in (("pooled", p_chance, p_sd),
                            ("perword", w_chance, w_sd)):
        print(f"\n  {mode}  (measured chance = {floor:.4f} +/- {sd:.4f})")
        for name, val in OBSERVED[mode]:
            z = (val - floor) / sd if sd > 0 else 0.0
            if val >= floor:
                verdict = "at/above chance -> learned NOTHING"
            elif z < -3:
                verdict = f"below chance by {abs(z):.1f} sd -> learned something"
            else:
                verdict = f"below chance by only {abs(z):.1f} sd -> within noise"
            print(f"    {name:16s} {val:.4f}   {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16, help="E5/E6/E8 used 16.")
    ap.add_argument("--d_model", type=int, default=1024, help="BART-large d_model.")
    ap.add_argument("--n_words", type=int, default=25, help="EEG word positions.")
    ap.add_argument("--n_tokens", type=int, default=30,
                    help="Valid text token positions per sentence.")
    ap.add_argument("--temperature", type=float, default=0.07, help="--align_temp.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
