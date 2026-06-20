"""
E2: EEG-ablation at inference (RQ3 — does the decoder actually use the EEG?)

Generate from a trained checkpoint under three input conditions and compare:

  real    : the true EEG for each sentence
  swap    : each sentence decoded from ANOTHER sentence's (real) EEG
  zero    : all-zero EEG (attention mask unchanged)

If the decoder has taken the shortcut, its output is EEG-independent: swap and
zero produce (near-)identical token sequences to `real`, and F1 barely moves.
If the bridge genuinely routes signal through cross-attention (the hoped-for
frozen-BART behaviour), swap/zero should *change* the output and drop F1 below
the real-EEG score.

Reported per condition: triplet F1, and "identical_to_real" = fraction of
samples whose generated token sequence is byte-identical to the real-EEG run.

Usage (cluster GPU, or locally if a checkpoint is copied here):
    python eeg_ablation_eval.py \
        --checkpoint ../checkpoints_baseline/best_model.pt \
        --tokenizer_dir ../checkpoints_baseline/tokenizer \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --split test --beam_size 1 \
        --name baseline --out ../exp_logs/e2_eeg_ablation.json
"""

import argparse
import json
import os

import numpy as np
import torch

from vocabulary import load_tokenizer, delinearize
from eeg_graph_dataset import fixation_attention_mask
from inference import load_model
from train import compute_triplet_f1


def load_split(processed_dir, triplets_path, split):
    """EEG arrays, per-sample fixation masks, and gold triplets (joined by text)."""
    eeg = np.load(os.path.join(processed_dir, f"{split}_eeg.npy"), allow_pickle=True)
    meta = json.load(open(os.path.join(processed_dir, f"{split}_meta.json")))
    raw = json.load(open(triplets_path))
    if isinstance(raw, dict):
        index = {t.strip(): e.get("triplets", []) for t, e in raw.items()}
    else:
        index = {e["text"].strip(): e.get("triplets", []) for e in raw}
    golds = [index.get(m["text"].strip(), []) for m in meta]
    return eeg, meta, golds


@torch.no_grad()
def generate_condition(model, eeg, meta, tokenizer, device, perm, mode,
                       num_beams, max_len, batch_size):
    """Generate token-id sequences for every sample under one ablation mode.

    perm maps output-slot i -> EEG-source index (identity for `real`/`zero`,
    a derangement for `swap`). mode in {real, swap, zero}.
    """
    seqs, preds = [], []
    N = len(eeg)
    for start in range(0, N, batch_size):
        sl = list(range(start, min(start + batch_size, N)))
        srcs = [np.asarray(eeg[perm[i]], dtype=np.float32) for i in sl]
        max_src = max(s.shape[0] for s in srcs)
        feat = srcs[0].shape[1]
        src = torch.zeros(len(sl), max_src, feat, device=device)
        real_mask = torch.zeros(len(sl), max_src, dtype=torch.bool, device=device)
        fix_mask = torch.zeros(len(sl), max_src, dtype=torch.bool, device=device)
        for r, i in enumerate(sl):
            s = srcs[r]
            n = s.shape[0]
            if mode != "zero":
                src[r, :n] = torch.tensor(s)
            real_mask[r, :n] = True
            hf = meta[perm[i]].get("has_fixation", [True] * n)[:n]
            fix_mask[r, :len(hf)] = torch.tensor(hf, dtype=torch.bool, device=device)
        src_mask = fixation_attention_mask(real_mask, fix_mask)
        gen = model.generate(src, src_mask, max_len=max_len, num_beams=num_beams)
        for r in range(gen.size(0)):
            ids = gen[r].cpu().tolist()
            seqs.append(tuple(ids))
            preds.append(delinearize(ids, tokenizer))
    return seqs, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--triplets_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="run")
    p.add_argument("--out", default="../exp_logs/e2_eeg_ablation.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model(args.checkpoint, tokenizer, device)

    eeg, meta, golds = load_split(args.processed_dir, args.triplets_path, args.split)
    N = len(eeg) if args.limit <= 0 else min(args.limit, len(eeg))
    eeg, meta, golds = eeg[:N], meta[:N], golds[:N]
    print(f"{N} {args.split} samples")

    rng = np.random.default_rng(args.seed)
    identity = list(range(N))
    swap = rng.permutation(N)
    # avoid accidental fixed points so every sample really gets a different EEG
    for i in range(N):
        if swap[i] == i:
            swap[i] = (swap[i] + 1) % N
    num_beams = max(1, args.beam_size)
    nb = dict(num_beams=num_beams, max_len=args.max_len, batch_size=args.batch_size)

    print("  generating: real ...")
    real_seqs, real_preds = generate_condition(
        model, eeg, meta, tokenizer, device, identity, "real", **nb)
    print("  generating: swap ...")
    swap_seqs, swap_preds = generate_condition(
        model, eeg, meta, tokenizer, device, swap, "swap", **nb)
    print("  generating: zero ...")
    zero_seqs, zero_preds = generate_condition(
        model, eeg, meta, tokenizer, device, identity, "zero", **nb)

    def ident_rate(seqs):
        return float(np.mean([a == b for a, b in zip(seqs, real_seqs)]))

    report = {
        "name": args.name,
        "split": args.split,
        "n_samples": N,
        "num_beams": num_beams,
        "real": {**compute_triplet_f1(real_preds, golds), "identical_to_real": 1.0},
        "swap": {**compute_triplet_f1(swap_preds, golds),
                 "identical_to_real": ident_rate(swap_seqs)},
        "zero": {**compute_triplet_f1(zero_preds, golds),
                 "identical_to_real": ident_rate(zero_seqs)},
    }
    for cond in ("real", "swap", "zero"):
        r = report[cond]
        print(f"  {cond}: F1={r['f1']:.4f}  identical_to_real={r['identical_to_real']:.3f}")

    # merge into a multi-run file keyed by name
    out = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[args.name] = report
    json.dump(allr, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
