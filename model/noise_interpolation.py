"""
E1: Text-teacher upper bound + noise-interpolation curve
========================================================

The centerpiece RQ2 experiment: *how good would an EEG encoder need to be
for REBEL to emit correct triplets?*

We never touch the EEG here. Instead we take REBEL's OWN encoder states for
the gold sentence text — the perfect target the alignment loss tries to make
the bridge reproduce — and progressively corrupt them:

    states(alpha) = alpha * text_states + (1 - alpha) * noise

At alpha=1 the decoder sees the real text representation (upper bound); at
alpha=0 it sees pure noise. Sweeping alpha traces F1 as a function of *how
faithfully an encoder reproduces REBEL's text manifold*, measured by the same
InfoNCE retrieval accuracy the alignment loss optimizes.

Two parsers are scored at every alpha:
  - `extract` (extract_triplets_from_rebel_output) — the parser that CREATED
    the gold labels in generateTriplets/. alpha=1 reproduces ~1.0 F1, so this
    is the honest encoder-fidelity axis.
  - `delinearize` — the parser train.py's evaluate() grades every experiment
    with. It is lossier (leaks structural tokens into the relation field on
    nested/duplicate triplets), so even perfect text states cap below 1.0.
    The gap between the two curves is the harness measurement artifact.

Generation matches generateTriplets GEN_KWARGS (num_beams=5, max_length=256,
length_penalty=1.0) so alpha=1 reproduces the gold byte-for-byte.

Usage (from model/):
    python noise_interpolation.py \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --split test --out ../exp_logs/e1_noise_interpolation.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from vocabulary import build_tokenizer, delinearize
from eeg_graph_model import EEGBartModel
from train import compute_triplet_f1

# The parser that produced the gold labels — the faithful round-trip scorer.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generateTriplets"))
from generate_triplets import extract_triplets_from_rebel_output  # noqa: E402


def load_unique_sentences(processed_dir, triplets_path, split):
    """Return (texts, gold_triplets) deduplicated by sentence text."""
    with open(os.path.join(processed_dir, f"{split}_meta.json")) as f:
        meta = json.load(f)
    with open(triplets_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        index = {t.strip(): e.get("triplets", []) for t, e in raw.items()}
    else:
        index = {e["text"].strip(): e.get("triplets", []) for e in raw}

    seen, texts, golds = set(), [], []
    for m in meta:
        t = m["text"].strip()
        if t in seen:
            continue
        seen.add(t)
        if t in index:
            texts.append(t)
            golds.append(index[t])
    return texts, golds


def _masked_mean(h, mask):
    """Mean-pool (B, L, D) over valid positions — mirrors train._masked_mean."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def alignment_metrics(states, ref_states, mask):
    """Two views of how aligned `states` are to the text `ref_states`.

    - cosine: mean per-sentence cosine of pooled vectors (what the InfoNCE
      numerator rewards; inflated by the shared mean, never reaches 0).
    - retrieval@1: fraction of sentences whose pooled state retrieves its OWN
      text state as nearest neighbour among all sentences. This is the InfoNCE
      *accuracy* — spans [1/N, 1.0] — and is the primary fidelity axis.
    """
    e = F.normalize(_masked_mean(states, mask), dim=-1)
    t = F.normalize(_masked_mean(ref_states, mask), dim=-1)
    cosine = (e * t).sum(dim=-1).mean().item()
    sim = e @ t.t()
    nn = sim.argmax(dim=1)
    retrieval = (nn == torch.arange(e.size(0))).float().mean().item()
    return cosine, retrieval


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, max_text_len, batch_size):
    """REBEL encoder states + attention masks for a list of sentences."""
    all_states, all_masks = [], []
    d_model = model.bart.config.d_model
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_text_len)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        states = model.encode_text(ids, mask.bool())  # (b, L, D), detached
        all_states.append(states.cpu())
        all_masks.append(mask.cpu())
    Lmax = max(s.shape[1] for s in all_states)
    N = len(texts)
    states = torch.zeros(N, Lmax, d_model)
    masks = torch.zeros(N, Lmax, dtype=torch.long)
    row = 0
    for s, m in zip(all_states, all_masks):
        b, L = s.shape[0], s.shape[1]
        states[row:row + b, :L] = s
        masks[row:row + b, :L] = m
        row += b
    return states, masks


@torch.no_grad()
def generate_from_states(model, states, masks, device, gen_kwargs, batch_size):
    """Decode token IDs from arbitrary encoder states (the injection point)."""
    out = []
    for start in range(0, states.shape[0], batch_size):
        s = states[start:start + batch_size].to(device)
        m = masks[start:start + batch_size].to(device)
        gen = model.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=s),
            attention_mask=m,
            decoder_start_token_id=model.bart.config.decoder_start_token_id,
            **gen_kwargs,
        )
        out.extend(gen[i].cpu().tolist() for i in range(gen.size(0)))
    return out


@torch.no_grad()
def generate_from_text(model, tokenizer, texts, device, max_text_len,
                       gen_kwargs, batch_size):
    """Pure text -> token IDs (the alpha=1 upper bound, via input_ids)."""
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_text_len)
        gen = model.bart.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            **gen_kwargs,
        )
        out.extend(gen[i].cpu().tolist() for i in range(gen.size(0)))
    return out


def score_both_parsers(id_lists, golds, tokenizer):
    """F1 under the faithful gold parser and under the train-eval parser."""
    ext, deli = [], []
    for ids in id_lists:
        s = tokenizer.decode(ids, skip_special_tokens=False)
        ext.append(extract_triplets_from_rebel_output(s))
        deli.append(delinearize(ids, tokenizer))
    return {
        "extract": compute_triplet_f1(ext, golds),
        "delinearize": compute_triplet_f1(deli, golds),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gen_kwargs = dict(max_length=args.max_len, num_beams=args.num_beams,
                      length_penalty=args.length_penalty,
                      early_stopping=args.early_stopping)

    tokenizer = build_tokenizer(args.bart_name)
    model = EEGBartModel(tokenizer=tokenizer, bart_name=args.bart_name,
                         dropout=0.0).to(device)
    model.eval()

    texts, golds = load_unique_sentences(args.processed_dir, args.triplets_path,
                                         args.split)
    if args.limit > 0:
        texts, golds = texts[:args.limit], golds[:args.limit]
    print(f"{len(texts)} unique {args.split} sentences with triplets")

    # ---- alpha = 1 upper bound: pure text -> triplets ----
    print("\n[upper bound] text -> REBEL -> triplets")
    ub_ids = generate_from_text(model, tokenizer, texts, device,
                                args.max_text_len, gen_kwargs, args.batch_size)
    ub = score_both_parsers(ub_ids, golds, tokenizer)
    print(f"  extract     F1={ub['extract']['f1']:.4f}  "
          f"(P={ub['extract']['precision']:.3f} R={ub['extract']['recall']:.3f})")
    print(f"  delinearize F1={ub['delinearize']['f1']:.4f}  "
          f"(harness ceiling — lower than 1.0 is the parser artifact)")

    # ---- encoder states for the interpolation source ----
    states, masks = encode_texts(model, tokenizer, texts, device,
                                 args.max_text_len, args.batch_size)
    maskb = masks.bool()

    # Noise matched to per-dim mean/std of the valid (unpadded) text states.
    valid = maskb.unsqueeze(-1).expand_as(states)
    flat = states[valid].reshape(-1, states.shape[-1])
    mu, sd = flat.mean(dim=0), flat.std(dim=0)
    g = torch.Generator().manual_seed(args.seed)
    noise = mu + sd * torch.randn(states.shape, generator=g)

    alphas = [float(a) for a in args.alphas.split(",")]
    curve = []
    print("\n[interpolation] states(alpha) = alpha*text + (1-alpha)*noise")
    for a in alphas:
        interp = a * states + (1.0 - a) * noise
        cosine, retrieval = alignment_metrics(interp, states, maskb)
        ids = generate_from_states(model, interp, masks, device, gen_kwargs,
                                   args.batch_size)
        sc = score_both_parsers(ids, golds, tokenizer)
        curve.append({
            "alpha": a,
            "align_cosine": cosine,
            "align_retrieval": retrieval,
            "f1_extract": sc["extract"]["f1"],
            "f1_delinearize": sc["delinearize"]["f1"],
            "precision_extract": sc["extract"]["precision"],
            "recall_extract": sc["extract"]["recall"],
        })
        print(f"  alpha={a:4.2f}  retr@1={retrieval:.3f}  cos={cosine:+.3f}  "
              f"F1ext={sc['extract']['f1']:.4f}  F1del={sc['delinearize']['f1']:.4f}")

    result = {
        "split": args.split,
        "n_sentences": len(texts),
        "gen_kwargs": gen_kwargs,
        "upper_bound": ub,
        "curve": curve,
        "bridge_retrieval": args.bridge_retrieval,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out}")

    if args.plot:
        _plot(result, args.plot)
        print(f"Saved figure -> {args.plot}")


def _plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = sorted(result["curve"], key=lambda c: c["align_retrieval"])
    x = [c["align_retrieval"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [c["f1_extract"] for c in curve], "o-", color="#1f77b4",
            label="F1 (faithful parser)")
    ax.plot(x, [c["f1_delinearize"] for c in curve], "s--", color="#2ca02c",
            label="F1 (train-eval parser)")
    ax.set_xlabel("Encoder alignment to REBEL text states (InfoNCE retrieval@1)")
    ax.set_ylabel("Triplet F1")
    ax.set_title(f"Required encoder fidelity for triplet generation ({result['split']})")
    if result.get("bridge_retrieval") is not None:
        ax.axvline(result["bridge_retrieval"], ls="--", color="#d62728",
                   label=f"trained bridge retr@1 = {result['bridge_retrieval']:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E1: text-teacher upper bound + noise interpolation")
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--triplets_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--bart_name", default="Babelscape/rebel-large")
    p.add_argument("--alphas", default="1.0,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0",
                   help="comma-separated interpolation weights to sweep")
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--early_stopping", action="store_true",
                   help="off by default to match generateTriplets GEN_KWARGS")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_text_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap unique sentences (debug)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bridge_retrieval", type=float, default=None,
                   help="trained bridge's measured retrieval@1, drawn on the figure")
    p.add_argument("--out", default="../exp_logs/e1_noise_interpolation.json")
    p.add_argument("--plot", default="../exp_logs/e1_noise_interpolation.png")
    main(p.parse_args())
