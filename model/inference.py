"""
EEG-to-Graph: Inference
========================

Load a trained model and generate triplets from EEG features.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --vocab_path ./checkpoints/vocab.json \
        --processed_dir ./processed_zuco \
        --split test \
        --beam_size 4 \
        --output predictions.json
"""

import argparse
import json
import torch
import numpy as np

from vocabulary import Vocabulary
from eeg_graph_model import EEGGraphModel


def load_model(checkpoint_path, vocab_size, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("args", {})

    model = EEGGraphModel(
        vocab_size=vocab_size,
        eeg_dim=model_args.get("eeg_dim", 840),
        d_model=model_args.get("d_model", 256),
        n_heads=model_args.get("n_heads", 8),
        n_enc_layers=model_args.get("n_enc_layers", 4),
        n_dec_layers=model_args.get("n_dec_layers", 4),
        max_src_len=model_args.get("max_src_len", 128),
        max_tgt_len=model_args.get("max_tgt_len", 128),
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_f1={ckpt.get('val_f1', '?')}")
    return model


def predict_single(model, eeg_features, vocab, device, beam_size=4, max_len=128):
    """
    Generate triplets for a single EEG sample.

    Args:
        eeg_features: numpy array of shape (seq_len, 840)
        vocab: Vocabulary instance
        beam_size: 0 for greedy, >0 for beam search

    Returns:
        list of triplet dicts
    """
    src = torch.tensor(eeg_features, dtype=torch.float32).unsqueeze(0).to(device)
    src_mask = torch.ones(1, src.size(1), dtype=torch.bool, device=device)

    if beam_size > 0:
        token_ids = model.beam_search(src, src_mask, beam_size=beam_size, max_len=max_len)
    else:
        token_ids = model.generate(src, src_mask, max_len=max_len)
        token_ids = token_ids[0]

    return vocab.delinearize(token_ids.cpu().tolist())


def predict_batch(model, eeg_list, meta_list, vocab, device, beam_size=4, max_len=128):
    """
    Generate triplets for a batch of samples.

    For beam_size > 0, processes one at a time (beam search is per-sample).
    For greedy (beam_size=0), processes in a single forward pass.
    """
    results = []

    if beam_size > 0:
        for i, (eeg, meta) in enumerate(zip(eeg_list, meta_list)):
            triplets = predict_single(model, eeg, vocab, device, beam_size, max_len)
            results.append({
                "text": meta.get("text", ""),
                "subject_id": meta.get("subject_id", ""),
                "predicted_triplets": triplets,
            })
    else:
        # Pad and batch for greedy
        max_src = max(e.shape[0] for e in eeg_list)
        feat_dim = eeg_list[0].shape[1]
        B = len(eeg_list)

        src = torch.zeros(B, max_src, feat_dim, device=device)
        src_mask = torch.zeros(B, max_src, dtype=torch.bool, device=device)

        for i, eeg in enumerate(eeg_list):
            n = eeg.shape[0]
            src[i, :n] = torch.tensor(eeg, dtype=torch.float32)
            src_mask[i, :n] = True

        generated = model.generate(src, src_mask, max_len=max_len)

        for i in range(B):
            tokens = generated[i].cpu().tolist()
            triplets = vocab.delinearize(tokens)
            results.append({
                "text": meta_list[i].get("text", ""),
                "subject_id": meta_list[i].get("subject_id", ""),
                "predicted_triplets": triplets,
            })

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocab and model
    vocab = Vocabulary.load(args.vocab_path)
    model = load_model(args.checkpoint, len(vocab), device)

    # Load data
    import os
    eeg_path = os.path.join(args.processed_dir, f"{args.split}_eeg.npy")
    meta_path = os.path.join(args.processed_dir, f"{args.split}_meta.json")

    eeg_data = np.load(eeg_path, allow_pickle=True)
    with open(meta_path) as f:
        meta_data = json.load(f)

    print(f"Loaded {len(eeg_data)} samples from {args.split} split")

    # Run inference
    all_results = []
    batch_size = 8

    for start in range(0, len(eeg_data), batch_size):
        end = min(start + batch_size, len(eeg_data))
        batch_eeg = [eeg_data[i] for i in range(start, end)]
        batch_meta = [meta_data[i] for i in range(start, end)]

        results = predict_batch(
            model, batch_eeg, batch_meta, vocab, device,
            beam_size=args.beam_size, max_len=args.max_len,
        )
        all_results.extend(results)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{len(eeg_data)} samples")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} predictions to {args.output}")

    # Print a few examples
    print("\nExample predictions:")
    for r in all_results[:3]:
        print(f"  Text: {r['text'][:80]}...")
        print(f"  Triplets: {r['predicted_triplets']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-to-Graph Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()
    main(args)