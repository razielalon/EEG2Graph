"""
EEG-to-Graph: Inference
========================

Load a trained Bridge + BART model and generate triplets from EEG features.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --tokenizer_dir ./checkpoints/tokenizer \
        --processed_dir ./processed_zuco \
        --split test \
        --beam_size 4 \
        --output predictions.json
"""

import argparse
import json
import os

import numpy as np
import torch

from vocabulary import load_tokenizer, delinearize
from eeg_graph_model import EEGBartModel


def load_model(checkpoint_path, tokenizer, device):
    """Load a trained Bridge + BART model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("args", {})
    bart_name = ckpt.get("bart_name") or model_args.get("bart_name", "Babelscape/rebel-large")

    model = EEGBartModel(
        tokenizer=tokenizer,
        eeg_dim=model_args.get("eeg_dim", 840),
        bart_name=bart_name,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_f1={ckpt.get('val_f1', '?')}")
    return model


def predict_batch(model, eeg_list, meta_list, tokenizer, device, num_beams=1, max_len=128):
    """Generate triplets for a batch of samples in one forward pass."""
    max_src = max(e.shape[0] for e in eeg_list)
    feat_dim = eeg_list[0].shape[1]
    B = len(eeg_list)

    src = torch.zeros(B, max_src, feat_dim, device=device)
    src_mask = torch.zeros(B, max_src, dtype=torch.bool, device=device)

    for i, eeg in enumerate(eeg_list):
        n = eeg.shape[0]
        src[i, :n] = torch.tensor(eeg, dtype=torch.float32)
        src_mask[i, :n] = True

    generated = model.generate(src, src_mask, max_len=max_len, num_beams=num_beams)

    results = []
    for i in range(B):
        tokens = generated[i].cpu().tolist()
        triplets = delinearize(tokens, tokenizer)
        results.append({
            "text": meta_list[i].get("text", ""),
            "subject_id": meta_list[i].get("subject_id", ""),
            "predicted_triplets": triplets,
        })
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model(args.checkpoint, tokenizer, device)

    eeg_path = os.path.join(args.processed_dir, f"{args.split}_eeg.npy")
    meta_path = os.path.join(args.processed_dir, f"{args.split}_meta.json")

    eeg_data = np.load(eeg_path, allow_pickle=True)
    with open(meta_path) as f:
        meta_data = json.load(f)

    print(f"Loaded {len(eeg_data)} samples from {args.split} split")

    all_results = []
    batch_size = args.batch_size
    num_beams = max(1, args.beam_size)

    for start in range(0, len(eeg_data), batch_size):
        end = min(start + batch_size, len(eeg_data))
        batch_eeg = [eeg_data[i] for i in range(start, end)]
        batch_meta = [meta_data[i] for i in range(start, end)]

        results = predict_batch(
            model, batch_eeg, batch_meta, tokenizer, device,
            num_beams=num_beams, max_len=args.max_len,
        )
        all_results.extend(results)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{len(eeg_data)} samples")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} predictions to {args.output}")

    print("\nExample predictions:")
    for r in all_results[:3]:
        print(f"  Text: {r['text'][:80]}...")
        print(f"  Triplets: {r['predicted_triplets']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-to-Graph Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Directory containing the saved BART tokenizer")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="num_beams (1 = greedy)")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()
    main(args)
