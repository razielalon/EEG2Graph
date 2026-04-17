"""
EEG-to-Graph: Seq2Seq Dataset
==============================

Pairs preprocessed ZuCo EEG features (encoder input) with linearized
triplet token sequences (decoder target), tokenized with BART's BPE
tokenizer plus the four structural markers.

Expected triplets format (JSON):
[
  {
    "text": "Barack Obama was born in Hawaii .",
    "triplets": [
      {"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}
    ]
  },
  ...
]
"""

import os
import json
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from vocabulary import build_tokenizer, linearize_triplets


class EEGGraphDataset(Dataset):
    """
    PyTorch Dataset for EEG-to-Graph seq2seq training.

    Each sample contains:
        - eeg:         (seq_len, 840) — encoder input
        - target_ids:  (target_len,)  — linearized triplet token IDs
        - has_fixation:(seq_len,)     — fixation mask
        - meta: dict with text, words, subject_id, etc.
    """

    def __init__(self, eeg_path, meta_path, triplets_path, tokenizer, max_src_len=128, max_tgt_len=128):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.eeg_data = np.load(eeg_path, allow_pickle=True)

        with open(meta_path) as f:
            self.meta = json.load(f)

        with open(triplets_path) as f:
            triplets_raw = json.load(f)
        if isinstance(triplets_raw, dict):
            self.triplet_index = {
                text.strip(): entry.get("triplets", [])
                for text, entry in triplets_raw.items()
            }
        else:
            self.triplet_index = {
                e["text"].strip(): e.get("triplets", [])
                for e in triplets_raw
            }

        self.samples = []
        n_matched = 0
        for idx in range(len(self.meta)):
            m = self.meta[idx]
            text = m["text"].strip()
            triplets = self.triplet_index.get(text, [])

            target_ids = linearize_triplets(triplets, tokenizer)
            target_ids = target_ids[:self.max_tgt_len]
            # Guarantee the sequence ends with EOS after truncation
            if target_ids[-1] != tokenizer.eos_token_id:
                target_ids[-1] = tokenizer.eos_token_id

            eeg = self.eeg_data[idx]
            n_words = min(len(m["words"]), self.max_src_len)
            eeg = eeg[:n_words]
            has_fix = np.array(m["has_fixation"][:n_words], dtype=bool)

            if len(triplets) > 0:
                n_matched += 1

            self.samples.append({
                "eeg": torch.tensor(eeg, dtype=torch.float32),
                "target_ids": torch.tensor(target_ids, dtype=torch.long),
                "has_fixation": torch.tensor(has_fix, dtype=torch.bool),
                "n_src": n_words,
                "n_tgt": len(target_ids),
                "meta": {
                    "text": text,
                    "words": m["words"][:n_words],
                    "subject_id": m["subject_id"],
                    "task": m["task"],
                    "triplets": triplets,
                },
            })

        print(f"  {len(self.samples)} samples, {n_matched} with triplets "
              f"({n_matched / max(len(self.samples), 1) * 100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Collate: pad both encoder and decoder sequences
# =============================================================================

def collate_fn(batch, pad_id):
    """
    Pads EEG (encoder) and target (decoder) sequences.

    Returns dict with:
        - src:         (B, max_src, 840)
        - src_mask:    (B, max_src) — True for real tokens
        - src_fixation:(B, max_src)
        - tgt:         (B, max_tgt - 1) — decoder input (all but last token)
        - tgt_labels:  (B, max_tgt - 1) — decoder target (all but first token)
        - tgt_mask:    (B, max_tgt - 1) — True for non-pad
        - meta:        list of dicts
    """
    B = len(batch)
    max_src = max(b["n_src"] for b in batch)
    max_tgt = max(b["n_tgt"] for b in batch)
    feat_dim = batch[0]["eeg"].shape[-1]

    src = torch.zeros(B, max_src, feat_dim)
    src_mask = torch.zeros(B, max_src, dtype=torch.bool)
    src_fix = torch.zeros(B, max_src, dtype=torch.bool)

    tgt_in = torch.full((B, max_tgt - 1), pad_id, dtype=torch.long)
    tgt_lbl = torch.full((B, max_tgt - 1), pad_id, dtype=torch.long)
    tgt_mask = torch.zeros(B, max_tgt - 1, dtype=torch.bool)

    metas = []

    for i, b in enumerate(batch):
        ns, nt = b["n_src"], b["n_tgt"]

        src[i, :ns] = b["eeg"]
        src_mask[i, :ns] = True
        src_fix[i, :ns] = b["has_fixation"]

        tgt_in[i, :nt - 1] = b["target_ids"][:-1]
        tgt_lbl[i, :nt - 1] = b["target_ids"][1:]
        tgt_mask[i, :nt - 1] = True

        metas.append(b["meta"])

    return {
        "src": src,
        "src_mask": src_mask,
        "src_fixation": src_fix,
        "tgt": tgt_in,
        "tgt_labels": tgt_lbl,
        "tgt_mask": tgt_mask,
        "meta": metas,
    }


# =============================================================================
# Build dataloaders
# =============================================================================

def build_dataloaders(
    processed_dir,
    triplets_path,
    batch_size=16,
    max_src_len=128,
    max_tgt_len=128,
    num_workers=0,
    bart_name="facebook/bart-base",
    tokenizer=None,
    limits=None,
):
    """
    Build train/val/test dataloaders and a tokenizer.

    If `tokenizer` is provided, reuse it (e.g., at inference time).
    Otherwise build a fresh one from `bart_name`.

    `limits`: optional dict like {"train": 64, "val": 16, "test": 16}. Values
    that are None or 0 leave the corresponding split at full size. Useful for
    fast CPU sanity runs.

    Returns:
        dataloaders: dict of DataLoaders
        tokenizer:   the BART tokenizer (with structural tokens added)
    """
    if tokenizer is None:
        tokenizer = build_tokenizer(bart_name)

    collate = partial(collate_fn, pad_id=tokenizer.pad_token_id)
    limits = limits or {}

    loaders = {}
    for split in ["train", "val", "test"]:
        eeg_path = os.path.join(processed_dir, f"{split}_eeg.npy")
        meta_path = os.path.join(processed_dir, f"{split}_meta.json")
        if not os.path.exists(eeg_path):
            print(f"  Skipping {split}: {eeg_path} not found")
            continue

        print(f"\nLoading {split}:")
        ds = EEGGraphDataset(eeg_path, meta_path, triplets_path, tokenizer, max_src_len, max_tgt_len)

        limit = limits.get(split)
        if limit and limit < len(ds):
            ds = Subset(ds, list(range(limit)))
            print(f"  (subset to first {limit} samples)")

        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=collate, num_workers=num_workers, pin_memory=True,
        )

    return loaders, tokenizer
