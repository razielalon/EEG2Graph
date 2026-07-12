#!/usr/bin/env python
"""
E8 — cross-subject eval: seen vs held-out subject triplet-F1 for one checkpoint.

E8 (RQ5) asks whether the decodable EEG->triplet code is *shared* across subjects
or *subject-specific*. A model is trained with some subjects dropped from TRAIN
(`--exclude_subjects`); val/test keep all 18 subjects, so the held-out subjects are
still scored at test time — they are the "generalize to a new person" cases.

Aggregate test F1 (what train.py writes to test_metrics.json) dilutes this: with 3
of 18 subjects held out, the unseen samples are a small minority, so the aggregate
barely moves even if unseen decoding totally fails. This script re-scores the test
split and partitions it into SEEN vs UNSEEN subjects (plus a per-subject table), so
the true transfer gap is visible.

It reuses inference.load_model / inference.predict_batch (generation) and
train.compute_triplet_f1 (the exact metric train.py reports), so the numbers are
comparable to test_metrics.json -- but only at the same beam width: --beam_size
must match train.py's (default 4). Greedy seen/unseen F1 against beam-4 aggregate
F1 is an apples-to-oranges comparison.

Writes `test_by_subject.json` into the checkpoint dir (or --output). Held-out set
comes from --exclude_subjects (same grammar as train.py: comma-separated ids); when
empty, every subject is "seen" and the unseen partition is null.

Usage (run from model/):
    python cross_subject_eval.py \
        --checkpoint ../checkpoints_e8_x09/best_model.pt \
        --tokenizer_dir ../checkpoints_e8_x09/tokenizer \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --exclude_subjects YLS,YMD,YMS,YRH,YRK,YRP,YSD,YSL,YTL \
        --output ../checkpoints_e8_x09/test_by_subject.json
"""
import argparse
import json
import os

import numpy as np
import torch

from vocabulary import load_tokenizer
from inference import load_model, predict_batch
from train import compute_triplet_f1


def _load_gold_index(triplets_path):
    """text -> [triplet dicts], tolerating both dict-keyed and list formats."""
    with open(triplets_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {text.strip(): entry.get("triplets", []) for text, entry in raw.items()}
    return {e["text"].strip(): e.get("triplets", []) for e in raw}


def _f1(pred_batch, gold_batch):
    """compute_triplet_f1 on empty input returns zeros; guard the empty case."""
    if not pred_batch:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": 0, "n_gold": 0, "n_correct": 0, "n_samples": 0}
    m = compute_triplet_f1(pred_batch, gold_batch)
    m["n_samples"] = len(pred_batch)
    return m


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    excluded = {s for s in args.exclude_subjects.split(",") if s.strip()}
    print(f"Held-out (unseen) subjects: {sorted(excluded) or '(none)'}")

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model(args.checkpoint, tokenizer, device)
    gold_index = _load_gold_index(args.triplets_path)

    eeg_path = os.path.join(args.processed_dir, f"{args.split}_eeg.npy")
    meta_path = os.path.join(args.processed_dir, f"{args.split}_meta.json")
    eeg_data = np.load(eeg_path, allow_pickle=True)
    with open(meta_path) as f:
        meta_data = json.load(f)
    if args.limit and args.limit > 0:
        eeg_data = eeg_data[:args.limit]
        meta_data = meta_data[:args.limit]
    print(f"Loaded {len(eeg_data)} {args.split} samples")

    num_beams = max(1, args.beam_size)
    # Per-sample records: (subject_id, predicted_triplets, gold_triplets).
    records = []
    for start in range(0, len(eeg_data), args.batch_size):
        end = min(start + args.batch_size, len(eeg_data))
        batch_eeg = [eeg_data[i] for i in range(start, end)]
        batch_meta = [meta_data[i] for i in range(start, end)]
        preds = predict_batch(model, batch_eeg, batch_meta, tokenizer, device,
                              num_beams=num_beams, max_len=args.max_len)
        for r in preds:
            gold = gold_index.get(r["text"].strip(), [])
            records.append((r["subject_id"], r["predicted_triplets"], gold))
        if (start // args.batch_size) % 10 == 0:
            print(f"  scored {end}/{len(eeg_data)}")

    # Partition preds/golds into seen vs unseen and by subject.
    def score(keep):
        p = [pt for sid, pt, gt in records if keep(sid)]
        g = [gt for sid, pt, gt in records if keep(sid)]
        return _f1(p, g)

    subjects = sorted({sid for sid, _, _ in records})
    result = {
        "split": args.split,
        "excluded_subjects": sorted(excluded),
        "n_excluded": len(excluded),
        "overall": score(lambda sid: True),
        "seen": score(lambda sid: sid not in excluded),
        "unseen": score(lambda sid: sid in excluded) if excluded else None,
        "per_subject": {sid: score(lambda s, t=sid: s == t) for sid in subjects},
    }

    out = args.output or os.path.join(os.path.dirname(args.checkpoint),
                                      "test_by_subject.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    def line(name, m):
        if m is None:
            print(f"  {name:8s}  (none)")
            return
        print(f"  {name:8s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  n={m['n_samples']}")
    print("\nTest triplet-F1 by partition:")
    line("overall", result["overall"])
    line("seen", result["seen"])
    line("unseen", result["unseen"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E8 seen-vs-unseen subject eval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--triplets_path", required=True)
    ap.add_argument("--exclude_subjects", default="",
                    help="Comma-separated held-out subject ids (same list passed "
                         "to train.py --exclude_subjects). Empty = all seen.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam_size", type=int, default=4,
                    help="num_beams. MUST match train.py's --beam_size (default 4) "
                         "or seen/unseen F1 is not comparable to test_metrics.json.")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, score only the first N test samples (CPU smoke).")
    ap.add_argument("--output", default="",
                    help="Output JSON path (default: <checkpoint dir>/test_by_subject.json).")
    main(ap.parse_args())
