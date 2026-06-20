"""
E3: Prediction-collapse analysis (RQ3 — anatomy of the decoder shortcut)

A model that has fallen into the decoder shortcut stops conditioning on the
EEG and emits (near-)constant, memorised output. This quantifies that across
runs from their saved predictions — no GPU, no model load.

For each run's test_results.json ([{"gold": [...], "pred": [...]}, ...]) we
report:
  - unique_pred_rate     : distinct predicted triplet-SETS / n_samples
                           (1.0 = every input gets a different answer;
                            ~0 = collapse to one constant answer)
  - top1_share           : fraction of samples emitting the single most common
                           prediction (high = collapse)
  - pred_entropy_bits     : Shannon entropy of the prediction distribution
  - empty_rate           : fraction predicting zero triplets
  - mean_pred_triplets   : average #triplets per sample
  - distinct1 / distinct2: token-level diversity of predicted relation strings
  - train_memorization   : fraction of predicted triplet-tuples that appear
                           verbatim in the TRAIN gold set (needs --triplets_path
                           + --processed_dir); high = parroting training labels

Contrast: unfrozen runs (decoder shortcut) collapse — low unique_pred_rate,
high top1_share, high train_memorization. The frozen-BART run, where the bridge
is the only EEG->loss path, should be more diverse.

Usage:
    python prediction_analysis.py \
        --runs baseline:../checkpoints_baseline/test_results.json \
               frozen:../checkpoints_subjtemp/test_results.json \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --processed_dir ../processed_zuco2 \
        --out ../exp_logs/e3_prediction_collapse.json
"""

import argparse
import json
import math
import os
from collections import Counter

from train import triplet_set_to_tuples


def canon(triplets):
    """Canonical hashable signature of a predicted triplet SET (order-free)."""
    return tuple(sorted(triplet_set_to_tuples(triplets)))


def load_results(path):
    """Return list of (gold_triplets, pred_triplets). Accepts test_results.json
    or best_examples.json (same {gold, pred} schema)."""
    data = json.load(open(path))
    return [(r.get("gold", []), r.get("pred", [])) for r in data]


def train_triplet_tuples(triplets_path, processed_dir):
    """Set of all triplet-tuples that occur in TRAIN gold (memorization ref)."""
    raw = json.load(open(triplets_path))
    if isinstance(raw, dict):
        index = {t.strip(): e.get("triplets", []) for t, e in raw.items()}
    else:
        index = {e["text"].strip(): e.get("triplets", []) for e in raw}
    meta = json.load(open(os.path.join(processed_dir, "train_meta.json")))
    tuples = set()
    for m in meta:
        for tup in triplet_set_to_tuples(index.get(m["text"].strip(), [])):
            tuples.add(tup)
    return tuples


def analyze(pairs, train_tuples=None):
    n = len(pairs)
    sigs = [canon(p) for _, p in pairs]
    counts = Counter(sigs)
    total_sig = sum(counts.values())
    entropy = -sum((c / total_sig) * math.log2(c / total_sig)
                   for c in counts.values())

    n_pred_triplets = [len(triplet_set_to_tuples(p)) for _, p in pairs]
    empty = sum(1 for k in n_pred_triplets if k == 0)

    # token diversity over predicted relation strings
    rel_tokens, bigrams = [], []
    for _, p in pairs:
        for t in p:
            toks = t.get("relation", "").lower().split()
            rel_tokens += toks
            bigrams += list(zip(toks, toks[1:]))
    distinct1 = len(set(rel_tokens)) / max(len(rel_tokens), 1)
    distinct2 = len(set(bigrams)) / max(len(bigrams), 1)

    res = {
        "n_samples": n,
        "unique_pred_rate": len(counts) / max(n, 1),
        "n_unique_predictions": len(counts),
        "top1_share": counts.most_common(1)[0][1] / max(n, 1) if counts else 0,
        "pred_entropy_bits": entropy,
        "empty_rate": empty / max(n, 1),
        "mean_pred_triplets": sum(n_pred_triplets) / max(n, 1),
        "distinct1": distinct1,
        "distinct2": distinct2,
    }
    if train_tuples is not None:
        pred_tuples = [tup for _, p in pairs for tup in triplet_set_to_tuples(p)]
        in_train = sum(1 for tup in pred_tuples if tup in train_tuples)
        res["train_memorization"] = in_train / max(len(pred_tuples), 1)
        res["n_pred_tuples"] = len(pred_tuples)
    # most common prediction, for the report
    if counts:
        top_sig = counts.most_common(1)[0][0]
        res["most_common_prediction"] = [list(t) for t in top_sig]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="name:path/to/test_results.json entries")
    p.add_argument("--triplets_path", default=None)
    p.add_argument("--processed_dir", default=None)
    p.add_argument("--out", default="../exp_logs/e3_prediction_collapse.json")
    args = p.parse_args()

    train_tuples = None
    if args.triplets_path and args.processed_dir:
        train_tuples = train_triplet_tuples(args.triplets_path, args.processed_dir)
        print(f"Train gold triplet-tuples: {len(train_tuples)}")

    report = {}
    for entry in args.runs:
        name, path = entry.split(":", 1)
        if not os.path.exists(path):
            print(f"  ! {name}: {path} not found, skipping")
            continue
        res = analyze(load_results(path), train_tuples)
        report[name] = res
        print(f"\n=== {name} ===")
        for k in ("n_samples", "unique_pred_rate", "top1_share",
                  "pred_entropy_bits", "empty_rate", "mean_pred_triplets",
                  "distinct1", "distinct2", "train_memorization"):
            if k in res:
                print(f"  {k:20s}: {res[k]:.4f}" if isinstance(res[k], float)
                      else f"  {k:20s}: {res[k]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
