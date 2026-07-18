"""
E10 — Sentence-level retrieve-and-copy baseline (RQ1: what does the signal buy?).

E4/E9 show the only decodable EEG signal is *sentence identity* (not word-level
triplet structure). This baseline uses that signal in the most direct way: retrieve
the nearest sentence from the EEG, EMIT that sentence's gold triplets, and score
triplet-F1 against the true sentence's gold triplets — then compare to the trained
generative model's ~0.008.

It is REBEL-free (numpy + sklearn only) and reuses E4/E9's exact retrieval front
end (`retrieval_topk`'s StandardScaler -> Ridge -> TF-IDF/SVD map), so the retrieval
is byte-comparable with the rest of the series.

Conditions
----------
  oracle      : gallery = unique TEST sentences (transductive). Diagnostic *ceiling*
                on what sentence identity buys — the candidate set is handed to it.
  deployable  : gallery = unique TRAIN sentences; copy the retrieved train sentence's
                triplets. Honest generalization setting (test sentences are unseen,
                since splits are by sentence), bounded by cross-sentence triplet overlap.
  random      : copy a random gallery sentence's triplets (chance for this F1 metric).
  freq_train  : emit the single most-frequent TRAIN triplet for every sample — this is
                the collapse mode the trained model falls into (E3).
  self_copy   : copy the TRUE sentence's own triplets (sanity, must be 1.0).

Needs <data>/sentence_triplets.json alongside the EEG (keyed by sentence text).

Usage:
  python model/sentence_retrieval_baseline.py --data processed_zuco \
      --out exp_logs/e10_sentence_baseline_zuco2.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def load_sentence_level(data, split):
    """Mean-pool each sentence's EEG over fixated words -> (N, 840). (E4 loader.)"""
    eeg = np.load(f"{data}/{split}_eeg.npy", allow_pickle=True)
    meta = json.load(open(f"{data}/{split}_meta.json", encoding="utf-8"))
    X = np.zeros((len(eeg), 840), dtype=np.float32)
    for i, e in enumerate(eeg):
        e = np.asarray(e, dtype=np.float32)
        fix = np.asarray(meta[i]["has_fixation"][:e.shape[0]], dtype=bool)
        rows = e[fix] if fix.any() else e
        X[i] = rows.mean(axis=0) if len(rows) else 0.0
    return np.nan_to_num(X), meta


def fit_retriever(Xtr, Xte, mtr, mte, svd_dim=128):
    """E4/E9 retrieval front end. Returns (predict_embed(gallery)->ranker) closure.

    Fitting matches `retrieval_topk` exactly: StandardScaler on features, TF-IDF fit
    on train+test texts, SVD on train, Ridge from EEG to the SVD text embedding.
    """
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    txt_tr = [m["text"] for m in mtr]
    txt_te = [m["text"] for m in mte]
    tfidf = TfidfVectorizer(stop_words="english", min_df=2).fit(txt_tr + txt_te)
    n_comp = min(svd_dim, tfidf.transform(txt_tr).shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(tfidf.transform(txt_tr))

    def embed(texts):
        v = svd.transform(tfidf.transform(texts))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    reg = Ridge(alpha=10.0).fit(Xtr, embed(txt_tr))
    pred = reg.predict(Xte)
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)

    def top1(gallery_texts):
        gal = embed(gallery_texts)
        order = np.argsort(-(pred @ gal.T), axis=1)
        return [gallery_texts[o[0]] for o in order]
    return top1


def load_triplets(data):
    raw = json.load(open(f"{data}/sentence_triplets.json", encoding="utf-8"))

    def to_set(entry):
        trips = entry.get("triplets", entry) if isinstance(entry, dict) else entry
        out = set()
        for t in trips or []:
            s = str(t.get("subject", "")).lower().strip()
            r = str(t.get("relation", "")).lower().strip()
            o = str(t.get("object", "")).lower().strip()
            if s or r or o:
                out.add((s, r, o))
        return frozenset(out)
    return {k: to_set(v) for k, v in raw.items()}


def micro_f1(preds, golds):
    """Micro-averaged exact-match triplet F1 — matches train.py:compute_triplet_f1."""
    correct = sum(len(p & g) for p, g in zip(preds, golds))
    npred = sum(len(p) for p in preds)
    ngold = sum(len(g) for g in golds)
    P = correct / npred if npred else 0.0
    R = correct / ngold if ngold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return dict(precision=P, recall=R, f1=F, correct=correct, n_pred=npred, n_gold=ngold)


def run(data, seed=0):
    Xtr, mtr = load_sentence_level(data, "train")
    Xte, mte = load_sentence_level(data, "test")
    txt_tr = [m["text"] for m in mtr]
    txt_te = [m["text"] for m in mte]
    trip = load_triplets(data)

    def trips_for(t):
        return trip.get(t, frozenset())

    gold = [trips_for(t) for t in txt_te]
    top1 = fit_retriever(Xtr, Xte, mtr, mte)

    gal_test = sorted(set(txt_te))
    gal_train = sorted(set(txt_tr))
    ret_test = top1(gal_test)     # retrieved sentence TEXT per test sample
    ret_train = top1(gal_train)
    ret_top1 = float(np.mean([p == g for p, g in zip(ret_test, txt_te)]))
    pred_oracle = [trips_for(t) for t in ret_test]    # -> their triplet sets
    pred_deploy = [trips_for(t) for t in ret_train]

    # random-sentence copy (chance for this F1 metric), averaged over draws
    rng = np.random.default_rng(seed)
    rand = [micro_f1([trips_for(gal_test[i]) for i in rng.integers(0, len(gal_test), len(txt_te))],
                     gold)["f1"] for _ in range(20)]

    # most-frequent train triplet (the model's collapse mode, E3)
    c = Counter(tp for t in set(txt_tr) for tp in trips_for(t))
    freq = frozenset({c.most_common(1)[0][0]}) if c else frozenset()

    return {
        "data": os.path.basename(os.path.normpath(data)),
        "n_test": len(txt_te), "n_test_unique": len(gal_test),
        "n_train_unique": len(gal_train), "n_gold": sum(len(g) for g in gold),
        "retrieval_top1_acc": ret_top1,
        "self_copy_f1": micro_f1(gold, gold)["f1"],
        "oracle": micro_f1(pred_oracle, gold),
        "deployable": micro_f1(pred_deploy, gold),
        "random_copy_f1": float(np.mean(rand)),
        "freq_train_triplet": micro_f1([freq] * len(txt_te), gold),
        "model_reference": {"e5_best_test_f1": 0.0083, "typical_test_f1": 0.0},
    }


def main():
    p = argparse.ArgumentParser(description="E10 sentence-level retrieve-and-copy baseline")
    p.add_argument("--data", default="processed_zuco",
                   help="dir with {train,test}_eeg.npy + _meta.json + sentence_triplets.json")
    p.add_argument("--out", default="exp_logs/e10_sentence_baseline.json")
    args = p.parse_args()

    r = run(args.data)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)

    print(f"=== {r['data']}  (n_test={r['n_test']}, {r['n_test_unique']} uniq sents, "
          f"{r['n_gold']} gold triplets) ===")
    print(f"  retrieval top-1 acc     : {r['retrieval_top1_acc']:.3f}")
    print(f"  self-copy F1 (sanity)   : {r['self_copy_f1']:.3f}")
    print(f"  ORACLE retrieve+copy F1 : {r['oracle']['f1']:.4f}  "
          f"(P={r['oracle']['precision']:.3f} R={r['oracle']['recall']:.3f} "
          f"correct={r['oracle']['correct']}/{r['n_gold']})")
    print(f"  deployable F1           : {r['deployable']['f1']:.4f}")
    print(f"  random-copy F1 (chance) : {r['random_copy_f1']:.4f}")
    print(f"  freq-train triplet F1   : {r['freq_train_triplet']['f1']:.4f}")
    print(f"  [ref] model test F1     : ~0.008 (E5 best) / ~0 (most)")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
