"""
E4: Extended EEG signal probes (RQ1 — where does the decodable signal live?)

Builds on tests/eeg_signal_probe.py (task / subject / sentence-retrieval). Adds
four localisation studies, all REBEL-free linear probes trained on train and
evaluated on the held-out test split:

  A. Frequency-band ablation   — which of the 8 bands carry semantic signal
  B. Per-channel importance    — which of the 105 channels carry it (montage-
                                 free: ZuCo ships no electrode labels here, so
                                 channels are reported by index, not scalp region)
  C. Word-class probes         — what *kind* of word-level info is decodable
                                 (content vs function, word length)
  D. Per-subject retrieval     — retrieval quality per subject (motivates E8)

Feature layout (from preprocess_zuco.py): 840 = 8 bands x 105 channels,
BAND-MAJOR, so band b occupies columns [b*105 : (b+1)*105] and channel c is
columns [b*105 + c for b in range(8)]. Bands: t1 t2 a1 a2 b1 b2 g1 g2.

Usage (from repo root):
    python tests/eeg_signal_probe_extended.py [processed_dir] [--out results.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import (
    TfidfVectorizer, ENGLISH_STOP_WORDS)
from sklearn.decomposition import TruncatedSVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
N_CH = 105


# =============================================================================
# Loaders
# =============================================================================

def load_sentence_level(data, split):
    """Mean-pool each sentence's EEG over fixated words -> (N, 840)."""
    eeg = np.load(f"{data}/{split}_eeg.npy", allow_pickle=True)
    meta = json.load(open(f"{data}/{split}_meta.json"))
    pooled = np.zeros((len(eeg), 840), dtype=np.float32)
    for i, e in enumerate(eeg):
        e = np.asarray(e, dtype=np.float32)
        fix = np.asarray(meta[i]["has_fixation"][:e.shape[0]], dtype=bool)
        rows = e[fix] if fix.any() else e
        pooled[i] = rows.mean(axis=0) if len(rows) else 0.0
    return np.nan_to_num(pooled), meta


def load_word_level(data, split, max_words=60000):
    """Per fixated word -> (M, 840) plus word string and subject id."""
    eeg = np.load(f"{data}/{split}_eeg.npy", allow_pickle=True)
    meta = json.load(open(f"{data}/{split}_meta.json"))
    X, words, subjects = [], [], []
    for i, e in enumerate(eeg):
        e = np.asarray(e, dtype=np.float32)
        n = e.shape[0]
        ws = meta[i]["words"][:n]
        fix = np.asarray(meta[i]["has_fixation"][:n], dtype=bool)
        sid = meta[i]["subject_id"]
        for j in range(n):
            if not fix[j]:
                continue
            X.append(e[j])
            words.append(str(ws[j]))
            subjects.append(sid)
        if len(X) >= max_words:
            break
    return np.nan_to_num(np.asarray(X, dtype=np.float32)), words, subjects


def acc(pred, gold):
    return float(np.mean(np.asarray(pred) == np.asarray(gold)))


# =============================================================================
# Retrieval probe (parameterised on a feature-column subset)
# =============================================================================

def retrieval_topk(Xtr, Xte, mtr, mte, cols=None, svd_dim=128, k=10,
                   per_subject=False):
    """EEG -> sentence-semantics retrieval restricted to feature columns `cols`.

    Returns dict with top-1 / top-k / median rank (and per-subject top-k if
    requested). Mirrors Probe 3 in eeg_signal_probe.py.
    """
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    txt_tr = [m["text"] for m in mtr]
    txt_te = [m["text"] for m in mte]
    tfidf = TfidfVectorizer(stop_words="english", min_df=2).fit(txt_tr + txt_te)
    n_comp = min(svd_dim, tfidf.transform(txt_tr).shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(
        tfidf.transform(txt_tr))

    def embed(texts):
        v = svd.transform(tfidf.transform(texts))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    reg = Ridge(alpha=10.0).fit(Xtr, embed(txt_tr))
    pred = reg.predict(Xte)
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)

    uniq = sorted(set(txt_te))
    gal = embed(uniq)
    gal_idx = {t: i for i, t in enumerate(uniq)}
    sims = pred @ gal.T
    order = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(order[i] == gal_idx[txt_te[i]])[0][0]) + 1
                      for i in range(len(txt_te))])

    res = {
        "gallery": len(uniq),
        "top1": float(np.mean(ranks == 1)),
        "topk": float(np.mean(ranks <= k)),
        "median_rank": int(np.median(ranks)),
        "chance_topk": k / len(uniq),
    }
    if per_subject:
        subs = np.array([m["subject_id"] for m in mte])
        res["per_subject"] = {
            s: float(np.mean(ranks[subs == s] <= k))
            for s in sorted(set(subs))
        }
    return res


# =============================================================================
# Studies
# =============================================================================

def band_cols(b):
    return list(range(b * N_CH, (b + 1) * N_CH))


def channel_cols(c):
    return [b * N_CH + c for b in range(8)]


def study_bands(Xtr, Xte, mtr, mte):
    print("\n=== A. Frequency-band ablation (retrieval top-10) ===")
    out = {}
    for b, name in enumerate(BANDS):
        r = retrieval_topk(Xtr, Xte, mtr, mte, cols=band_cols(b))
        out[name] = r
        print(f"  {name}: top10={r['topk']:.3f}  top1={r['top1']:.3f}  "
              f"med_rank={r['median_rank']}  (chance {r['chance_topk']:.3f})")
    full = retrieval_topk(Xtr, Xte, mtr, mte)
    out["ALL"] = full
    print(f"  ALL bands: top10={full['topk']:.3f}  top1={full['top1']:.3f}")
    return out


def study_channels(Xtr, Xte, mtr, mte, top_report=15):
    print("\n=== B. Per-channel importance (retrieval top-10, montage-free) ===")
    scores = []
    for c in range(N_CH):
        r = retrieval_topk(Xtr, Xte, mtr, mte, cols=channel_cols(c), svd_dim=64)
        scores.append((c, r["topk"]))
    scores.sort(key=lambda x: -x[1])
    print(f"  best {top_report} channels (index: top10):")
    print("   " + "  ".join(f"ch{c}:{s:.2f}" for c, s in scores[:top_report]))
    return {"per_channel_top10": {str(c): s for c, s in scores}}


def study_word_class(data):
    print("\n=== C. Word-class probes (per fixated word) ===")
    Xtr, wtr, _ = load_word_level(data, "train")
    Xte, wte, _ = load_word_level(data, "test")
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}

    def is_function(w):
        return w.lower().strip(".,;:!?\"'()") in ENGLISH_STOP_WORDS

    ytr = [is_function(w) for w in wtr]
    yte = [is_function(w) for w in wte]
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr_s, ytr)
    maj = max(set(yte), key=yte.count)
    out["content_vs_function"] = {
        "accuracy": acc(clf.predict(Xte_s), yte),
        "majority": acc([maj] * len(yte), yte),
        "n_train": len(ytr), "n_test": len(yte),
        "frac_function_test": float(np.mean(yte)),
    }
    print(f"  content/function: acc={out['content_vs_function']['accuracy']:.3f} "
          f"(majority {out['content_vs_function']['majority']:.3f})")

    # Word length regression (R^2 on held-out)
    Ltr = np.array([len(w) for w in wtr], dtype=np.float32)
    Lte = np.array([len(w) for w in wte], dtype=np.float32)
    reg = Ridge(alpha=10.0).fit(Xtr_s, Ltr)
    pred = reg.predict(Xte_s)
    ss_res = float(np.sum((Lte - pred) ** 2))
    ss_tot = float(np.sum((Lte - Lte.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    out["word_length_r2"] = r2
    print(f"  word length R^2 = {r2:.3f}")
    return out


def study_subjects(Xtr, Xte, mtr, mte):
    print("\n=== D. Per-subject retrieval quality (top-10) ===")
    r = retrieval_topk(Xtr, Xte, mtr, mte, per_subject=True)
    ps = r["per_subject"]
    for s in sorted(ps, key=lambda k: -ps[k]):
        print(f"  {s}: {ps[s]:.3f}")
    print(f"  overall top10={r['topk']:.3f}  (chance {r['chance_topk']:.3f})")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="processed_zuco2")
    p.add_argument("--out", default="exp_logs/e4_signal_probes.json")
    p.add_argument("--skip_channels", action="store_true",
                   help="channel study is 105 retrieval fits; skip for speed")
    args = p.parse_args()

    data = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    print(f"Loading {data} ...")
    Xtr, mtr = load_sentence_level(data, "train")
    Xte, mte = load_sentence_level(data, "test")
    print(f"  train {Xtr.shape}  test {Xte.shape}")

    results = {"data": os.path.basename(data)}
    results["bands"] = study_bands(Xtr, Xte, mtr, mte)
    if not args.skip_channels:
        results["channels"] = study_channels(Xtr, Xte, mtr, mte)
    results["word_class"] = study_word_class(data)
    results["subjects"] = study_subjects(Xtr, Xte, mtr, mte)

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
