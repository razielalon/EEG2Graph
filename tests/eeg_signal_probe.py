"""
REBEL-free EEG signal probe.

The overfit tests (train_overfit*.sbatch) can only tell us the bridge -> REBEL
pipeline fails to learn. They cannot tell us whether the EEG itself carries
decodable signal, because both the frozen and unfrozen runs are confounded by
the REBEL conditioning path. This probe bypasses REBEL entirely: simple linear
models, trained on the train split, evaluated on the held-out test split
(genuine generalization).

  Probe 1  task decoding      (NR vs TSR)             -- gross cognitive signal
  Probe 2  subject decoding   (18-way)               -- recording-level structure
  Probe 3  sentence retrieval (EEG -> text-semantics) -- the real question

If even Probe 1 sits at chance, the preprocessed EEG features are noise.
If Probe 3 beats chance (and the shuffled-pairing control collapses to
chance), there is decodable semantic signal and the bottleneck is the REBEL
conditioning pipeline, not the data.

Usage:
    python tests/eeg_signal_probe.py [processed_dir]      # default: processed_zuco2
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else "processed_zuco2"
if not os.path.isabs(DATA):
    DATA = os.path.join(ROOT, DATA)


def load_split(split):
    """Mean-pool each sentence's EEG over its fixated words -> one 840-d vector."""
    eeg = np.load(f"{DATA}/{split}_eeg.npy", allow_pickle=True)
    meta = json.load(open(f"{DATA}/{split}_meta.json"))
    pooled = np.zeros((len(eeg), 840), dtype=np.float32)
    for i, e in enumerate(eeg):
        e = np.asarray(e, dtype=np.float32)
        fix = np.asarray(meta[i]["has_fixation"][:e.shape[0]], dtype=bool)
        rows = e[fix] if fix.any() else e
        pooled[i] = rows.mean(axis=0) if len(rows) else 0.0
    return np.nan_to_num(pooled), meta


print(f"Loading {DATA} ...")
Xtr, mtr = load_split("train")
Xte, mte = load_split("test")
print(f"  train {Xtr.shape}  test {Xte.shape}")

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)


def acc(pred, gold):
    return float(np.mean(np.asarray(pred) == np.asarray(gold)))


# ---- Probe 1: task decoding (NR vs TSR) ----
ytr = [m["task"] for m in mtr]
yte = [m["task"] for m in mte]
clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr_s, ytr)
maj = max(set(yte), key=yte.count)
print("\n=== Probe 1: task decoding (NR vs TSR) ===")
print(f"  test accuracy : {acc(clf.predict(Xte_s), yte):.3f}")
print(f"  majority-class: {acc([maj] * len(yte), yte):.3f}  <- chance")

# ---- Probe 2: subject decoding ----
ytr = [m["subject_id"] for m in mtr]
yte = [m["subject_id"] for m in mte]
clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr_s, ytr)
n_sub = len(set(ytr))
print(f"\n=== Probe 2: subject decoding ({n_sub}-way) ===")
print(f"  test accuracy : {acc(clf.predict(Xte_s), yte):.3f}")
print(f"  chance (1/N)  : {1.0 / n_sub:.3f}")

# ---- Probe 3: EEG -> sentence-semantics retrieval ----
txt_tr = [m["text"] for m in mtr]
txt_te = [m["text"] for m in mte]
tfidf = TfidfVectorizer(stop_words="english", min_df=2).fit(txt_tr + txt_te)
svd = TruncatedSVD(n_components=128, random_state=0).fit(tfidf.transform(txt_tr))


def embed(texts):
    v = svd.transform(tfidf.transform(texts))
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


Ttr = embed(txt_tr)
reg = Ridge(alpha=10.0).fit(Xtr_s, Ttr)
pred = reg.predict(Xte_s)
pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)

uniq = sorted(set(txt_te))
gal = embed(uniq)
gal_idx = {t: i for i, t in enumerate(uniq)}


def ranks_of(prediction):
    sims = prediction @ gal.T
    order = np.argsort(-sims, axis=1)
    return np.array([
        int(np.where(order[i] == gal_idx[txt_te[i]])[0][0]) + 1
        for i in range(len(txt_te))
    ])


ranks = ranks_of(pred)
print("\n=== Probe 3: EEG -> sentence retrieval ===")
print(f"  gallery size (unique test sentences): {len(uniq)}")
print(f"  top-1   : {np.mean(ranks == 1):.3f}   (chance {1 / len(uniq):.4f})")
print(f"  top-10  : {np.mean(ranks <= 10):.3f}   (chance {10 / len(uniq):.4f})")
print(f"  median rank : {int(np.median(ranks))}   (chance {len(uniq) // 2})")

# control: shuffle EEG<->text pairing -> must collapse to chance
perm = np.random.default_rng(0).permutation(len(Xte_s))
predc = reg.predict(Xte_s[perm])
predc = predc / (np.linalg.norm(predc, axis=1, keepdims=True) + 1e-8)
ranksc = ranks_of(predc)
print(f"  [shuffled control] top-10: {np.mean(ranksc <= 10):.3f}  "
      f"median rank: {int(np.median(ranksc))}")
