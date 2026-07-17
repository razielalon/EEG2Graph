"""
E9 — Bridge/encoder retrieval probe (RQ1/RQ2): does the TRAINED bridge preserve
the sentence-identity signal E4 found in the raw EEG, or destroy it?

Motivation
----------
E4 showed a linear probe on the *raw* 840-dim EEG retrieves the read sentence at
~2.6x chance (top-10 0.394). E7 showed the *trained bridge*'s alignment loss never
beats its InfoNCE chance floor. Those two look contradictory ("signal exists" vs
"bridge learns nothing") but they measure different things. E9 removes the
ambiguity by running E4's *identical* retrieval probe on the representation the
model actually produces, so the ONLY thing that changes between conditions is the
feature matrix:

    raw            : E4's mean-pooled raw EEG                    (reference, ~0.394)
    random_bridge  : untrained Linear(840->1024)+LayerNorm      (control, ~=raw)
    bridge         : a trained checkpoint's bridge output        (verdict)
    encoder        : REBEL encoder states over the bridge output (what the decoder reads)
    text           : REBEL's own text-encoder states (gold text) (upper bound, ~1.0)

Decision rule
-------------
  bridge/encoder ~= raw  -> training PRESERVED the signal; the wall is downstream
                            (frozen decoder / CE+InfoNCE objective can't consume it)
                            => "information present but unusable" — a different, more
                            tractable thesis than "no signal".
  bridge/encoder ~= chance-> training DESTROYED the signal (bridge collapsed);
                            the objective actively harms a signal that was there.
Unlike triplet-F1 (pinned at 0 by exact match) this metric has real dynamic range.

The retrieval core (`retrieval_topk`) and the raw loader are copied VERBATIM from
tests/eeg_signal_probe_extended.py (E4) so the comparison is exact.

Usage
-----
  # local, no torch needed (baselines only):
  python model/bridge_retrieval_probe.py --processed_dir processed_zuco \
      --sources raw,random_bridge --out exp_logs/e9_retrieval.json

  # cluster, with checkpoints (adds bridge/encoder/text):
  python model/bridge_retrieval_probe.py --processed_dir ../processed_zuco \
      --sources raw,random_bridge,bridge,encoder,text \
      --checkpoints checkpoints_e8_k0 checkpoints_e5_s1t0aPL:s1t0aPL ... \
      --out ../exp_logs/e9_retrieval.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =============================================================================
# E4 retrieval core — copied verbatim from tests/eeg_signal_probe_extended.py
# so E9 is byte-comparable with E4. Only the input feature matrix `X*` varies.
# =============================================================================
def retrieval_topk(Xtr, Xte, mtr, mte, cols=None, svd_dim=128, k=10):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
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

    uniq = sorted(set(txt_te))
    gal = embed(uniq)
    gal_idx = {t: i for i, t in enumerate(uniq)}
    sims = pred @ gal.T
    order = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(order[i] == gal_idx[txt_te[i]])[0][0]) + 1
                      for i in range(len(txt_te))])
    return {
        "gallery": len(uniq),
        "top1": float(np.mean(ranks == 1)),
        "topk": float(np.mean(ranks <= k)),
        "median_rank": int(np.median(ranks)),
        "chance_topk": k / len(uniq),
        "chance_top1": 1.0 / len(uniq),
    }


# =============================================================================
# Numpy feature builders (no torch) — `raw` and `random_bridge`
# =============================================================================
def _iter_sentences(data, split):
    """Yield (fixated word matrix (n_fix, 840), meta) per sentence.

    Mirrors E4's load_sentence_level: fixated words only, NaN-safe.
    """
    eeg = np.load(os.path.join(data, f"{split}_eeg.npy"), allow_pickle=True)
    meta = json.load(open(os.path.join(data, f"{split}_meta.json")))
    for i, e in enumerate(eeg):
        e = np.asarray(e, dtype=np.float32)
        fix = np.asarray(meta[i]["has_fixation"][:e.shape[0]], dtype=bool)
        rows = e[fix] if fix.any() else e
        yield np.nan_to_num(rows), meta[i]


def make_random_bridge(din=840, dout=1024, seed=0):
    """Untrained Linear(din->dout) [nn.Linear default init] + LayerNorm, numpy."""
    rng = np.random.default_rng(seed)
    bound = 1.0 / np.sqrt(din)
    W = rng.uniform(-bound, bound, size=(din, dout)).astype(np.float32)
    b = rng.uniform(-bound, bound, size=(dout,)).astype(np.float32)

    def fn(x):  # (n, din) -> (n, dout)
        y = x @ W + b
        mu = y.mean(axis=1, keepdims=True)
        var = y.var(axis=1, keepdims=True)
        return (y - mu) / np.sqrt(var + 1e-5)
    return fn


def numpy_features(data, split, transform=None):
    """Mean-pool (optionally transformed) fixated-word features -> (N, d)."""
    X, M = [], []
    for rows, m in _iter_sentences(data, split):
        r = transform(rows) if transform is not None else rows
        if len(r) == 0:
            d = r.shape[1] if r.ndim == 2 else 840
            X.append(np.zeros(d, dtype=np.float32))
        else:
            X.append(r.mean(axis=0))
        M.append(m)
    return np.asarray(X, dtype=np.float32), M


# =============================================================================
# Torch feature builders (lazy imports) — `bridge`, `encoder`, `text`
# =============================================================================
def load_probe_model(ckpt_path, tokenizer, device):
    """Instantiate EEGBartModel with the checkpoint's saved arch flags and load
    weights. Mirrors inference.load_model (tier2-grid)."""
    import torch
    from eeg_graph_model import EEGBartModel

    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get("args", {})
    bart_name = ckpt.get("bart_name") or a.get("bart_name", "Babelscape/rebel-large")
    model = EEGBartModel(
        tokenizer=tokenizer,
        eeg_dim=a.get("eeg_dim", 840),
        bart_name=bart_name,
        dropout=0.0,
        bridge_layers=a.get("bridge_layers", 1),
        n_subject_buckets=a.get("n_subject_buckets", 0),
        bridge_transformer_layers=a.get("bridge_transformer_layers", 0),
        bridge_nhead=a.get("bridge_nhead", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_subj = int(a.get("n_subject_buckets", 0) or 0)
    return model, n_subj, ckpt


def _masked_mean(states, mask):
    """states (B,S,d), mask (B,S) bool/long -> (B,d) mean over True positions."""
    import torch
    m = mask.bool().unsqueeze(-1).float()
    s = (states * m).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1.0)
    return s / n


def torch_features(model, n_subj, data, split, source, tokenizer, device,
                   batch_size=16, max_src=None, max_tgt=64):
    """Return (X (N,d), meta) for source in {bridge, encoder, text}."""
    import torch
    from eeg_graph_dataset import fixation_attention_mask, subject_bucket

    eeg = np.load(os.path.join(data, f"{split}_eeg.npy"), allow_pickle=True)
    meta = json.load(open(os.path.join(data, f"{split}_meta.json")))
    feats, M = [], []

    for start in range(0, len(eeg), batch_size):
        batch = list(range(start, min(start + batch_size, len(eeg))))
        eeg_list = [np.asarray(eeg[i], dtype=np.float32) for i in batch]
        meta_list = [meta[i] for i in batch]

        if source == "text":
            texts = [m.get("text", "") for m in meta_list]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            ids = enc["input_ids"].to(device)
            amask = enc["attention_mask"].to(device)
            with torch.no_grad():
                states = model.encode_text(ids, amask)
            pooled = _masked_mean(states, amask)
        else:
            ms = max(e.shape[0] for e in eeg_list)
            if max_src:
                ms = min(ms, max_src)
            B, dim = len(eeg_list), eeg_list[0].shape[1]
            src = torch.zeros(B, ms, dim, device=device)
            real = torch.zeros(B, ms, dtype=torch.bool, device=device)
            fix = torch.zeros(B, ms, dtype=torch.bool, device=device)
            for i, e in enumerate(eeg_list):
                n = min(e.shape[0], ms)
                src[i, :n] = torch.tensor(np.nan_to_num(e[:n]), dtype=torch.float32)
                real[i, :n] = True
                hf = meta_list[i].get("has_fixation", [True] * n)[:n]
                fix[i, :len(hf)] = torch.tensor(hf, dtype=torch.bool, device=device)
            src_mask = fixation_attention_mask(real, fix)

            subject_idx = None
            if n_subj > 0:
                subject_idx = torch.tensor(
                    [subject_bucket(m["subject_id"], n_subj) for m in meta_list],
                    device=device)

            with torch.no_grad():
                bridge_out = model._bridge_forward(src, src_mask, subject_idx)
                if source == "bridge":
                    states = bridge_out
                elif source == "encoder":
                    states = model.bart.model.encoder(
                        inputs_embeds=bridge_out,
                        attention_mask=src_mask.long(),
                    ).last_hidden_state
                else:
                    raise ValueError(source)
            pooled = _masked_mean(states, src_mask)

        feats.append(pooled.float().cpu().numpy())
        M.extend(meta_list)

    return np.concatenate(feats, axis=0), M


# =============================================================================
# Orchestration
# =============================================================================
def _parse_ckpt(spec):
    """'dir' or 'dir:label' -> (label, dir)."""
    if ":" in spec and not (len(spec) > 1 and spec[1] == ":"):  # avoid C:\ on win
        d, label = spec.rsplit(":", 1)
        return label, d
    return os.path.basename(spec.rstrip("/\\")), spec


def main():
    p = argparse.ArgumentParser(description="E9 bridge/encoder retrieval probe")
    p.add_argument("--processed_dir", default="processed_zuco")
    p.add_argument("--sources", default="raw,random_bridge",
                   help="comma list of raw,random_bridge,bridge,encoder,text")
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="checkpoint dirs (each has best_model.pt + tokenizer/); "
                        "optionally dir:label")
    p.add_argument("--tokenizer_dir", default=None,
                   help="tokenizer dir; defaults to first checkpoint's tokenizer/ "
                        "or Babelscape/rebel-large")
    p.add_argument("--random_seeds", default="0,1,2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="exp_logs/e9_retrieval.json")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    torch_sources = [s for s in sources if s in ("bridge", "encoder", "text")]
    data = args.processed_dir
    results = {"data": os.path.basename(os.path.normpath(data)),
               "sources": sources, "baselines": {}, "checkpoints": {}}

    # ---- numpy baselines (always available) --------------------------------
    Mtr = Mte = None
    if "raw" in sources:
        Xtr, Mtr = numpy_features(data, "train")
        Xte, Mte = numpy_features(data, "test")
        r = retrieval_topk(Xtr, Xte, Mtr, Mte)
        results["baselines"]["raw"] = r
        print(f"[raw]            top1={r['top1']:.3f} top10={r['topk']:.3f} "
              f"med={r['median_rank']} (chance10={r['chance_topk']:.3f}, gallery={r['gallery']})")
    if "random_bridge" in sources:
        seeds = [int(s) for s in args.random_seeds.split(",")]
        per = []
        for sd in seeds:
            fn = make_random_bridge(seed=sd)
            Xtr, mtr = numpy_features(data, "train", fn)
            Xte, mte = numpy_features(data, "test", fn)
            per.append(retrieval_topk(Xtr, Xte, mtr, mte))
        results["baselines"]["random_bridge"] = {
            "seeds": seeds,
            "top1_mean": float(np.mean([x["top1"] for x in per])),
            "top10_mean": float(np.mean([x["topk"] for x in per])),
            "top10_std": float(np.std([x["topk"] for x in per])),
            "median_rank_mean": float(np.mean([x["median_rank"] for x in per])),
            "per_seed": per,
        }
        b = results["baselines"]["random_bridge"]
        print(f"[random_bridge]  top1={b['top1_mean']:.3f} top10={b['top10_mean']:.3f}"
              f"+/-{b['top10_std']:.3f} med={b['median_rank_mean']:.0f}")

    # ---- torch sources (need checkpoints / stock REBEL) --------------------
    if torch_sources:
        import torch
        from vocabulary import load_tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        tok_dir = args.tokenizer_dir
        if tok_dir is None and args.checkpoints:
            cand = os.path.join(_parse_ckpt(args.checkpoints[0])[1], "tokenizer")
            tok_dir = cand if os.path.isdir(cand) else None
        tokenizer = load_tokenizer(tok_dir) if tok_dir else _load_stock_tok()

        text_done = False
        for spec in args.checkpoints:
            label, cdir = _parse_ckpt(spec)
            ckpt_path = os.path.join(cdir, "best_model.pt")
            if not os.path.isfile(ckpt_path):
                print(f"  !! missing {ckpt_path}, skipping")
                results["checkpoints"][label] = {"dir": cdir, "error": "missing best_model.pt"}
                continue
            print(f"\n=== checkpoint {label} ({ckpt_path}) ===")
            # Per-checkpoint isolation: one bad checkpoint must not abort the
            # whole (multi-hour, 19-checkpoint) job. Record the error and go on;
            # partial results are still written at the end.
            model = None
            try:
                model, n_subj, ckpt = load_probe_model(ckpt_path, tokenizer, device)
                entry = {"dir": cdir, "epoch": ckpt.get("epoch"),
                         "val_f1": ckpt.get("val_f1"), "n_subject_buckets": n_subj}
                for src in ("bridge", "encoder"):
                    if src in sources:
                        Xtr, mtr = torch_features(model, n_subj, data, "train", src,
                                                  tokenizer, device, args.batch_size)
                        Xte, mte = torch_features(model, n_subj, data, "test", src,
                                                  tokenizer, device, args.batch_size)
                        r = retrieval_topk(Xtr, Xte, mtr, mte)
                        entry[src] = r
                        print(f"  [{src:7}] top1={r['top1']:.3f} top10={r['topk']:.3f} "
                              f"med={r['median_rank']}")
                # text upper bound: REBEL weights are identical across frozen
                # ckpts, so compute it once from the first model that succeeds.
                if "text" in sources and not text_done:
                    Xtr, mtr = torch_features(model, n_subj, data, "train", "text",
                                              tokenizer, device, args.batch_size)
                    Xte, mte = torch_features(model, n_subj, data, "test", "text",
                                              tokenizer, device, args.batch_size)
                    r = retrieval_topk(Xtr, Xte, mtr, mte)
                    results["baselines"]["text_upper_bound"] = r
                    text_done = True
                    print(f"  [text UB] top1={r['top1']:.3f} top10={r['topk']:.3f} "
                          f"med={r['median_rank']}")
                results["checkpoints"][label] = entry
            except Exception as ex:  # noqa: BLE001 - keep the job alive per ckpt
                print(f"  !! FAILED on {label}: {type(ex).__name__}: {ex}")
                results["checkpoints"][label] = {"dir": cdir, "error": f"{type(ex).__name__}: {ex}"}
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_md(results, os.path.splitext(out)[0] + ".md")
    print(f"\nSaved -> {out}")


def _load_stock_tok():
    from vocabulary import build_tokenizer
    return build_tokenizer("Babelscape/rebel-large")


def _write_md(results, path):
    lines = [f"# E9 - Bridge/encoder retrieval probe: results ({results['data']})", ""]
    b = results["baselines"]
    lines += ["## Baselines (shared across checkpoints)", "",
              "| source | top1 | top10 | median rank |", "|---|---|---|---|"]
    if "raw" in b:
        r = b["raw"]
        lines.append(f"| raw EEG (E4) | {r['top1']:.3f} | {r['topk']:.3f} | {r['median_rank']} |")
        lines.append(f"| *chance* | {r['chance_top1']:.3f} | {r['chance_topk']:.3f} | {(r['gallery']+1)//2} |")
    if "random_bridge" in b:
        rb = b["random_bridge"]
        lines.append(f"| random_bridge (untrained) | {rb['top1_mean']:.3f} | "
                     f"{rb['top10_mean']:.3f} | {rb['median_rank_mean']:.0f} |")
    if "text_upper_bound" in b:
        t = b["text_upper_bound"]
        lines.append(f"| text encoder (REBEL, upper bound) | {t['top1']:.3f} | {t['topk']:.3f} | {t['median_rank']} |")
    lines += ["", "## Trained checkpoints", "",
              "| label | val_f1 | bridge top10 | encoder top10 | bridge top1 | encoder top1 |",
              "|---|---|---|---|---|---|"]
    for label, e in results["checkpoints"].items():
        bt = e.get("bridge", {})
        en = e.get("encoder", {})
        lines.append(
            f"| {label} | {e.get('val_f1')} | "
            f"{bt.get('topk', float('nan')):.3f} | {en.get('topk', float('nan')):.3f} | "
            f"{bt.get('top1', float('nan')):.3f} | {en.get('top1', float('nan')):.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
