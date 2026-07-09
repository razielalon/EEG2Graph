#!/usr/bin/env python
"""
E8 — cross-subject (RQ5): aggregate the paired excl/ctrl points.

E5 (architecture) and E6 (data scale) both left the frozen-BART ~0 ceiling in
place. E8 asks whether the decodable EEG->triplet code is subject-SHARED or
subject-SPECIFIC, using a paired design that strips out the sample-count confound:

  excl arm (x03/x06/x09): drop N whole subjects from TRAIN.
  ctrl arm (c03/c06/c09): drop a MATCHED sample count, keep all 18 subjects.
  k0: shared baseline (full train, all subjects; == E5 cell s0t0aPL).

The subject-specificity signal at each level is  F1(ctrl) - F1(excl)  at matched
data size: excl << ctrl => subject-specific codes; excl ~= ctrl => subject-shared.

This scans checkpoints_e8_<TAG>/ and reads:
  test_metrics.json    -> aggregate test f1/precision/recall (train.py's number)
  test_by_subject.json -> seen vs unseen (held-out) subject F1 (cross_subject_eval)
  history.json         -> best val f1 (+epoch), align at best/final, epochs run

Writes exp_logs/e8_cross_subject.{json,md}. Missing/unfinished points are PENDING
(safe to run mid-queue).

Usage:
    cd model && python aggregate_e8.py
    python aggregate_e8.py --ckpt_root .. --out_dir ../exp_logs
"""
import os
import json
import argparse

# Canonical point order. Each excl point is paired to the ctrl point at the same
# kept-sample fraction (frac = kept/10216 for the nested held-out subject sets).
BASELINE = ("k0", None, "baseline", 1.000)
POINTS = [
    BASELINE,
    ("x03", 3, "excl", 0.827),
    ("x06", 6, "excl", 0.664),
    ("x09", 9, "excl", 0.491),
    ("c03", 3, "ctrl", 0.827),
    ("c06", 6, "ctrl", 0.664),
    ("c09", 9, "ctrl", 0.491),
]
# excl<->ctrl pairing by matched fraction, for the read-off.
PAIRS = [("x03", "c03", 0.827), ("x06", "c06", 0.664), ("x09", "c09", 0.491)]


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_point(ckpt_root, tag, n_held, arm, frac):
    d = os.path.join(ckpt_root, f"checkpoints_e8_{tag}")
    row = {"tag": tag, "n_held": n_held, "arm": arm, "frac": frac,
           "status": "PENDING", "dir": d}
    if not os.path.isdir(d):
        return row

    history = _load_json(os.path.join(d, "history.json"))
    metrics = _load_json(os.path.join(d, "test_metrics.json"))
    by_subj = _load_json(os.path.join(d, "test_by_subject.json"))

    if history:
        best = max(history, key=lambda e: e.get("val_f1", 0))
        row["best_val_f1"] = round(best.get("val_f1", 0), 4)
        row["best_val_epoch"] = best.get("epoch")
        row["align_final"] = round(history[-1].get("train_align", 0), 4)
        row["epochs_run"] = len(history)
        row["status"] = "TRAINED"

    if metrics:
        row["test_f1"] = round(metrics.get("f1", 0), 4)
        row["test_precision"] = round(metrics.get("precision", 0), 4)
        row["test_recall"] = round(metrics.get("recall", 0), 4)
        row["status"] = "DONE"

    if by_subj:
        row["seen_f1"] = round((by_subj.get("seen") or {}).get("f1", 0), 4)
        unseen = by_subj.get("unseen")
        row["unseen_f1"] = round(unseen["f1"], 4) if unseen else None

    return row


def fmt(v, spec="{:.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "—"


def to_markdown(rows):
    by_tag = {r["tag"]: r for r in rows}
    done = [r for r in rows if r["status"] == "DONE"]
    pending = [r for r in rows if r["status"] != "DONE"]

    L = []
    L.append("# E8 — Cross-subject: shared vs subject-specific EEG codes (RQ5): results\n")
    L.append("Frozen BART, subject embedding **OFF**, temporal OFF, pooled align, "
             "seed 42 (k0 == E5 cell `s0t0aPL`). Paired design: the **excl** arm "
             "drops whole subjects from TRAIN; the **ctrl** arm drops a matched "
             "sample count but keeps all 18 subjects. `unseen F1` = test F1 on the "
             "held-out subjects only (via `cross_subject_eval.py`).\n")
    L.append(f"**Completed: {len(done)}/{len(POINTS)} points.**\n")

    L.append("| TAG | arm | held-out | kept frac | test F1 | seen F1 | unseen F1 "
             "| best val F1 (ep) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ep = r.get("best_val_epoch")
        valcell = (f"{fmt(r.get('best_val_f1'))} ({ep})"
                   if "best_val_f1" in r else "PENDING")
        held = "—" if r["n_held"] is None else r["n_held"]
        L.append(
            f"| {r['tag']} | {r['arm']} | {held} | {r['frac']:.3f} "
            f"| {fmt(r.get('test_f1'))} | {fmt(r.get('seen_f1'))} "
            f"| {fmt(r.get('unseen_f1'))} | {valcell} |"
        )

    if pending:
        tags = ", ".join(f"{r['tag']} ({r['status']})" for r in pending)
        L.append(f"\n_Pending: {tags}._")

    # Read-off: paired excl-vs-ctrl deltas at matched data size.
    ready = [(x, c, f) for x, c, f in PAIRS
             if by_tag.get(x, {}).get("status") == "DONE"
             and by_tag.get(c, {}).get("status") == "DONE"]
    if ready:
        L.append("\n## Read-off — subject-specificity penalty (ctrl − excl)\n")
        L.append("At matched kept-sample fraction, `F1(ctrl) − F1(excl)` isolates "
                 "the cost of removing whole *subjects* (vs random samples). "
                 "Positive & growing with held-out count ⇒ subject-**specific** "
                 "codes; ≈0 ⇒ subject-**shared** (only sample count matters).\n")
        L.append("| kept frac | held-out | excl F1 | ctrl F1 | ctrl − excl | unseen F1 |")
        L.append("|---|---|---|---|---|---|")
        for x, c, f in ready:
            rx, rc = by_tag[x], by_tag[c]
            delta = rc["test_f1"] - rx["test_f1"]
            L.append(f"| {f:.3f} | {rx['n_held']} | {fmt(rx['test_f1'])} "
                     f"| {fmt(rc['test_f1'])} | {delta:+.4f} "
                     f"| {fmt(rx.get('unseen_f1'))} |")

        deltas = [by_tag[c]["test_f1"] - by_tag[x]["test_f1"] for x, c, f in ready]
        biggest = max(deltas)
        L.append(
            f"\n- Largest ctrl−excl gap: {biggest:+.4f}. A gap near 0 across all "
            "levels says the frozen ~0 ceiling is **not** a cross-subject-transfer "
            "artifact — codes are shared (or, at floor, equally undecodable) and "
            "removing subjects costs no more than removing the same #samples. A gap "
            "that grows with held-out count says the codes are subject-specific: a "
            "model never trained on a person cannot decode them. Either way, with "
            "E5 (architecture) and E6 (data) this closes the RQ4/RQ5 arc.")
        if any(by_tag[x].get("unseen_f1") not in (None, "—") for x, c, f in ready):
            L.append(
                "- `unseen F1` (held-out subjects only) is the direct "
                "generalize-to-a-new-person number; compare it to `seen F1` at the "
                "same point — a seen≫unseen split is the sharpest subject-specificity "
                "signal, undiluted by the seen majority.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="..",
                    help="Dir containing the checkpoints_e8_<TAG>/ folders.")
    ap.add_argument("--out_dir", default="../exp_logs",
                    help="Where to write e8_cross_subject.{json,md}.")
    args = ap.parse_args()

    rows = [read_point(args.ckpt_root, tag, n, arm, frac)
            for tag, n, arm, frac in POINTS]
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "e8_cross_subject.json")
    md_path = os.path.join(args.out_dir, "e8_cross_subject.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(md_path, "w") as f:
        f.write(to_markdown(rows))

    done = sum(1 for r in rows if r["status"] == "DONE")
    print(f"Wrote {json_path} and {md_path}  ({done}/{len(POINTS)} points DONE)")
    for r in rows:
        print(f"  {r['tag']:4s} [{r['arm']:8s}] {r['status']:8s} "
              f"test_F1={fmt(r.get('test_f1'))}  unseen_F1={fmt(r.get('unseen_f1'))}")


if __name__ == "__main__":
    main()
