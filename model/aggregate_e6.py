#!/usr/bin/env python
"""
E6 — data-scaling curve (RQ4 follow-up): aggregate the fraction points.

E5 showed the frozen-BART ceiling is architecture-independent (~0 F1 across the
2x2x2 grid). E6 holds the best E5 cell fixed (#2: subject=64, temporal=0,
align=pooled, frozen, seed 42) and sweeps `--train_sentence_frac`. This scans the
per-point output dirs written by train.py (`checkpoints_e6_<TAG>/`) and builds the
data-scaling table + a slope read-off.

Each point contributes:
  test_metrics.json  -> test f1 / precision / recall (+ n_pred/n_gold/n_correct)
  history.json       -> best val_f1 (and its epoch), align loss at best + final,
                        and the actual #train samples kept (if train.py logs it)

TAG grammar (set by exp_e6_scale.sbatch): f{frac*100 as 3 digits}
  f010 = --train_sentence_frac 0.1   f025 = 0.25
  f050 = 0.5                         f100 = 1.0 (== E5 cell 2)

Writes exp_logs/e6_data_scaling.{json,md}. Points whose dir is missing or whose
run hasn't reached test eval yet are listed as PENDING (safe to run mid-queue).

Usage:
    cd model && python aggregate_e6.py
    python aggregate_e6.py --ckpt_root .. --out_dir ../exp_logs
"""
import os
import re
import json
import argparse

# Canonical point order (small -> large), TAG + fraction.
POINTS = [
    ("f010", 0.10),
    ("f025", 0.25),
    ("f050", 0.50),
    ("f100", 1.00),
]
TAG_RE = re.compile(r"^f(\d{3})$")


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"bad TAG {tag!r}")
    return int(m.group(1)) / 100.0


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_point(ckpt_root, tag, frac):
    d = os.path.join(ckpt_root, f"checkpoints_e6_{tag}")
    row = {"tag": tag, "frac": frac, "status": "PENDING", "dir": d}
    if not os.path.isdir(d):
        return row

    metrics = _load_json(os.path.join(d, "test_metrics.json"))
    history = _load_json(os.path.join(d, "history.json"))

    if history:
        best = max(history, key=lambda e: e.get("val_f1", 0))
        row["best_val_f1"] = round(best.get("val_f1", 0), 4)
        row["best_val_epoch"] = best.get("epoch")
        row["align_at_best"] = round(best.get("train_align", 0), 4)
        row["align_final"] = round(history[-1].get("train_align", 0), 4)
        row["epochs_run"] = len(history)
        # train.py may record the kept-sample count on the first history entry;
        # tolerate its absence.
        if "n_train" in history[0]:
            row["n_train"] = history[0]["n_train"]
        row["status"] = "TRAINED"

    if metrics:
        row["test_f1"] = round(metrics.get("f1", 0), 4)
        row["test_precision"] = round(metrics.get("precision", 0), 4)
        row["test_recall"] = round(metrics.get("recall", 0), 4)
        row["n_pred"] = metrics.get("n_pred")
        row["n_gold"] = metrics.get("n_gold")
        row["n_correct"] = metrics.get("n_correct")
        row["status"] = "DONE"

    return row


def fmt(v, spec="{:.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "—"


def to_markdown(rows):
    done = [r for r in rows if r["status"] == "DONE"]
    pending = [r for r in rows if r["status"] != "DONE"]

    lines = []
    lines.append("# E6 — Frozen-BART data-scaling curve (RQ4 follow-up): results\n")
    lines.append("Best E5 cell (#2: subject=64, temporal=0, align=pooled, frozen "
                 "BART, seed 42) held fixed; the only moving part is "
                 "`--train_sentence_frac` — the fraction of TRAIN *unique "
                 "sentences* seen. val/test stay full.\n")
    lines.append(f"**Completed: {len(done)}/{len(POINTS)} points.**\n")

    lines.append("| frac | test F1 | test P | test R | best val F1 (ep) "
                 "| align final | n_train |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        ep = r.get("best_val_epoch")
        valcell = (f"{fmt(r.get('best_val_f1'))} ({ep})"
                   if "best_val_f1" in r else "PENDING")
        lines.append(
            f"| {r['frac']:.2f} | {fmt(r.get('test_f1'))} "
            f"| {fmt(r.get('test_precision'))} | {fmt(r.get('test_recall'))} "
            f"| {valcell} | {fmt(r.get('align_final'))} "
            f"| {r.get('n_train', '—')} |"
        )

    if pending:
        tags = ", ".join(f"frac {r['frac']:.2f} ({r['status']})" for r in pending)
        lines.append(f"\n_Pending: {tags}._")

    if len(done) == len(POINTS):
        done_sorted = sorted(done, key=lambda r: r["frac"])
        f1s = [r["test_f1"] for r in done_sorted]
        lo, hi = f1s[0], f1s[-1]
        spread = max(f1s) - min(f1s)
        monotone = all(b >= a for a, b in zip(f1s, f1s[1:]))
        lines.append("\n## Read-off\n")
        lines.append(f"- Test F1 at frac 0.10 → 1.00: "
                     f"{', '.join(f'{v:.4f}' for v in f1s)}.")
        lines.append(f"- Range across the curve: {spread:.4f} "
                     f"(min {min(f1s):.4f}, max {max(f1s):.4f}).")
        lines.append(f"- Monotone non-decreasing in data: "
                     f"{'yes' if monotone else 'no'}.")
        lines.append(
            f"- Slope of the top half (frac 0.5 → 1.0): {hi - f1s[-2]:+.4f}. "
            "A clear positive slope still climbing at frac=1.0 argues the frozen "
            "ceiling is a sample-size artifact (more data would help). A flat "
            "curve at ~0 argues the E1 fidelity cliff is the binding constraint "
            "at this data scale — architecture (E5) and data (E6) are both "
            "exhausted, and the bottleneck is EEG signal fidelity itself.\n")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="..",
                    help="Dir containing the checkpoints_e6_<TAG>/ folders.")
    ap.add_argument("--out_dir", default="../exp_logs",
                    help="Where to write e6_data_scaling.{json,md}.")
    args = ap.parse_args()

    rows = [read_point(args.ckpt_root, tag, frac) for tag, frac in POINTS]
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "e6_data_scaling.json")
    md_path = os.path.join(args.out_dir, "e6_data_scaling.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(md_path, "w") as f:
        f.write(to_markdown(rows))

    done = sum(1 for r in rows if r["status"] == "DONE")
    print(f"Wrote {json_path} and {md_path}  ({done}/{len(POINTS)} points DONE)")
    for r in rows:
        f1 = fmt(r.get("test_f1"))
        print(f"  frac {r['frac']:.2f} [{r['tag']}] {r['status']:8s} test_F1={f1}")


if __name__ == "__main__":
    main()
