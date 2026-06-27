#!/usr/bin/env python
"""
E5 — frozen-BART component ablation grid (RQ4): aggregate the 8 cells.

Scans the per-cell output dirs written by train.py (one per grid cell,
`checkpoints_e5_<TAG>/`) and builds the deliverable table. Each cell contributes:

  test_metrics.json  -> test f1 / precision / recall (+ n_pred/n_gold/n_correct)
  history.json       -> best val_f1 (and its epoch), and the align loss at that
                        epoch + the final-epoch align loss

TAG grammar (set by exp_e5_cell.sbatch): s{0,1} t{0,1} a{PL,PW}
  s1 = subject embedding ON (--n_subject_buckets 64), s0 = OFF
  t1 = temporal bridge ON  (--bridge_transformer_layers 2), t0 = OFF
  aPL = --align_mode pooled, aPW = --align_mode perword

Writes exp_logs/e5_ablation_grid.{json,md}. Cells whose dir is missing or whose
run hasn't reached test eval yet are listed as PENDING (so this is safe to run
while jobs are still in the queue).

Usage:
    cd model && python aggregate_e5.py
    python aggregate_e5.py --ckpt_root .. --out_dir ../exp_logs
"""
import os
import re
import json
import argparse

# Canonical cell order (matches the plan's 8-row table and the sbatch submit list).
CELLS = [
    ("s0t0aPL", 1), ("s1t0aPL", 2), ("s0t1aPL", 3), ("s1t1aPL", 4),
    ("s0t0aPW", 5), ("s1t0aPW", 6), ("s0t1aPW", 7), ("s1t1aPW", 8),
]
TAG_RE = re.compile(r"^s([01])t([01])a(PL|PW)$")


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"bad TAG {tag!r}")
    return {
        "subject": "on" if m.group(1) == "1" else "off",
        "temporal": "on" if m.group(2) == "1" else "off",
        "align": "perword" if m.group(3) == "PW" else "pooled",
    }


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_cell(ckpt_root, tag):
    d = os.path.join(ckpt_root, f"checkpoints_e5_{tag}")
    row = {"tag": tag, **parse_tag(tag), "status": "PENDING", "dir": d}
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
        row["status"] = "TRAINED"  # finished training; may still lack test eval

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
    lines.append("# E5 — Frozen-BART component ablation grid (RQ4): results\n")
    lines.append("Under frozen BART, which architectural component moves the "
                 "needle? Each cell varies only the three axes; all other "
                 "hyperparameters are fixed (seed 42, 60 epochs, patience 25, "
                 "`--align_weight 1.0`, `--align_temp 0.07`).\n")
    lines.append(f"**Completed: {len(done)}/8 cells.**\n")

    lines.append("| # | subject | temporal | align | test F1 | test P | test R "
                 "| best val F1 (ep) | align@best | align final |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        n = next(num for tag, num in CELLS if tag == r["tag"])
        ep = r.get("best_val_epoch")
        valcell = (f"{fmt(r.get('best_val_f1'))} ({ep})"
                   if "best_val_f1" in r else "PENDING")
        lines.append(
            f"| {n} | {r['subject']} | {r['temporal']} | {r['align']} "
            f"| {fmt(r.get('test_f1'))} | {fmt(r.get('test_precision'))} "
            f"| {fmt(r.get('test_recall'))} | {valcell} "
            f"| {fmt(r.get('align_at_best'))} | {fmt(r.get('align_final'))} |"
        )

    if pending:
        tags = ", ".join(f"cell {next(n for t, n in CELLS if t==r['tag'])} "
                         f"({r['status']})" for r in pending)
        lines.append(f"\n_Pending: {tags}._")

    if len(done) == 8:
        best = max(done, key=lambda r: r["test_f1"])
        worst_align = max(done, key=lambda r: r.get("align_final", 0))
        lines.append("\n## Read-off\n")
        lines.append(
            f"- Best cell by test F1: **#{next(n for t,n in CELLS if t==best['tag'])}** "
            f"(subject={best['subject']}, temporal={best['temporal']}, "
            f"align={best['align']}) → test F1 {best['test_f1']:.4f}.")
        spread = best["test_f1"] - min(r["test_f1"] for r in done)
        lines.append(f"- Test-F1 spread across the grid: {spread:.4f}.")
        lines.append(
            "- If the spread is ~0 the frozen-bridge ceiling is "
            "architecture-independent at this data scale (hands the story to E6 "
            "data-scaling, reinforces E1's fidelity cliff). If one axis lifts F1, "
            "that names what helps under frozen BART.\n")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="..",
                    help="Dir containing the checkpoints_e5_<TAG>/ folders.")
    ap.add_argument("--out_dir", default="../exp_logs",
                    help="Where to write e5_ablation_grid.{json,md}.")
    args = ap.parse_args()

    rows = [read_cell(args.ckpt_root, tag) for tag, _ in CELLS]
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "e5_ablation_grid.json")
    md_path = os.path.join(args.out_dir, "e5_ablation_grid.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(md_path, "w") as f:
        f.write(to_markdown(rows))

    done = sum(1 for r in rows if r["status"] == "DONE")
    print(f"Wrote {json_path} and {md_path}  ({done}/8 cells DONE)")
    for r in rows:
        n = next(num for tag, num in CELLS if tag == r["tag"])
        f1 = fmt(r.get("test_f1"))
        print(f"  cell {n} [{r['tag']}] {r['status']:8s} test_F1={f1}")


if __name__ == "__main__":
    main()
