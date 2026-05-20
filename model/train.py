"""
EEG-to-Graph: Training Pipeline
=================================

Trains the Bridge + BART model to generate linearized triplets from EEG.
Includes:
  - Teacher-forced training with label-smoothed cross-entropy
  - Validation with greedy decoding + triplet-level F1
  - Checkpointing, LR scheduling, gradient clipping
  - Differential LRs: high for Bridge, low for BART

Usage:
    python train.py \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --output_dir ../checkpoints \
        --epochs 80 \
        --batch_size 16
"""

import os
import json
import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from vocabulary import delinearize, save_tokenizer, STRUCT_TOKENS
from eeg_graph_dataset import build_dataloaders
from eeg_graph_model import EEGBartModel


# =============================================================================
# Loss
# =============================================================================

class LabelSmoothedCE(nn.Module):
    """Cross-entropy with label smoothing, ignoring PAD tokens."""

    def __init__(self, vocab_size, pad_id, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_id = pad_id

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, T, V)
            targets: (B, T) with pad_id for ignored positions
        """
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)

        non_pad = targets != self.pad_id
        logits = logits[non_pad]
        targets = targets[non_pad]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets, reduction="mean")
        smooth = -log_probs.mean(dim=-1).mean()

        return (1 - self.smoothing) * nll + self.smoothing * smooth


# =============================================================================
# Evaluation Metrics
# =============================================================================

def triplet_set_to_tuples(triplets):
    return set(
        (t["subject"].lower().strip(), t["relation"].lower().strip(), t["object"].lower().strip())
        for t in triplets
    )


def compute_triplet_f1(pred_triplets_batch, gold_triplets_batch):
    total_pred = 0
    total_gold = 0
    total_correct = 0

    for preds, golds in zip(pred_triplets_batch, gold_triplets_batch):
        pred_set = triplet_set_to_tuples(preds)
        gold_set = triplet_set_to_tuples(golds)

        total_pred += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set & gold_set)

    precision = total_correct / max(total_pred, 1)
    recall = total_correct / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": total_pred,
        "n_gold": total_gold,
        "n_correct": total_correct,
    }


def _log_sample_predictions(preds, golds, n=3):
    """Print sample predictions vs gold for debugging."""
    for i in range(min(n, len(preds))):
        gold = [(t["subject"], t["relation"], t["object"]) for t in golds[i][:3]]
        pred = [(t["subject"], t["relation"], t["object"]) for t in preds[i][:3]]
        n_gold = len(golds[i])
        n_pred = len(preds[i])
        print(f"      [{i}] Gold ({n_gold}): {gold}")
        print(f"          Pred ({n_pred}): {pred}")


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)

        logits = model(src, src_mask, tgt)
        loss = criterion(logits, tgt_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, tokenizer, device, max_gen_len=128, num_beams=1):
    """Evaluate with both loss and triplet F1."""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_pred_triplets = []
    all_gold_triplets = []

    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)
        metas = batch["meta"]

        logits = model(src, src_mask, tgt)
        loss = criterion(logits, tgt_labels)
        total_loss += loss.item()
        n_batches += 1

        gen = model.generate(src, src_mask, max_len=max_gen_len, num_beams=num_beams)
        for i in range(gen.size(0)):
            tokens = gen[i].cpu().tolist()
            pred = delinearize(tokens, tokenizer)
            all_pred_triplets.append(pred)
            all_gold_triplets.append(metas[i]["triplets"])

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_triplet_f1(all_pred_triplets, all_gold_triplets)
    metrics["loss"] = avg_loss

    return metrics, all_pred_triplets, all_gold_triplets


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    print("\n" + "=" * 60)
    print("Loading data")
    print("=" * 60)

    limits = {
        "train": args.limit_train,
        "val": args.limit_val,
        "test": args.limit_test,
    }
    loaders, tokenizer = build_dataloaders(
        args.processed_dir, args.triplets_path,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        bart_name=args.bart_name,
        limits=limits,
    )

    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    save_tokenizer(tokenizer, tokenizer_dir)
    print(f"\nTokenizer vocab size: {len(tokenizer)}")

    # ---- Model ----
    print("\n" + "=" * 60)
    print("Building model")
    print("=" * 60)

    model = EEGBartModel(
        tokenizer=tokenizer,
        eeg_dim=args.eeg_dim,
        bart_name=args.bart_name,
        dropout=args.dropout,
        bridge_layers=args.bridge_layers,
        bart_dropout=args.bart_dropout,
        bart_attention_dropout=args.bart_attention_dropout,
    ).to(device)

    if args.freeze_bart:
        model.freeze_bart()
        print("  BART/REBEL frozen — training bridge only.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bridge = sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,} (bridge: {n_bridge:,}, bart: {n_params - n_bridge:,})")

    # ---- Optimizer ----
    # Phased training: freeze BART during warmup so the bridge learns a useful
    # EEG→embedding mapping before BART's representations get disrupted.
    warmup_active = args.warmup_epochs > 0 and not args.freeze_bart
    if warmup_active:
        model.freeze_bart()
        print(f"\n  Phase 1: BART frozen for {args.warmup_epochs} warmup epochs (bridge-only)")

    optimizer = AdamW(
        model.param_groups(args.bridge_lr, args.bart_lr, args.weight_decay),
        betas=(0.9, 0.98),
    )
    sched_T = args.warmup_epochs if warmup_active else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(sched_T, 1), eta_min=args.bart_lr * 0.01)

    criterion = LabelSmoothedCE(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        smoothing=args.label_smoothing,
    )

    # ---- Training ----
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    # Checkpoint selection key: (val_f1, -val_loss). F1 dominates; when F1 is
    # uninformative (e.g. stuck at 0.0 on a small/hard run) the tie breaks on
    # the lowest val_loss, so best_model.pt tracks the genuinely-best epoch
    # instead of freezing at epoch 1. Starts below any real score.
    best_ckpt_score = (-1.0, float("-inf"))
    patience_counter = 0
    history = []

    phase2_start_epoch = None  # set at the unfreeze transition
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Phase transition: unfreeze BART after warmup ---
        if warmup_active and epoch == args.warmup_epochs + 1:
            model.unfreeze_bart()
            optimizer = AdamW(
                model.param_groups_llrd(
                    bridge_lr=args.bridge_lr * 0.3,
                    bart_lr=args.bart_lr,
                    bridge_wd=args.weight_decay,
                    bart_wd=args.bart_weight_decay,
                    llrd_gamma=args.llrd_gamma,
                ),
                betas=(0.9, 0.98),
            )
            remaining = args.epochs - args.warmup_epochs
            W = max(args.bart_warmup_epochs, 0)

            def _bridge_lambda(e, _r=remaining):
                # pure cosine over remaining epochs (no second warmup for bridge)
                t = min(e, max(_r - 1, 0))
                return 0.5 * (1 + math.cos(math.pi * t / max(_r - 1, 1)))

            def _bart_lambda(e, _W=W, _r=remaining):
                if e < _W:
                    return (e + 1) / max(_W, 1)  # linear warmup: 1/W .. 1
                progress = (e - _W) / max(_r - _W - 1, 1)
                progress = min(progress, 1.0)
                return 0.5 * (1 + math.cos(math.pi * progress))

            lambdas = [
                _bart_lambda if g.get("name", "").startswith("bart_") else _bridge_lambda
                for g in optimizer.param_groups
            ]
            scheduler = LambdaLR(optimizer, lr_lambda=lambdas)

            patience_counter = 0  # keep best_val_loss continuous, just reset counter
            phase2_start_epoch = epoch
            n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_groups = len(optimizer.param_groups)
            print(f"\n  Phase 2: BART unfrozen ({n_total:,} trainable params, {n_groups} LLRD groups, "
                  f"{remaining} epochs, BART warmup={W} epochs, llrd_gamma={args.llrd_gamma})")
            warmup_active = False

        train_loss = train_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, device,
            grad_clip=args.grad_clip,
        )

        val_metrics = {}
        val_preds = val_golds = None
        if "val" in loaders:
            val_metrics, val_preds, val_golds = evaluate(
                model, loaders["val"], criterion, tokenizer, device,
                max_gen_len=args.max_tgt_len,
                num_beams=1,  # greedy during training for speed
            )

        train_metrics = {}
        if args.eval_train and "train" in loaders:
            train_metrics, _, _ = evaluate(
                model, loaders["train"], criterion, tokenizer, device,
                max_gen_len=args.max_tgt_len,
                num_beams=1,
            )

        elapsed = time.time() - t0
        bridge_lr_log = 0.0
        bart_lrs = []
        for pg in optimizer.param_groups:
            if pg.get("name") == "bridge":
                bridge_lr_log = pg["lr"]
            else:
                bart_lrs.append(pg["lr"])
        bart_lr_max = max(bart_lrs) if bart_lrs else 0.0
        bart_lr_min = min(bart_lrs) if bart_lrs else 0.0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_metrics.get("f1", 0),
            "train_precision": train_metrics.get("precision", 0),
            "train_recall": train_metrics.get("recall", 0),
            "val_loss": val_metrics.get("loss", 0),
            "val_f1": val_metrics.get("f1", 0),
            "val_precision": val_metrics.get("precision", 0),
            "val_recall": val_metrics.get("recall", 0),
            "lr_bridge": bridge_lr_log,
            "lr_bart_max": bart_lr_max,
            "lr_bart_min": bart_lr_min,
            "time": elapsed,
        }
        history.append(log)

        train_f1_str = ""
        if args.eval_train:
            train_f1_str = (f"train_F1={train_metrics.get('f1', 0):.4f} "
                            f"(P={train_metrics.get('precision', 0):.3f} "
                            f"R={train_metrics.get('recall', 0):.3f}) | ")
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"{train_f1_str}"
              f"val_loss={val_metrics.get('loss', 0):.4f} | "
              f"val_F1={val_metrics.get('f1', 0):.4f} "
              f"(P={val_metrics.get('precision', 0):.3f} R={val_metrics.get('recall', 0):.3f}) | "
              f"lr=[bridge={bridge_lr_log:.2e}, bart={bart_lr_min:.2e}..{bart_lr_max:.2e}] | "
              f"{elapsed:.1f}s")

        # Diagnostic: show sample predictions periodically
        if epoch % 10 == 0 and val_preds and val_golds:
            print("    Sample predictions:")
            _log_sample_predictions(val_preds, val_golds, n=3)

        # Early stopping on val loss. Suppress the patience counter during the
        # BART linear-warmup window — val loss is expected to fluctuate while
        # BART thaws and would otherwise trigger early stop spuriously.
        in_bart_warmup_window = (
            phase2_start_epoch is not None
            and epoch < phase2_start_epoch + max(args.bart_warmup_epochs, 0)
        )
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        elif not in_bart_warmup_window:
            patience_counter += 1

        # Checkpoint best — see best_ckpt_score above.
        ckpt_score = (val_metrics.get("f1", 0.0), -val_loss)
        if ckpt_score > best_ckpt_score:
            best_ckpt_score = ckpt_score
            best_val_f1 = val_metrics.get("f1", 0.0)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "args": vars(args),
                "bart_name": args.bart_name,
                "struct_tokens": STRUCT_TOKENS,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"    New best checkpoint: val_F1={best_val_f1:.4f}, "
                  f"val_loss={val_loss:.4f}")

            if val_preds and val_golds:
                examples = []
                for i in range(min(5, len(val_preds))):
                    examples.append({
                        "gold": val_golds[i],
                        "pred": val_preds[i],
                    })
                with open(os.path.join(args.output_dir, "best_examples.json"), "w") as f:
                    json.dump(examples, f, indent=2)

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "bart_name": args.bart_name,
            }, os.path.join(args.output_dir, f"checkpoint_ep{epoch}.pt"))

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no val loss improvement for {args.patience} epochs)")
            break

    # ---- Final Test Evaluation ----
    if "test" in loaders:
        print("\n" + "=" * 60)
        print("Test Evaluation (beam search)")
        print("=" * 60)

        ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics, test_preds, test_golds = evaluate(
            model, loaders["test"], criterion, tokenizer, device,
            max_gen_len=args.max_tgt_len,
            num_beams=args.beam_size,
        )

        print(f"  Test F1:        {test_metrics['f1']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test Recall:    {test_metrics['recall']:.4f}")
        print(f"  Predicted:      {test_metrics['n_pred']} triplets")
        print(f"  Gold:           {test_metrics['n_gold']} triplets")
        print(f"  Correct:        {test_metrics['n_correct']} triplets")

        test_results = []
        for i in range(len(test_preds)):
            test_results.append({
                "gold": test_golds[i],
                "pred": test_preds[i],
            })
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone!")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-to-Graph Training (Bridge + BART)")

    # Paths
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Directory with preprocessed EEG data")
    parser.add_argument("--triplets_path", type=str, required=True,
                        help="Path to ground-truth triplets JSON")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Where to save checkpoints and results")

    # Model
    parser.add_argument("--bart_name", type=str, default="Babelscape/rebel-large",
                        help="HuggingFace model name (default: REBEL, a BART-large "
                             "checkpoint already fine-tuned for relation extraction)")
    parser.add_argument("--eeg_dim", type=int, default=840)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bridge_lr", type=float, default=3e-4,
                        help="Learning rate for the Bridge projection")
    parser.add_argument("--bart_lr", type=float, default=3e-5,
                        help="Peak learning rate for top BART layers (LLRD-decayed for "
                             "lower layers, see --llrd_gamma)")
    parser.add_argument("--freeze_bart", action="store_true",
                        help="Freeze BART/REBEL entirely and train only the bridge. "
                             "Recommended for CPU runs and small-data regimes.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Freeze BART for the first N epochs (bridge-only warmup). "
                             "Set to 0 to disable. Ignored if --freeze_bart is set.")
    parser.add_argument("--bart_warmup_epochs", type=int, default=3,
                        help="After BART unfreezes, linearly ramp its LR from 0 to the "
                             "target over N epochs to avoid the post-unfreeze overfitting "
                             "cliff. 0 disables the ramp (cold-start at full LR).")
    parser.add_argument("--llrd_gamma", type=float, default=0.8,
                        help="Layer-wise LR decay factor for BART. Top encoder/decoder "
                             "layers get bart_lr; layer k below the top gets "
                             "bart_lr * llrd_gamma**k. Embeddings get the smallest LR. "
                             "1.0 = no decay.")
    parser.add_argument("--bart_dropout", type=float, default=0.2,
                        help="Override REBEL's native hidden/activation dropout (default "
                             "0.1) — higher values regularize fine-tuning on small data.")
    parser.add_argument("--bart_attention_dropout", type=float, default=0.2,
                        help="Override REBEL's native attention dropout.")
    parser.add_argument("--bart_weight_decay", type=float, default=0.05,
                        help="Weight decay applied to BART param groups (separate from "
                             "--weight_decay, which only applies to the bridge).")
    parser.add_argument("--bridge_layers", type=int, default=2,
                        help="Number of bridge projection layers (1=linear, 2+=MLP with GELU)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience on val loss (0=disabled). "
                             "Suppressed during the BART warmup window.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for the bridge group only; BART uses "
                             "--bart_weight_decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_train", action="store_true",
                        help="Each epoch, also run generation-based triplet F1 on the "
                             "training set (logged as train_F1). Diagnostic for "
                             "tiny-overfit tests — slow on the full train set.")

    # Data
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_train", type=int, default=0,
                        help="If >0, use only the first N training samples (CPU sanity runs)")
    parser.add_argument("--limit_val", type=int, default=0,
                        help="If >0, use only the first N val samples")
    parser.add_argument("--limit_test", type=int, default=0,
                        help="If >0, use only the first N test samples")

    # Inference
    parser.add_argument("--beam_size", type=int, default=4,
                        help="num_beams for final test evaluation (1=greedy)")

    args = parser.parse_args()
    main(args)
