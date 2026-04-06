"""
EEG-to-Graph: Ground-Truth Triplet Label Generation
=====================================================

Extracts (subject, relation, object) triplets from sentence texts using
REBEL (Babelscape/rebel-large), then creates training-ready label files
aligned with the preprocessed EEG data.

Pipeline:
    1. Collect unique sentences from processed metadata
    2. Run REBEL on each sentence to extract triplets
    3. Linearize triplets into decoder target strings
    4. Join back to per-sample metadata and save label files
    5. Generate a quality report for manual inspection

Requirements:
    pip install transformers torch tqdm

Usage:
    python generate_triplets.py --processed_dir ./processed_zuco --output_dir ./processed_zuco

Output:
    - sentence_triplets.json    : sentence text -> extracted triplets (663 entries)
    - train_labels.json         : per-sample linearized target strings (aligned with train_eeg.npy)
    - val_labels.json           : same for val
    - test_labels.json          : same for test
    - triplet_quality_report.txt: human-readable report for manual review
"""

import os
import json
import argparse
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

REBEL_MODEL = "Babelscape/rebel-large"

# Special tokens for linearized output format
# Format: <triplet> subject <subj> relation <rel> object <obj>
# Multiple triplets are concatenated, each starting with <triplet>
SPECIAL_TOKENS = {
    "triplet_start": "<triplet>",
    "subject": "<subj>",
    "relation": "<rel>",
    "object": "<obj>",
    "none": "<none>",  # For sentences with no extractable triplets
}

# REBEL generation parameters
GEN_KWARGS = {
    "max_length": 256,
    "num_beams": 5,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
}

SPLITS = ["train", "val", "test"]


# =============================================================================
# REBEL Triplet Extraction
# =============================================================================

def load_rebel_model(device="cpu"):
    """Load the REBEL model and tokenizer."""
    print(f"Loading REBEL model: {REBEL_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, tokenizer


def extract_triplets_from_rebel_output(text):
    """
    Parse REBEL's generated text into structured triplets.

    REBEL outputs text in the format:
        <triplet> Subject Name <subj> Relation Type <rel> Object Name <obj> ...

    Returns:
        List of dicts: [{"subject": str, "relation": str, "object": str}, ...]
    """
    triplets = []
    current = None

    # Split on special tokens and parse
    text = text.strip()

    for token in text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").split():
        if token == "<triplet>":
            if current and all(current.values()):
                triplets.append(current)
            current = {"subject": "", "relation": "", "object": ""}
        elif token == "<subj>":
            if current is not None:
                current["_filling"] = "relation"
        elif token == "<rel>":
            if current is not None:
                current["_filling"] = "object"
        elif token == "<obj>":
            if current and all(v for k, v in current.items() if k != "_filling"):
                triplets.append(current)
            current = {"subject": "", "relation": "", "object": ""}
        else:
            if current is not None:
                filling = current.get("_filling", "subject")
                if current[filling]:
                    current[filling] += " " + token
                else:
                    current[filling] = token

    # Don't forget the last one
    if current and all(v for k, v in current.items() if k != "_filling"):
        triplets.append(current)

    # Clean up internal state key
    for t in triplets:
        t.pop("_filling", None)

    # Deduplicate
    seen = set()
    unique_triplets = []
    for t in triplets:
        key = (t["subject"].strip(), t["relation"].strip(), t["object"].strip())
        if key not in seen and all(key):
            seen.add(key)
            unique_triplets.append({
                "subject": t["subject"].strip(),
                "relation": t["relation"].strip(),
                "object": t["object"].strip(),
            })

    return unique_triplets


def run_rebel_on_sentences(sentences, model, tokenizer, device="cpu", batch_size=8):
    """
    Run REBEL on a list of sentences and return extracted triplets.

    Args:
        sentences: List of sentence strings
        model: REBEL model
        tokenizer: REBEL tokenizer
        device: torch device
        batch_size: Batch size for inference

    Returns:
        Dict mapping sentence text -> list of triplet dicts
    """
    results = {}

    for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting triplets"):
        batch = sentences[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            **inputs,
            **GEN_KWARGS,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        for sentence, raw_output in zip(batch, decoded):
            triplets = extract_triplets_from_rebel_output(raw_output)
            results[sentence] = {
                "triplets": triplets,
                "raw_rebel_output": raw_output,
            }

    return results


# =============================================================================
# Linearization
# =============================================================================

def linearize_triplets(triplets):
    """
    Convert a list of triplet dicts into a linearized target string.

    Format: <triplet> subject <subj> relation <rel> object <obj> <triplet> ...

    If no triplets, returns "<none>".
    """
    if not triplets:
        return SPECIAL_TOKENS["none"]

    parts = []
    for t in triplets:
        parts.append(
            f"{SPECIAL_TOKENS['triplet_start']} "
            f"{t['subject']} {SPECIAL_TOKENS['subject']} "
            f"{t['relation']} {SPECIAL_TOKENS['relation']} "
            f"{t['object']} {SPECIAL_TOKENS['object']}"
        )

    return " ".join(parts)


# =============================================================================
# Quality Report
# =============================================================================

def generate_quality_report(sentence_triplets, output_path):
    """
    Generate a human-readable report for manual inspection.

    Includes:
        - Overall statistics
        - Sentences with 0 triplets (need review)
        - Sentences with many triplets (might have noise)
        - Random sample for spot-checking
        - Relation frequency distribution
    """
    total = len(sentence_triplets)
    triplet_counts = [len(v["triplets"]) for v in sentence_triplets.values()]
    zero_triplet_sentences = [
        s for s, v in sentence_triplets.items() if len(v["triplets"]) == 0
    ]

    all_relations = []
    for v in sentence_triplets.values():
        for t in v["triplets"]:
            all_relations.append(t["relation"])
    relation_freq = Counter(all_relations)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EEG-to-Graph: Triplet Extraction Quality Report\n")
        f.write("=" * 70 + "\n\n")

        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total unique sentences:     {total}\n")
        f.write(f"  Total triplets extracted:   {sum(triplet_counts)}\n")
        f.write(f"  Avg triplets per sentence:  {sum(triplet_counts)/total:.2f}\n")
        f.write(f"  Sentences with 0 triplets:  {len(zero_triplet_sentences)} "
                f"({100*len(zero_triplet_sentences)/total:.1f}%)\n")
        f.write(f"  Max triplets in a sentence: {max(triplet_counts)}\n")

        # Distribution
        f.write(f"\n  Triplet count distribution:\n")
        count_dist = Counter(triplet_counts)
        for k in sorted(count_dist.keys()):
            bar = "█" * count_dist[k]
            f.write(f"    {k:2d} triplets: {count_dist[k]:4d} sentences  {bar}\n")

        # Relation frequency (top 30)
        f.write(f"\nTOP RELATIONS ({len(relation_freq)} unique)\n")
        f.write("-" * 40 + "\n")
        for rel, count in relation_freq.most_common(30):
            f.write(f"  {count:4d}  {rel}\n")

        # Zero-triplet sentences
        f.write(f"\nSENTENCES WITH 0 TRIPLETS (review these)\n")
        f.write("-" * 40 + "\n")
        for s in zero_triplet_sentences[:50]:  # Show first 50
            f.write(f"  - {s}\n")
        if len(zero_triplet_sentences) > 50:
            f.write(f"  ... and {len(zero_triplet_sentences) - 50} more\n")

        # Sample of extractions for spot-checking
        f.write(f"\nSAMPLE EXTRACTIONS (first 30 sentences with triplets)\n")
        f.write("-" * 40 + "\n")
        shown = 0
        for sentence, data in sentence_triplets.items():
            if data["triplets"] and shown < 30:
                f.write(f"\n  Sentence: {sentence}\n")
                for t in data["triplets"]:
                    f.write(f"    ({t['subject']}) --[{t['relation']}]--> ({t['object']})\n")
                shown += 1

    print(f"Quality report saved to {output_path}")


# =============================================================================
# Join with Per-Sample Metadata
# =============================================================================

def create_label_files(sentence_triplets, processed_dir, output_dir):
    """
    Create per-sample label files aligned with the EEG data.

    For each split, reads the metadata, looks up the sentence's triplets,
    and creates a label file with the linearized target string.
    """
    stats = {}

    for split in SPLITS:
        meta_path = os.path.join(processed_dir, f"{split}_meta.json")
        if not os.path.exists(meta_path):
            print(f"  WARNING: {meta_path} not found, skipping")
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        labels = []
        missing = 0

        for sample in meta:
            sentence_text = sample["text"]

            if sentence_text in sentence_triplets:
                triplets = sentence_triplets[sentence_text]["triplets"]
                linearized = linearize_triplets(triplets)
            else:
                # Sentence not found — shouldn't happen, but handle gracefully
                linearized = SPECIAL_TOKENS["none"]
                missing += 1

            labels.append({
                "linearized": linearized,
                "triplets": sentence_triplets.get(sentence_text, {}).get("triplets", []),
                "n_triplets": len(sentence_triplets.get(sentence_text, {}).get("triplets", [])),
            })

        # Save
        label_path = os.path.join(output_dir, f"{split}_labels.json")
        with open(label_path, "w") as f:
            json.dump(labels, f, indent=2)

        n_with_triplets = sum(1 for l in labels if l["n_triplets"] > 0)
        stats[split] = {
            "n_samples": len(labels),
            "n_with_triplets": n_with_triplets,
            "n_empty": len(labels) - n_with_triplets,
            "missing_sentences": missing,
        }

        print(f"  {split}: {len(labels)} labels saved "
              f"({n_with_triplets} with triplets, {len(labels) - n_with_triplets} empty)")

    return stats


# =============================================================================
# Main
# =============================================================================

def main(processed_dir, output_dir, device="cpu", batch_size=8):
    print("=" * 60)
    print("EEG-to-Graph: Triplet Label Generation (REBEL)")
    print("=" * 60)

    # ---- Step 1: Collect unique sentences ----
    print("\nStep 1: Collecting unique sentences from metadata...")

    all_sentences = set()
    for split in SPLITS:
        meta_path = os.path.join(processed_dir, f"{split}_meta.json")
        if not os.path.exists(meta_path):
            print(f"  WARNING: {meta_path} not found")
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for sample in meta:
            all_sentences.add(sample["text"])

    sentences = sorted(all_sentences)  # Sort for reproducibility
    print(f"  Found {len(sentences)} unique sentences")

    # ---- Step 2: Run REBEL ----
    print(f"\nStep 2: Running REBEL on {len(sentences)} sentences...")

    model, tokenizer = load_rebel_model(device=device)
    sentence_triplets = run_rebel_on_sentences(
        sentences, model, tokenizer, device=device, batch_size=batch_size
    )

    # Save raw extraction results
    # (strip raw_rebel_output for the clean version, keep for debugging)
    clean_triplets = {
        s: {"triplets": v["triplets"]} for s, v in sentence_triplets.items()
    }
    triplet_path = os.path.join(output_dir, "sentence_triplets.json")
    with open(triplet_path, "w") as f:
        json.dump(clean_triplets, f, indent=2)
    print(f"  Saved sentence-level triplets to {triplet_path}")

    # Also save the version with raw REBEL output for debugging
    debug_path = os.path.join(output_dir, "sentence_triplets_debug.json")
    with open(debug_path, "w") as f:
        json.dump(sentence_triplets, f, indent=2)

    # ---- Step 3: Quality report ----
    print(f"\nStep 3: Generating quality report...")
    report_path = os.path.join(output_dir, "triplet_quality_report.txt")
    generate_quality_report(sentence_triplets, report_path)

    # ---- Step 4: Create per-sample label files ----
    print(f"\nStep 4: Creating per-sample label files...")
    os.makedirs(output_dir, exist_ok=True)
    label_stats = create_label_files(sentence_triplets, processed_dir, output_dir)

    # ---- Summary ----
    total_triplets = sum(len(v["triplets"]) for v in sentence_triplets.values())
    n_empty = sum(1 for v in sentence_triplets.values() if len(v["triplets"]) == 0)

    print(f"\n{'='*60}")
    print("DONE! Triplet generation complete.")
    print(f"{'='*60}")
    print(f"\n  Sentences processed:  {len(sentences)}")
    print(f"  Total triplets:       {total_triplets}")
    print(f"  Avg per sentence:     {total_triplets/len(sentences):.2f}")
    print(f"  Empty sentences:      {n_empty} ({100*n_empty/len(sentences):.1f}%)")
    print(f"\n  Output files in: {output_dir}")
    print(f"    - sentence_triplets.json       (sentence -> triplets mapping)")
    print(f"    - sentence_triplets_debug.json  (with raw REBEL output)")
    print(f"    - triplet_quality_report.txt    (review this!)")
    print(f"    - train_labels.json, val_labels.json, test_labels.json")
    print(f"\n  IMPORTANT: Review triplet_quality_report.txt with Ziv before training!")
    print(f"  Look for: zero-triplet sentences, noisy relations, missing entities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG-to-Graph: Generate triplet labels using REBEL"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./processed_zuco",
        help="Path to preprocessed ZuCo data (default: ./processed_zuco)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_zuco",
        help="Path to save label files (default: same as processed_dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for REBEL inference: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for REBEL inference (default: 8)",
    )
    args = parser.parse_args()
    main(args.processed_dir, args.output_dir, args.device, args.batch_size)
