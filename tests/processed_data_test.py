"""
EEG-to-Graph: Processed Data Validator
=======================================

Reads the preprocessed ZuCo data (npy + json) and prints summary statistics
and sample details to verify the preprocessing was done correctly.

Usage:
    python validate_processed.py --data_dir ./processed_zuco
    python validate_processed.py --data_dir ./processed_zuco --n_samples 5
"""

import os
import argparse
import json
import numpy as np


def load_split(data_dir, split_name):
    """Load EEG data and metadata for a given split."""
    eeg_path = os.path.join(data_dir, f"{split_name}_eeg.npy")
    meta_path = os.path.join(data_dir, f"{split_name}_meta.json")

    if not os.path.exists(eeg_path):
        print(f"  WARNING: {eeg_path} not found")
        return None, None
    if not os.path.exists(meta_path):
        print(f"  WARNING: {meta_path} not found")
        return None, None

    eeg_data = np.load(eeg_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return eeg_data, meta


def print_sample(eeg, meta_entry, index):
    """Print detailed info about a single sample."""
    print(f"\n  --- Sample {index} ---")
    print(f"  Subject:    {meta_entry['subject_id']}")
    print(f"  Task:       {meta_entry['task']}")
    print(f"  Text:       {meta_entry['text'][:100]}{'...' if len(meta_entry['text']) > 100 else ''}")
    print(f"  Words:      {meta_entry['words'][:10]}{'...' if len(meta_entry['words']) > 10 else ''}")
    print(f"  n_words:    {meta_entry['n_words']}")
    print(f"  EEG shape:  {eeg.shape}")

    fixation_mask = meta_entry["has_fixation"]
    n_fixated = sum(fixation_mask)
    n_total = len(fixation_mask)
    print(f"  Fixated:    {n_fixated}/{n_total} words ({100 * n_fixated / n_total:.1f}%)")

    # Check consistency between meta and eeg
    issues = []
    if eeg.shape[0] != meta_entry["n_words"]:
        issues.append(f"EEG rows ({eeg.shape[0]}) != n_words ({meta_entry['n_words']})")
    if eeg.shape[1] != 840:
        issues.append(f"EEG cols ({eeg.shape[1]}) != 840")
    if len(meta_entry["words"]) != meta_entry["n_words"]:
        issues.append(f"len(words) ({len(meta_entry['words'])}) != n_words ({meta_entry['n_words']})")
    if len(fixation_mask) != meta_entry["n_words"]:
        issues.append(f"len(has_fixation) ({len(fixation_mask)}) != n_words ({meta_entry['n_words']})")

    # Check that unfixated words are actually zero
    for word_idx in range(n_total):
        if not fixation_mask[word_idx]:
            if not np.allclose(eeg[word_idx], 0.0):
                issues.append(f"Word {word_idx} is unfixated but EEG is not zero")
                break

    # Check that fixated words are NOT all zero
    for word_idx in range(n_total):
        if fixation_mask[word_idx]:
            if np.allclose(eeg[word_idx], 0.0):
                issues.append(f"Word {word_idx} is fixated but EEG is all zeros")
                break

    if issues:
        print(f"  ISSUES:     {issues}")
    else:
        print(f"  Checks:     ALL PASSED")

    # Show EEG stats for fixated words
    fixated_eeg = eeg[fixation_mask]
    if len(fixated_eeg) > 0:
        print(f"  EEG stats (fixated words):")
        print(f"    mean: {fixated_eeg.mean():.4f}")
        print(f"    std:  {fixated_eeg.std():.4f}")
        print(f"    min:  {fixated_eeg.min():.4f}")
        print(f"    max:  {fixated_eeg.max():.4f}")
        print(f"    any NaN: {np.any(np.isnan(fixated_eeg))}")


def validate_split(data_dir, split_name, n_samples=3):
    """Validate and print info for a split."""
    print(f"\n{'=' * 60}")
    print(f"Split: {split_name.upper()}")
    print(f"{'=' * 60}")

    eeg_data, meta = load_split(data_dir, split_name)
    if eeg_data is None or meta is None:
        return

    print(f"  Total samples:      {len(meta)}")
    print(f"  EEG array length:   {len(eeg_data)}")

    if len(meta) != len(eeg_data):
        print(f"  ERROR: meta ({len(meta)}) and eeg ({len(eeg_data)}) have different lengths!")
        return

    # Aggregate stats
    unique_sentences = len(set(m["text"] for m in meta))
    unique_subjects = len(set(m["subject_id"] for m in meta))
    tasks = set(m["task"] for m in meta)
    word_counts = [m["n_words"] for m in meta]
    fixation_rates = [sum(m["has_fixation"]) / len(m["has_fixation"]) for m in meta]

    print(f"  Unique sentences:   {unique_sentences}")
    print(f"  Unique subjects:    {unique_subjects}")
    print(f"  Tasks:              {tasks}")
    print(f"  Sentence lengths:   min={min(word_counts)}, max={max(word_counts)}, avg={np.mean(word_counts):.1f}")
    print(f"  Fixation rates:     min={min(fixation_rates):.2f}, max={max(fixation_rates):.2f}, avg={np.mean(fixation_rates):.2f}")

    # Print individual samples
    print(f"\n  Showing {n_samples} sample(s):")
    for i in range(min(n_samples, len(meta))):
        print_sample(eeg_data[i], meta[i], i)


def main():
    parser = argparse.ArgumentParser(description="Validate processed ZuCo data")
    parser.add_argument("--data_dir", type=str, default="./processed_zuco", help="Path to processed data directory")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples to print per split")
    args = parser.parse_args()

    print("=" * 60)
    print("EEG-to-Graph: Processed Data Validation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")

    # Load and print dataset info
    info_path = os.path.join(args.data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        print(f"\nDataset Info:")
        print(f"  Feature dim:       {info.get('feature_dim')}")
        print(f"  Fixation window:   {info.get('fixation_window')}")
        print(f"  Frequency bands:   {info.get('freq_bands')}")
        print(f"  Channels:          {info.get('n_channels')}")
        print(f"  Tasks:             {info.get('tasks')}")
        print(f"  Subjects:          {len(info.get('subjects', []))}")
        if "splits" in info:
            print(f"  Split summary:")
            for split, stats in info["splits"].items():
                print(f"    {split}: {stats}")
    else:
        print(f"\n  WARNING: {info_path} not found")

    # Validate each split
    for split in ["train", "val", "test"]:
        validate_split(args.data_dir, split, args.n_samples)

    # Cross-split leakage check
    print(f"\n{'=' * 60}")
    print("Cross-Split Leakage Check")
    print(f"{'=' * 60}")

    split_sentences = {}
    for split in ["train", "val", "test"]:
        _, meta = load_split(args.data_dir, split)
        if meta:
            split_sentences[split] = set(m["text"] for m in meta)

    if len(split_sentences) == 3:
        train_val = split_sentences["train"] & split_sentences["val"]
        train_test = split_sentences["train"] & split_sentences["test"]
        val_test = split_sentences["val"] & split_sentences["test"]
        print(f"  Train-Val overlap:  {len(train_val)} sentences {'PASS' if len(train_val) == 0 else 'FAIL!'}")
        print(f"  Train-Test overlap: {len(train_test)} sentences {'PASS' if len(train_test) == 0 else 'FAIL!'}")
        print(f"  Val-Test overlap:   {len(val_test)} sentences {'PASS' if len(val_test) == 0 else 'FAIL!'}")

        if train_val:
            print(f"  Leaked train-val examples: {list(train_val)[:3]}")
        if train_test:
            print(f"  Leaked train-test examples: {list(train_test)[:3]}")
        if val_test:
            print(f"  Leaked val-test examples: {list(val_test)[:3]}")
    else:
        print("  Could not load all splits for leakage check.")

    print(f"\n{'=' * 60}")
    print("Validation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()