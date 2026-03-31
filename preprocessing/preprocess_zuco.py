"""
EEG-to-Graph: ZuCo 2.0 Data Preprocessing Pipeline
====================================================

This script extracts word-level EEG frequency-domain features from the ZuCo 2.0 dataset,
preparing them for the EEG-to-Graph translation model.

Decisions baked in:
- Tasks: Both Task 1 (Normal Reading) + Task 2 (Task-Specific Reading)
- EEG features: Frequency-domain (8 bands × 105 channels = 840 dims per word)
- Fixation window: GD (Gaze Duration)
- Subjects: All 18 pooled
- Missing words: Zero-padded
- Normalization: Per-subject z-score

Requirements:
    pip install h5py numpy scipy scikit-learn tqdm

Usage:
    python preprocess_zuco.py --data_dir /path/to/zuco2 --output_dir /path/to/output

Expected directory structure:
    zuco2/
    ├── task1-NR/
    │   ├── resultsYAC_NR.mat
    │   ├── resultsYAG_NR.mat
    │   └── ...
    ├── task2-TSR/
    │   ├── resultsYAC_TSR.mat
    │   ├── resultsYAG_TSR.mat
    │   └── ...
    └── task_materials/
        └── ... (sentence texts and relation labels)
"""

import os
import argparse
import json
import h5py
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# The 8 EEG frequency bands in ZuCo (appended to fixation window prefix)
FREQ_BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]

# Fixation window to use (GD = Gaze Duration)
FIXATION_WINDOW = "GD"

# Feature field names: e.g., "GD_t1", "GD_t2", ..., "GD_g2"
FEATURE_FIELDS = [f"{FIXATION_WINDOW}_{band}" for band in FREQ_BANDS]

# Number of EEG channels (after removing EOG and face/neck channels)
N_CHANNELS = 105

# Total feature dimensionality per word: 8 bands × 105 channels = 840
FEATURE_DIM = len(FREQ_BANDS) * N_CHANNELS  # 840

# Subject IDs from ZuCo 2.0
SUBJECT_IDS = [
    "YAC", "YAG", "YAK", "YDG", "YDR", "YFR", "YFS", "YHS",
    "YIS", "YLS", "YMD", "YMS", "YRH", "YRK", "YRP", "YSD",
    "YSL", "YTL"
]

# Task configuration
TASKS = {
    "task1-NR": {"suffix": "NR", "description": "Normal Reading"},
    "task2-TSR": {"suffix": "TSR", "description": "Task-Specific Reading"},
}

# Train/val/test split ratios (split by sentence, not by sample)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


# =============================================================================
# HDF5 / MATLAB v7.3 Helper Functions
# =============================================================================

def h5_to_string(dataset, h5file):
    """
    Extract a string from an HDF5 reference or dataset.

    MATLAB v7.3 stores strings as uint16 arrays. This function handles
    both direct datasets and object references.
    """
    try:
        if isinstance(dataset, h5py.Dataset):
            data = dataset[()]
        elif isinstance(dataset, np.ndarray):
            data = dataset
        else:
            # It might be an object reference
            data = h5file[dataset][()]

        # MATLAB stores strings as uint16 arrays
        if data.dtype == np.uint16 or data.dtype == '>u2':
            return ''.join(chr(c) for c in data.flatten())
        elif data.dtype == np.object_:
            # Array of references — dereference the first one
            return h5_to_string(data.flat[0], h5file)
        else:
            return str(data)
    except Exception as e:
        return None


def h5_to_array(dataset, h5file):
    """
    Extract a numeric array from an HDF5 dataset or reference.
    Returns None if the data is empty or invalid.
    """
    try:
        if isinstance(dataset, h5py.Dataset):
            data = dataset[()]
        elif isinstance(dataset, h5py.Reference):
            data = h5file[dataset][()]
        elif isinstance(dataset, np.ndarray) and dataset.dtype == np.object_:
            data = h5file[dataset.flat[0]][()]
        else:
            data = np.array(dataset)

        # Check for empty or all-NaN
        if data.size == 0:
            return None
        if np.all(np.isnan(data)):
            return None

        return data.astype(np.float32)
    except Exception:
        return None


# =============================================================================
# HDF5 Shape Helpers
# =============================================================================

def get_ref_count(dataset):
    """
    Get the number of elements in an HDF5 dataset of references,
    handling MATLAB's transposed storage.

    MATLAB v7.3 can store a (1, N) cell array as either shape (1, N)
    or (N, 1) in HDF5 depending on the version. We take the max
    dimension to get the actual element count.
    """
    shape = dataset.shape
    if len(shape) == 1:
        return shape[0]
    else:
        return max(shape)


def get_ref_at(dataset, index):
    """
    Get a single reference from an HDF5 dataset at the given index,
    handling both (1, N) and (N, 1) layouts.
    """
    shape = dataset.shape
    if len(shape) == 1:
        return dataset[index]
    elif shape[0] == 1:
        # Shape is (1, N) — words are along axis 1
        return dataset[0, index]
    elif shape[1] == 1:
        # Shape is (N, 1) — words are along axis 0
        return dataset[index, 0]
    else:
        # Unexpected shape — try (0, index) as default
        return dataset[0, index]


# =============================================================================
# Core Extraction Functions
# =============================================================================

def extract_sentence_data(h5file, sentence_ref=None, sentence_text=None, word_group=None):
    """
    Extract word-level EEG features and sentence text from a single sentence.

    Supports two calling conventions:
      1. extract_sentence_data(h5file, sentence_ref)
         — when sentenceData is a Dataset of refs, each ref points to a group
           with 'content' and 'word' children.
      2. extract_sentence_data(h5file, sentence_text=text, word_group=grp)
         — when sentenceData is a struct-array Group, text and word data
           are accessed separately by the caller.

    Returns:
        dict with keys:
            - 'text': The sentence string
            - 'words': List of word strings
            - 'eeg_features': np.array of shape (num_words, 840)
            - 'has_fixation': Boolean mask of shape (num_words,)
        Or None if invalid/incomplete.
    """

    # --- Resolve sentence text and word group ---
    if sentence_text is not None and word_group is not None:
        # Called with explicit text + word_group (struct-array layout)
        pass
    elif sentence_ref is not None:
        # Called with a reference to a sentence group
        try:
            if isinstance(sentence_ref, h5py.Group):
                sent_group = sentence_ref
            else:
                sent_group = h5file[sentence_ref]
        except Exception:
            return None

        if "content" in sent_group:
            sentence_text = h5_to_string(sent_group["content"], h5file)

        if "word" in sent_group:
            word_ref = sent_group["word"]
            if isinstance(word_ref, h5py.Group):
                word_group = word_ref
            else:
                # It's a dataset/ref — dereference it
                try:
                    word_group = h5file[word_ref[()]]
                except Exception:
                    word_group = None
    else:
        return None

    if sentence_text is None or word_group is None:
        return None

    # --- Determine number of words ---
    if "content" not in word_group:
        return None

    word_content_refs = word_group["content"]
    n_words = get_ref_count(word_content_refs)

    # --- Extract word strings ---
    words = []
    for i in range(n_words):
        try:
            ref = get_ref_at(word_content_refs, i)
            word_str = h5_to_string(ref, h5file)
            words.append(word_str if word_str else "<UNK>")
        except Exception:
            words.append("<UNK>")

    # --- Extract EEG frequency features for each word ---
    eeg_features = np.zeros((n_words, FEATURE_DIM), dtype=np.float32)
    has_fixation = np.zeros(n_words, dtype=bool)

    for band_idx, field_name in enumerate(FEATURE_FIELDS):
        if field_name not in word_group:
            continue

        band_data_refs = word_group[field_name]

        for word_idx in range(n_words):
            try:
                ref = get_ref_at(band_data_refs, word_idx)
                band_values = h5_to_array(ref, h5file)

                if band_values is not None and band_values.size >= N_CHANNELS:
                    channel_values = band_values.flatten()[:N_CHANNELS]

                    if len(channel_values) == N_CHANNELS and not np.all(np.isnan(channel_values)):
                        start_idx = band_idx * N_CHANNELS
                        end_idx = start_idx + N_CHANNELS
                        channel_values = np.nan_to_num(channel_values, nan=0.0)
                        eeg_features[word_idx, start_idx:end_idx] = channel_values
                        has_fixation[word_idx] = True

            except Exception:
                continue

    if not np.any(has_fixation):
        return None

    return {
        "text": sentence_text,
        "words": words,
        "eeg_features": eeg_features,
        "has_fixation": has_fixation,
    }


def process_subject_file(filepath, subject_id, task_name):
    """
    Process a single subject's .mat file and extract all sentence data.

    Args:
        filepath: Path to the .mat file
        subject_id: e.g. "YAC"
        task_name: e.g. "task1-NR"

    Returns:
        List of dicts, one per valid sentence, each containing:
            - subject_id, task, text, words, eeg_features, has_fixation
    """
    samples = []

    try:
        with h5py.File(filepath, "r") as h5file:
            # The main data structure in ZuCo .mat files
            if "sentenceData" not in h5file:
                print(f"  WARNING: 'sentenceData' not found in {filepath}")
                return samples

            sentence_data = h5file["sentenceData"]

            if isinstance(sentence_data, h5py.Dataset):
                # Array of references
                n_sentences = get_ref_count(sentence_data)

                for i in range(n_sentences):
                    ref = get_ref_at(sentence_data, i)
                    result = extract_sentence_data(h5file, ref)

                    if result is not None:
                        result["subject_id"] = subject_id
                        result["task"] = task_name
                        samples.append(result)

            elif isinstance(sentence_data, h5py.Group):
                # Struct-array layout: sentenceData is a Group where each key
                # is a field name (content, word, mean_a1, ...) and each field
                # is a Dataset with one entry per sentence.
                if "content" not in sentence_data or "word" not in sentence_data:
                    print(f"  WARNING: sentenceData Group missing 'content' or 'word' in {filepath}")
                    return samples

                content_refs = sentence_data["content"]
                word_refs = sentence_data["word"]
                n_sentences = get_ref_count(content_refs)

                for i in range(n_sentences):
                    try:
                        # Get sentence text
                        text_ref = get_ref_at(content_refs, i)
                        sentence_text = h5_to_string(text_ref, h5file)

                        # Get word-level data group
                        word_ref = get_ref_at(word_refs, i)
                        word_group = h5file[word_ref]

                        result = extract_sentence_data(
                            h5file,
                            sentence_text=sentence_text,
                            word_group=word_group,
                        )

                        if result is not None:
                            result["subject_id"] = subject_id
                            result["task"] = task_name
                            samples.append(result)
                    except Exception:
                        continue

    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")

    return samples


# =============================================================================
# Normalization
# =============================================================================

def normalize_per_subject(all_samples):
    """
    Apply per-subject z-score normalization.

    For each subject, compute the mean and std across ALL their word-level
    EEG features (from all sentences, both tasks), then normalize.
    Only uses fixated words for computing statistics (ignores zero-padded).

    Args:
        all_samples: List of sample dicts (modified in-place)

    Returns:
        norm_stats: Dict mapping subject_id -> {"mean": array, "std": array}
    """
    print("\nComputing per-subject normalization statistics...")

    # Step 1: Gather all fixated features per subject
    subject_features = defaultdict(list)
    for sample in all_samples:
        subj = sample["subject_id"]
        eeg = sample["eeg_features"]
        mask = sample["has_fixation"]
        if np.any(mask):
            subject_features[subj].append(eeg[mask])

    # Step 2: Compute mean and std per subject
    norm_stats = {}
    for subj, feat_list in subject_features.items():
        all_feats = np.concatenate(feat_list, axis=0)  # (total_fixated_words, 840)
        mean = np.mean(all_feats, axis=0)
        std = np.std(all_feats, axis=0)
        # Avoid division by zero: if std is 0 for a feature, set to 1
        std[std < 1e-8] = 1.0
        norm_stats[subj] = {"mean": mean, "std": std}
        print(f"  {subj}: {all_feats.shape[0]} fixated words, "
              f"mean range [{mean.min():.4f}, {mean.max():.4f}]")

    # Step 3: Apply normalization
    print("Applying normalization...")
    for sample in all_samples:
        subj = sample["subject_id"]
        stats = norm_stats[subj]
        mask = sample["has_fixation"]

        # Only normalize fixated words; zero-padded words stay at zero
        if np.any(mask):
            sample["eeg_features"][mask] = (
                (sample["eeg_features"][mask] - stats["mean"]) / stats["std"]
            )

    return norm_stats


# =============================================================================
# Train / Val / Test Split
# =============================================================================

def split_by_sentence(all_samples, seed=42):
    """
    Split samples into train/val/test sets, grouping by sentence text.

    All readings of the same sentence (different subjects, different tasks)
    go into the same split. This prevents data leakage where the model
    memorizes sentence identity rather than learning from EEG.

    Args:
        all_samples: List of sample dicts

    Returns:
        dict with keys "train", "val", "test", each a list of sample dicts
    """
    # Assign a group ID to each unique sentence text
    unique_sentences = list(set(s["text"] for s in all_samples))
    sentence_to_group = {text: i for i, text in enumerate(unique_sentences)}

    groups = np.array([sentence_to_group[s["text"]] for s in all_samples])
    indices = np.arange(len(all_samples))

    # First split: separate test set
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=SPLIT_RATIOS["test"], random_state=seed)
    train_val_idx, test_idx = next(splitter1.split(indices, groups=groups))

    # Second split: separate val from train
    val_ratio_adjusted = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
    train_val_groups = groups[train_val_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    train_idx_local, val_idx_local = next(splitter2.split(train_val_idx, groups=train_val_groups))

    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]

    splits = {
        "train": [all_samples[i] for i in train_idx],
        "val": [all_samples[i] for i in val_idx],
        "test": [all_samples[i] for i in test_idx],
    }

    # Verify no sentence leakage
    train_sentences = set(s["text"] for s in splits["train"])
    val_sentences = set(s["text"] for s in splits["val"])
    test_sentences = set(s["text"] for s in splits["test"])

    assert len(train_sentences & val_sentences) == 0, "Leakage between train and val!"
    assert len(train_sentences & test_sentences) == 0, "Leakage between train and test!"
    assert len(val_sentences & test_sentences) == 0, "Leakage between val and test!"

    return splits


# =============================================================================
# Save to Disk
# =============================================================================

def save_splits(splits, norm_stats, output_dir):
    """
    Save processed data to disk in a format ready for the training pipeline.

    For each split, saves:
        - {split}_eeg.npy: List of EEG arrays (variable-length sequences)
        - {split}_meta.json: Metadata (text, words, subject, task, fixation mask)

    Also saves:
        - norm_stats.json: Per-subject normalization statistics
        - dataset_info.json: Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_info = {
        "feature_dim": FEATURE_DIM,
        "fixation_window": FIXATION_WINDOW,
        "freq_bands": FREQ_BANDS,
        "n_channels": N_CHANNELS,
        "tasks": list(TASKS.keys()),
        "subjects": SUBJECT_IDS,
        "splits": {},
    }

    for split_name, samples in splits.items():
        print(f"\nSaving {split_name} split: {len(samples)} samples")

        # Save EEG features as a list of variable-length arrays
        eeg_list = [s["eeg_features"] for s in samples]
        np.save(
            os.path.join(output_dir, f"{split_name}_eeg.npy"),
            np.array(eeg_list, dtype=object),
            allow_pickle=True,
        )

        # Save metadata
        meta = []
        for s in samples:
            meta.append({
                "text": s["text"],
                "words": s["words"],
                "subject_id": s["subject_id"],
                "task": s["task"],
                "has_fixation": s["has_fixation"].tolist(),
                "n_words": len(s["words"]),
            })

        with open(os.path.join(output_dir, f"{split_name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Stats
        n_words_list = [len(s["words"]) for s in samples]
        fixation_rates = [s["has_fixation"].mean() for s in samples]
        unique_sentences = len(set(s["text"] for s in samples))

        dataset_info["splits"][split_name] = {
            "n_samples": len(samples),
            "n_unique_sentences": unique_sentences,
            "avg_sentence_length": float(np.mean(n_words_list)),
            "avg_fixation_rate": float(np.mean(fixation_rates)),
        }

    # Save normalization stats (convert numpy arrays to lists for JSON)
    norm_stats_serializable = {
        subj: {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()}
        for subj, stats in norm_stats.items()
    }
    with open(os.path.join(output_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats_serializable, f)

    # Save dataset info
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDataset info saved to {os.path.join(output_dir, 'dataset_info.json')}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main(data_dir, output_dir):
    """Run the full preprocessing pipeline."""

    print("=" * 60)
    print("EEG-to-Graph: ZuCo 2.0 Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Fixation window:  {FIXATION_WINDOW}")
    print(f"  Feature fields:   {FEATURE_FIELDS}")
    print(f"  Feature dim:      {FEATURE_DIM}")
    print(f"  Tasks:            {list(TASKS.keys())}")
    print(f"  Subjects:         {len(SUBJECT_IDS)}")
    print(f"  Data directory:   {data_dir}")
    print(f"  Output directory: {output_dir}")

    # ---- Step 1: Extract data from all subjects and tasks ----
    print("\n" + "=" * 60)
    print("Step 1: Extracting EEG features from .mat files")
    print("=" * 60)

    all_samples = []

    for task_dir, task_info in TASKS.items():
        task_path = os.path.join(data_dir, task_dir)
        suffix = task_info["suffix"]

        print(f"\n--- {task_info['description']} ({task_dir}) ---")

        if not os.path.exists(task_path):
            print(f"  WARNING: Directory not found: {task_path}")
            print(f"  Trying alternative path structure...")
            # Try alternative naming conventions
            alt_paths = [
                os.path.join(data_dir, task_dir.replace("-", "_")),
                os.path.join(data_dir, suffix.lower()),
                os.path.join(data_dir, suffix),
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    task_path = alt
                    print(f"  Found at: {task_path}")
                    break
            else:
                print(f"  SKIPPING: No directory found for {task_dir}")
                continue

        for subject_id in tqdm(SUBJECT_IDS, desc=f"  Subjects ({suffix})"):
            # Try common filename patterns
            possible_filenames = [
                f"results{subject_id}_{suffix}.mat",
                f"results{subject_id.lower()}_{suffix}.mat",
                f"results{subject_id}_{suffix.lower()}.mat",
                f"{subject_id}_{suffix}.mat",
            ]

            filepath = None
            for fname in possible_filenames:
                candidate = os.path.join(task_path, fname)
                if os.path.exists(candidate):
                    filepath = candidate
                    break

            if filepath is None:
                # Try to find any .mat file containing the subject ID
                if os.path.exists(task_path):
                    for f in os.listdir(task_path):
                        if subject_id in f and f.endswith(".mat"):
                            filepath = os.path.join(task_path, f)
                            break

            if filepath is None:
                continue

            samples = process_subject_file(filepath, subject_id, task_dir)
            all_samples.extend(samples)

    print(f"\n  Total samples extracted: {len(all_samples)}")

    if len(all_samples) == 0:
        print("\nERROR: No data was extracted. Please check your data directory structure.")
        print("Expected structure:")
        print("  <data_dir>/task1-NR/resultsYAC_NR.mat")
        print("  <data_dir>/task2-TSR/resultsYAC_TSR.mat")
        return

    # Print summary
    unique_sentences = len(set(s["text"] for s in all_samples))
    unique_subjects = len(set(s["subject_id"] for s in all_samples))
    task_counts = defaultdict(int)
    for s in all_samples:
        task_counts[s["task"]] += 1

    print(f"  Unique sentences: {unique_sentences}")
    print(f"  Unique subjects:  {unique_subjects}")
    for task, count in task_counts.items():
        print(f"  Samples from {task}: {count}")

    # ---- Step 2: Per-subject normalization ----
    print("\n" + "=" * 60)
    print("Step 2: Per-subject z-score normalization")
    print("=" * 60)

    norm_stats = normalize_per_subject(all_samples)

    # ---- Step 3: Train/Val/Test split ----
    print("\n" + "=" * 60)
    print("Step 3: Splitting data (grouped by sentence)")
    print("=" * 60)

    splits = split_by_sentence(all_samples)

    for split_name, samples in splits.items():
        n_unique = len(set(s["text"] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_unique} unique sentences)")

    # ---- Step 4: Save to disk ----
    print("\n" + "=" * 60)
    print("Step 4: Saving processed data")
    print("=" * 60)

    save_splits(splits, norm_stats, output_dir)

    print("\n" + "=" * 60)
    print("DONE! Preprocessing complete.")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")
    print(f"  - train_eeg.npy, train_meta.json")
    print(f"  - val_eeg.npy, val_meta.json")
    print(f"  - test_eeg.npy, test_meta.json")
    print(f"  - norm_stats.json")
    print(f"  - dataset_info.json")
    print(f"\nNext step: Generate ground-truth labels (triplets) from sentence texts")
    print(f"  using REBEL or GPT-4, then pair with the EEG data for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-to-Graph: ZuCo 2.0 Preprocessing")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the ZuCo 2.0 data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_zuco",
        help="Path to save processed data (default: ./processed_zuco)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)