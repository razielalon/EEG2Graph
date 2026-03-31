"""
ZuCo 2.0 Data Inspector
========================

Use this script to explore the structure of the ZuCo .mat files before
running the full preprocessing pipeline. Helps verify that the HDF5
paths and field names match what the preprocessor expects.

Usage:
    python inspect_zuco.py /path/to/resultsYAC_NR.mat
    python inspect_zuco.py /path/to/resultsYAC_NR.mat --sentence 0 --detailed
"""

import sys
import argparse
import h5py
import numpy as np


def print_h5_structure(group, prefix="", max_depth=4, current_depth=0):
    """Recursively print the structure of an HDF5 file/group."""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    for key in sorted(group.keys()):
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix} {key}/ ({len(item)} items)")
            print_h5_structure(item, prefix + "    ", max_depth, current_depth + 1)
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            print(f"{prefix} {key}: shape={shape}, dtype={dtype}")
        else:
            print(f"{prefix} {key}: {type(item)}")


def h5_to_string(ref_or_dataset, h5file):
    """Extract string from HDF5 reference/dataset."""
    try:
        if isinstance(ref_or_dataset, h5py.Dataset):
            data = ref_or_dataset[()]
        else:
            data = h5file[ref_or_dataset][()]

        if data.dtype == np.uint16 or data.dtype == '>u2':
            return ''.join(chr(c) for c in data.flatten())
        elif data.dtype == np.object_:
            return h5_to_string(data.flat[0], h5file)
        return str(data)
    except Exception as e:
        return f"<error: {e}>"


def get_sentence_count(sentence_data, h5file):
    """
    Determine the number of sentences regardless of whether sentenceData
    is a Dataset (array of struct refs) or a Group (struct of arrays).
    """
    if isinstance(sentence_data, h5py.Dataset):
        return sentence_data.shape[-1] if len(sentence_data.shape) > 1 else sentence_data.shape[0]
    elif isinstance(sentence_data, h5py.Group):
        # Use 'content' field to determine count (it should always exist)
        if "content" in sentence_data:
            ds = sentence_data["content"]
            return ds.shape[0]
        # Fallback: use first available dataset
        for key in sentence_data.keys():
            item = sentence_data[key]
            if isinstance(item, h5py.Dataset):
                return item.shape[0]
    return 0


def get_sentence_ref(sentence_data, idx):
    """
    Get a reference to sentence idx from a Dataset-type sentenceData.
    """
    if len(sentence_data.shape) > 1:
        return sentence_data[0, idx]
    else:
        return sentence_data[idx]


def inspect_sentence_from_dataset(h5file, sentence_ref, sentence_idx, detailed=False):
    """Inspect a single sentence when sentenceData is a Dataset of references."""
    try:
        sent = h5file[sentence_ref]
    except Exception as e:
        print(f"  ERROR accessing sentence: {e}")
        return

    print(f"\n{'='*50}")
    print(f"Sentence {sentence_idx}")
    print(f"{'='*50}")

    # Print sentence text
    if "content" in sent:
        text = h5_to_string(sent["content"], h5file)
        print(f"  Text: {text}")

    # List all fields
    print(f"\n  Fields in sentence:")
    for key in sorted(sent.keys()):
        item = sent[key]
        if isinstance(item, h5py.Dataset):
            print(f"    {key}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"    {key}/ (group with {len(item)} items)")

    # Inspect word-level data
    if "word" in sent:
        _inspect_word_group(h5file, sent["word"], detailed)


def inspect_sentence_from_group(h5file, sentence_data, sentence_idx, detailed=False):
    """
    Inspect a single sentence when sentenceData is a Group (struct of arrays).

    In this layout:
      sentenceData/content  -> (N, 1) object refs, one per sentence
      sentenceData/word     -> (N, 1) object refs, each pointing to a word group
      sentenceData/mean_t1  -> (N, 1) object refs for sentence-level mean features
      ...
    """
    print(f"\n{'='*50}")
    print(f"Sentence {sentence_idx}")
    print(f"{'='*50}")

    # Print sentence text
    if "content" in sentence_data:
        try:
            ref = sentence_data["content"][sentence_idx, 0]
            text = h5_to_string(ref, h5file)
            print(f"  Text: {text}")
        except Exception as e:
            print(f"  Text: <error: {e}>")

    # List all sentenceData fields with this sentence's data
    print(f"\n  sentenceData fields (showing sentence {sentence_idx}):")
    for key in sorted(sentence_data.keys()):
        item = sentence_data[key]
        if isinstance(item, h5py.Dataset):
            try:
                ref = item[sentence_idx, 0]
                deref = h5file[ref]
                if isinstance(deref, h5py.Group):
                    print(f"    {key}: -> Group with {len(deref)} items")
                elif isinstance(deref, h5py.Dataset):
                    data = deref[()]
                    print(f"    {key}: -> shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"    {key}: -> {type(deref)}")
            except Exception as e:
                print(f"    {key}: <error: {e}>")

    # Inspect word-level data
    if "word" in sentence_data:
        try:
            word_ref = sentence_data["word"][sentence_idx, 0]
            word_group = h5file[word_ref]
            if isinstance(word_group, h5py.Group):
                _inspect_word_group(h5file, word_group, detailed)
            elif isinstance(word_group, h5py.Dataset):
                # word_ref might point to a dataset, dereference further
                data = word_group[()]
                print(f"\n  Word data (direct dataset): shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            print(f"\n  Word-level data: <error: {e}>")


def _inspect_word_group(h5file, word_group, detailed=False):
    """Inspect word-level data (shared between both sentenceData layouts)."""
    print(f"\n  Word-level fields ({len(word_group)} fields):")
    for key in sorted(word_group.keys()):
        item = word_group[key]
        if isinstance(item, h5py.Dataset):
            print(f"    {key}: shape={item.shape}, dtype={item.dtype}")

    # Try to extract word texts
    if "content" in word_group:
        word_refs = word_group["content"]
        n_words = word_refs.shape[0] if len(word_refs.shape) == 1 else word_refs.shape[-1]
        print(f"\n  Number of words: {n_words}")

        words = []
        for i in range(min(n_words, 10)):  # Show first 10 words
            try:
                ref = word_refs[i] if len(word_refs.shape) == 1 else word_refs[0, i]
                w = h5_to_string(ref, h5file)
                words.append(w)
            except Exception:
                words.append("<error>")
        print(f"  First words: {words}")
        if n_words > 10:
            print(f"  ... and {n_words - 10} more")

    # Check GD features specifically
    if detailed:
        gd_fields = [k for k in word_group.keys() if k.startswith("GD_")]
        print(f"\n  GD (Gaze Duration) fields found: {gd_fields}")

        for field in gd_fields[:2]:  # Show first 2 bands as example
            print(f"\n  Inspecting {field}:")
            field_data = word_group[field]
            print(f"    Shape: {field_data.shape}, Dtype: {field_data.dtype}")

            # Try to read first word's data
            try:
                if field_data.dtype == np.object_ or 'ref' in str(field_data.dtype):
                    ref = field_data[0] if len(field_data.shape) == 1 else field_data[0, 0]
                    word_data = h5file[ref][()]
                    print(f"    Word 0 data shape: {word_data.shape}")
                    print(f"    Word 0 data dtype: {word_data.dtype}")
                    if word_data.size > 0:
                        flat = word_data.flatten()
                        print(f"    Word 0 values (first 5): {flat[:5]}")
                        print(f"    Word 0 NaN count: {np.isnan(flat).sum()} / {len(flat)}")
                else:
                    print(f"    Direct data (first row): {field_data[0][:5]}")
            except Exception as e:
                print(f"    Error reading: {e}")


def check_field_availability(h5file, sentence_data, n_sent):
    """
    Check which GD/FFD/TRT fields exist across a few sentences.
    Works with both Dataset and Group sentenceData layouts.
    """
    print(f"\n{'='*50}")
    print("Quick field availability check (first 3 sentences):")
    print(f"{'='*50}")

    is_group = isinstance(sentence_data, h5py.Group)

    for i in range(min(3, n_sent)):
        try:
            if is_group:
                # Group layout: dereference sentenceData/word -> word group
                if "word" not in sentence_data:
                    print(f"  Sentence {i}: no 'word' field in sentenceData")
                    continue
                word_ref = sentence_data["word"][i, 0]
                word_obj = h5file[word_ref]
                if isinstance(word_obj, h5py.Group):
                    word_fields = list(word_obj.keys())
                else:
                    print(f"  Sentence {i}: word ref points to {type(word_obj)}, not a Group")
                    continue
            else:
                # Dataset layout: dereference sentence -> sentence group -> word group
                ref = get_sentence_ref(sentence_data, i)
                sent = h5file[ref]
                if "word" not in sent:
                    print(f"  Sentence {i}: no 'word' field")
                    continue
                word_fields = list(sent["word"].keys())

            gd_fields = [f for f in word_fields if f.startswith("GD_")]
            ffd_fields = [f for f in word_fields if f.startswith("FFD_")]
            trt_fields = [f for f in word_fields if f.startswith("TRT_")]
            print(f"  Sentence {i}: GD={gd_fields}, FFD={ffd_fields}, TRT={trt_fields}")
        except Exception as e:
            print(f"  Sentence {i}: error - {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect ZuCo .mat file structure")
    parser.add_argument("filepath", help="Path to a ZuCo .mat file")
    parser.add_argument("--sentence", type=int, default=0, help="Sentence index to inspect (default: 0)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed EEG feature inspection")
    parser.add_argument("--structure_only", action="store_true", help="Only show top-level structure")
    args = parser.parse_args()

    print(f"Opening: {args.filepath}\n")

    with h5py.File(args.filepath, "r") as h5file:
        # Show top-level structure
        print("Top-level structure:")
        print_h5_structure(h5file, max_depth=2)

        if args.structure_only:
            return

        # Check sentenceData
        if "sentenceData" not in h5file:
            print("\nWARNING: 'sentenceData' not found!")
            print("Available top-level keys:", list(h5file.keys()))
            return

        sentence_data = h5file["sentenceData"]
        print(f"\nsentenceData type: {type(sentence_data).__name__}")

        # Determine sentence count (works for both layouts)
        n_sent = get_sentence_count(sentence_data, h5file)
        print(f"Number of sentences: {n_sent}")

        if isinstance(sentence_data, h5py.Dataset):
            print(f"sentenceData layout: Dataset (array of struct references)")
            print(f"  shape: {sentence_data.shape}, dtype: {sentence_data.dtype}")

            if args.sentence < n_sent:
                ref = get_sentence_ref(sentence_data, args.sentence)
                inspect_sentence_from_dataset(h5file, ref, args.sentence, args.detailed)
            else:
                print(f"Sentence index {args.sentence} out of range (max: {n_sent - 1})")

        elif isinstance(sentence_data, h5py.Group):
            print(f"sentenceData layout: Group (struct of arrays)")
            print(f"  Fields: {list(sentence_data.keys())}")

            if args.sentence < n_sent:
                inspect_sentence_from_group(h5file, sentence_data, args.sentence, args.detailed)
            else:
                print(f"Sentence index {args.sentence} out of range (max: {n_sent - 1})")

        # Field availability check (works for both layouts)
        if n_sent > 0:
            check_field_availability(h5file, sentence_data, n_sent)


if __name__ == "__main__":
    main()