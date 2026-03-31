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


def get_ref_count(dataset):
    """Get element count, handling MATLAB's transposed storage."""
    shape = dataset.shape
    if len(shape) == 1:
        return shape[0]
    else:
        return max(shape)


def get_ref_at(dataset, index):
    """Get a reference at index, handling both (1,N) and (N,1) layouts."""
    shape = dataset.shape
    if len(shape) == 1:
        return dataset[index]
    elif shape[0] == 1:
        return dataset[0, index]
    elif shape[1] == 1:
        return dataset[index, 0]
    else:
        return dataset[0, index]


def inspect_sentence_from_parts(h5file, text, word_group, sentence_idx, detailed=False):
    """Inspect a sentence when text and word_group are provided separately (struct-array layout)."""
    print(f"\n{'='*50}")
    print(f"Sentence {sentence_idx}")
    print(f"{'='*50}")

    print(f"  Text: {text}")

    print(f"\n  Word-level fields ({len(word_group)} fields):")
    for key in sorted(word_group.keys()):
        item = word_group[key]
        if isinstance(item, h5py.Dataset):
            print(f"    {key}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"    {key}/ (group)")

    if "content" in word_group:
        word_refs = word_group["content"]
        n_words = get_ref_count(word_refs)
        print(f"\n  Number of words: {n_words}")
        print(f"  (content shape: {word_refs.shape})")

        words = []
        for i in range(min(n_words, 10)):
            try:
                ref = get_ref_at(word_refs, i)
                w = h5_to_string(ref, h5file)
                words.append(w)
            except Exception:
                words.append("<e>")
        print(f"  First words: {words}")
        if n_words > 10:
            print(f"  ... and {n_words - 10} more")

    if detailed:
        gd_fields = [k for k in word_group.keys() if k.startswith("GD_")]
        print(f"\n  GD (Gaze Duration) fields found: {gd_fields}")

        for field in gd_fields[:2]:
            print(f"\n  Inspecting {field}:")
            field_data = word_group[field]
            print(f"    Shape: {field_data.shape}, Dtype: {field_data.dtype}")
            print(f"    Element count: {get_ref_count(field_data)}")

            try:
                ref = get_ref_at(field_data, 0)
                word_data = h5file[ref][()]
                print(f"    Word 0 data shape: {word_data.shape}")
                print(f"    Word 0 data dtype: {word_data.dtype}")
                if word_data.size > 0:
                    flat = word_data.flatten()
                    print(f"    Word 0 values (first 5): {flat[:5]}")
                    print(f"    Word 0 NaN count: {np.isnan(flat).sum()} / {len(flat)}")
            except Exception as e:
                print(f"    Error reading: {e}")


def inspect_sentence(h5file, sentence_ref_or_group, sentence_idx, detailed=False):
    """Inspect a single sentence's data structure."""
    try:
        if isinstance(sentence_ref_or_group, h5py.Group):
            sent = sentence_ref_or_group
        else:
            sent = h5file[sentence_ref_or_group]
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
        word_group = sent["word"]
        print(f"\n  Word-level fields ({len(word_group)} fields):")
        for key in sorted(word_group.keys()):
            item = word_group[key]
            if isinstance(item, h5py.Dataset):
                print(f"    {key}: shape={item.shape}, dtype={item.dtype}")

        # Try to extract word texts
        if "content" in word_group:
            word_refs = word_group["content"]
            n_words = get_ref_count(word_refs)
            print(f"\n  Number of words: {n_words}")
            print(f"  (content shape: {word_refs.shape})")

            words = []
            for i in range(min(n_words, 10)):  # Show first 10 words
                try:
                    ref = get_ref_at(word_refs, i)
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
                print(f"    Element count: {get_ref_count(field_data)}")

                # Try to read first word's data
                try:
                    ref = get_ref_at(field_data, 0)
                    word_data = h5file[ref][()]
                    print(f"    Word 0 data shape: {word_data.shape}")
                    print(f"    Word 0 data dtype: {word_data.dtype}")
                    if word_data.size > 0:
                        flat = word_data.flatten()
                        print(f"    Word 0 values (first 5): {flat[:5]}")
                        print(f"    Word 0 NaN count: {np.isnan(flat).sum()} / {len(flat)}")
                except Exception as e:
                    print(f"    Error reading: {e}")


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
        print(f"\nsentenceData type: {type(sentence_data)}")

        n_sent = 0  # Will be set properly below

        if isinstance(sentence_data, h5py.Dataset):
            print(f"sentenceData shape: {sentence_data.shape}")
            print(f"sentenceData dtype: {sentence_data.dtype}")

            n_sent = get_ref_count(sentence_data)
            print(f"Number of sentences: {n_sent}")

            # Inspect requested sentence
            if args.sentence < n_sent:
                ref = get_ref_at(sentence_data, args.sentence)
                inspect_sentence(h5file, ref, args.sentence, args.detailed)
            else:
                print(f"Sentence index {args.sentence} out of range (max: {n_sent - 1})")

        elif isinstance(sentence_data, h5py.Group):
            # Struct-array layout: keys are field names, not sentence names
            field_keys = sorted(sentence_data.keys())
            print(f"sentenceData is a struct-array Group with fields: {field_keys}")

            if "content" not in sentence_data or "word" not in sentence_data:
                print("  ERROR: Missing 'content' or 'word' field in sentenceData")
                return

            content_refs = sentence_data["content"]
            word_refs = sentence_data["word"]
            n_sent = get_ref_count(content_refs)
            print(f"Number of sentences: {n_sent}")

            # Inspect requested sentence
            if args.sentence < n_sent:
                text_ref = get_ref_at(content_refs, args.sentence)
                text = h5_to_string(text_ref, h5file)
                word_ref = get_ref_at(word_refs, args.sentence)
                word_group = h5file[word_ref]
                inspect_sentence_from_parts(h5file, text, word_group, args.sentence, args.detailed)
            else:
                print(f"Sentence index {args.sentence} out of range (max: {n_sent - 1})")

        # Summary: check which GD fields exist across a few sentences
        if n_sent == 0:
            print("\nNo sentences found to inspect.")
            return

        print(f"\n{'='*50}")
        print("Quick field availability check (first 3 sentences):")
        print(f"{'='*50}")

        for i in range(min(3, n_sent)):
            try:
                if isinstance(sentence_data, h5py.Dataset):
                    ref = get_ref_at(sentence_data, i)
                    sent = h5file[ref]
                    if "word" in sent:
                        word_group = sent["word"]
                    else:
                        print(f"  Sentence {i}: no 'word' field")
                        continue
                else:
                    # Struct-array: dereference word[i]
                    word_ref = get_ref_at(sentence_data["word"], i)
                    word_group = h5file[word_ref]

                word_fields = list(word_group.keys())
                gd_fields = [f for f in word_fields if f.startswith("GD_")]
                ffd_fields = [f for f in word_fields if f.startswith("FFD_")]
                trt_fields = [f for f in word_fields if f.startswith("TRT_")]
                print(f"  Sentence {i}: GD={gd_fields}, FFD={ffd_fields}, TRT={trt_fields}")
            except Exception as e:
                print(f"  Sentence {i}: error - {e}")


if __name__ == "__main__":
    main()