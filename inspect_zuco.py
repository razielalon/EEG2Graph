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
            print(f"{prefix}📁 {key}/ ({len(item)} items)")
            print_h5_structure(item, prefix + "    ", max_depth, current_depth + 1)
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            print(f"{prefix}📄 {key}: shape={shape}, dtype={dtype}")
        else:
            print(f"{prefix}❓ {key}: {type(item)}")


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


def inspect_sentence(h5file, sentence_ref, sentence_idx, detailed=False):
    """Inspect a single sentence's data structure."""
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
        word_group = sent["word"]
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

        if isinstance(sentence_data, h5py.Dataset):
            print(f"sentenceData shape: {sentence_data.shape}")
            print(f"sentenceData dtype: {sentence_data.dtype}")

            n_sent = sentence_data.shape[-1] if len(sentence_data.shape) > 1 else sentence_data.shape[0]
            print(f"Number of sentences: {n_sent}")

            # Inspect requested sentence
            if args.sentence < n_sent:
                ref = sentence_data[0, args.sentence] if len(sentence_data.shape) > 1 else sentence_data[args.sentence]
                inspect_sentence(h5file, ref, args.sentence, args.detailed)
            else:
                print(f"Sentence index {args.sentence} out of range (max: {n_sent - 1})")

        elif isinstance(sentence_data, h5py.Group):
            print(f"sentenceData is a Group with {len(sentence_data)} items")
            print(f"Keys (first 5): {list(sentence_data.keys())[:5]}")

        # Summary: check which GD fields exist across a few sentences
        print(f"\n{'='*50}")
        print("Quick field availability check (first 3 sentences):")
        print(f"{'='*50}")

        for i in range(min(3, n_sent)):
            try:
                ref = sentence_data[0, i] if len(sentence_data.shape) > 1 else sentence_data[i]
                sent = h5file[ref]
                if "word" in sent:
                    word_fields = list(sent["word"].keys())
                    gd_fields = [f for f in word_fields if f.startswith("GD_")]
                    ffd_fields = [f for f in word_fields if f.startswith("FFD_")]
                    trt_fields = [f for f in word_fields if f.startswith("TRT_")]
                    print(f"  Sentence {i}: GD={gd_fields}, FFD={ffd_fields}, TRT={trt_fields}")
            except Exception as e:
                print(f"  Sentence {i}: error - {e}")


if __name__ == "__main__":
    main()
