"""
EEG-to-Graph: Tokenizer Helpers
================================

Wraps a HuggingFace BART tokenizer with REBEL-style linearization:

    <s> <triplet> Barack Obama <subj> place of birth <rel> Hawaii <obj> </s>

`<s>` / `</s>` / `<pad>` come from BART; the four structural markers
(`<triplet>`, `<subj>`, `<rel>`, `<obj>`) are added as special tokens.
"""

import os
from transformers import AutoTokenizer


STRUCT_TOKENS = ["<triplet>", "<subj>", "<rel>", "<obj>"]


def build_tokenizer(model_name="facebook/bart-base"):
    """Load a BART tokenizer and register the structural marker tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(STRUCT_TOKENS, special_tokens=True)
    return tokenizer


def _triplets_to_string(triplets):
    parts = []
    for t in triplets:
        parts.append(
            f"<triplet> {t['subject'].strip()} "
            f"<subj> {t['relation'].strip()} "
            f"<rel> {t['object'].strip()} <obj>"
        )
    return " ".join(parts)


def linearize_triplets(triplets, tokenizer):
    """
    Convert triplet dicts to a BART token ID sequence.

    Empty triplets yield just [<s>, </s>].
    """
    text = _triplets_to_string(triplets)
    return tokenizer(text, add_special_tokens=True)["input_ids"]


def delinearize(token_ids, tokenizer):
    """
    Convert a BART token ID sequence back into triplet dicts.

    Robust to partial/malformed output — decodes to a string and parses
    the structural markers, extracting as many well-formed triplets as
    possible.
    """
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for marker in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        if marker:
            text = text.replace(marker, " ")

    triplets = []
    chunks = text.split("<triplet>")
    for chunk in chunks[1:]:
        if "<subj>" not in chunk or "<rel>" not in chunk or "<obj>" not in chunk:
            continue
        subj, rest = chunk.split("<subj>", 1)
        rel, rest = rest.split("<rel>", 1)
        obj, _ = rest.split("<obj>", 1)

        subj = subj.strip()
        rel = rel.strip()
        obj = obj.strip()
        if subj and rel and obj:
            triplets.append({"subject": subj, "relation": rel, "object": obj})

    return triplets


def save_tokenizer(tokenizer, directory):
    os.makedirs(directory, exist_ok=True)
    tokenizer.save_pretrained(directory)


def load_tokenizer(directory):
    return AutoTokenizer.from_pretrained(directory)
