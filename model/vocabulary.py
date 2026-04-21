"""
EEG-to-Graph: Tokenizer Helpers
================================

Wraps the REBEL (Babelscape/rebel-large) tokenizer. REBEL is BART-large
already fine-tuned for end-to-end relation extraction, and it ships with
the structural markers (`<triplet>`, `<subj>`, `<obj>`) as native tokens.
We therefore reuse its native linearization format:

    <s> <triplet> Barack Obama <subj> Hawaii <obj> place of birth </s>

Subject first, then object, then relation — no `<rel>` marker. This
matches REBEL's pretraining so the decoder already knows the grammar.
"""

import os
from transformers import AutoTokenizer


# Native to REBEL's vocab — no add_tokens needed.
STRUCT_TOKENS = ["<triplet>", "<subj>", "<obj>"]

DEFAULT_MODEL_NAME = "Babelscape/rebel-large"


def build_tokenizer(model_name=DEFAULT_MODEL_NAME):
    """Load the REBEL tokenizer (structural tokens are already in vocab)."""
    return AutoTokenizer.from_pretrained(model_name)


def _triplets_to_string(triplets):
    parts = []
    for t in triplets:
        parts.append(
            f"<triplet> {t['subject'].strip()} "
            f"<subj> {t['object'].strip()} "
            f"<obj> {t['relation'].strip()}"
        )
    return " ".join(parts)


def linearize_triplets(triplets, tokenizer):
    """
    Convert triplet dicts to a REBEL-style token ID sequence.

    Empty triplets yield just [<s>, </s>].
    """
    text = _triplets_to_string(triplets)
    return tokenizer(text, add_special_tokens=True)["input_ids"]


def delinearize(token_ids, tokenizer):
    """
    Parse a REBEL-style token ID sequence back into triplet dicts.

    Robust to partial/malformed output — decodes to a string and walks
    the structural markers in their native order: subject, object, relation.
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
        if "<subj>" not in chunk or "<obj>" not in chunk:
            continue
        subj, rest = chunk.split("<subj>", 1)
        obj, rel_rest = rest.split("<obj>", 1)

        # Relation runs until the next <triplet> (already handled by outer split)
        # or end of string. Strip any trailing structural noise.
        rel = rel_rest.split("<triplet>")[0]

        subj = subj.strip()
        obj = obj.strip()
        rel = rel.strip()
        if subj and rel and obj:
            triplets.append({"subject": subj, "relation": rel, "object": obj})

    return triplets


def save_tokenizer(tokenizer, directory):
    os.makedirs(directory, exist_ok=True)
    tokenizer.save_pretrained(directory)


def load_tokenizer(directory):
    return AutoTokenizer.from_pretrained(directory)
