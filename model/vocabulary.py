"""
EEG-to-Graph: Decoder Vocabulary
=================================

Manages the token vocabulary for the autoregressive triplet decoder.
Uses REBEL-style linearization:

    <bos> <triplet> Barack Obama <subj> place of birth <rel> Hawaii <obj> <eos>

Multiple triplets per sentence are concatenated:
    <bos> <triplet> subj1 <subj> rel1 <rel> obj1 <obj>
          <triplet> subj2 <subj> rel2 <rel> obj2 <obj> <eos>
"""

import json
from collections import Counter


# Special tokens for triplet structure
SPECIAL_TOKENS = [
    "<pad>",      # 0 — padding
    "<bos>",      # 1 — beginning of sequence
    "<eos>",      # 2 — end of sequence
    "<unk>",      # 3 — unknown word
    "<triplet>",  # 4 — triplet boundary marker
    "<subj>",     # 5 — end of subject, start of relation
    "<rel>",      # 6 — end of relation, start of object
    "<obj>",      # 7 — end of object
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class Vocabulary:
    """
    Word-level vocabulary for the triplet decoder.

    Built from the ground-truth triplets in the training set.
    All subject/relation/object words are included.
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self._frozen = False

        # Add special tokens first
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def add(self, token):
        """Add a token (only if vocabulary is not frozen)."""
        if self._frozen and token not in self.token2id:
            return UNK_ID
        return self._add(token)

    def freeze(self):
        self._frozen = True

    def encode(self, token):
        return self.token2id.get(token, UNK_ID)

    def decode(self, idx):
        return self.id2token.get(idx, "<unk>")

    def __len__(self):
        return len(self.token2id)

    def __contains__(self, token):
        return token in self.token2id

    # ----- Build from triplets -----

    @classmethod
    def build_from_triplets(cls, triplets_data, min_freq=1):
        """
        Build vocabulary from triplet entries.

        Args:
            triplets_data: Either:
                - list of {"text": ..., "triplets": [{"subject":..., "relation":..., "object":...}]}
                - dict of {sentence_text: {"triplets": [...]}}  (as produced by generate_triplets.py)
            min_freq: Minimum word frequency to include

        Returns:
            Vocabulary instance
        """
        vocab = cls()
        counter = Counter()

        # Normalize: accept both list-of-dicts and dict-of-dicts formats
        if isinstance(triplets_data, dict):
            entries = triplets_data.values()
        else:
            entries = triplets_data

        for entry in entries:
            for triplet in entry.get("triplets", []):
                for field in ["subject", "relation", "object"]:
                    tokens = triplet[field].split()
                    counter.update(tokens)

        for token, freq in counter.items():
            if freq >= min_freq:
                vocab.add(token)

        print(f"  Vocabulary: {len(vocab)} tokens "
              f"({len(SPECIAL_TOKENS)} special + {len(vocab) - len(SPECIAL_TOKENS)} words)")
        return vocab

    # ----- Linearization -----

    def linearize_triplets(self, triplets):
        """
        Convert a list of triplet dicts to a token ID sequence.

        Input:
            [{"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}]

        Output token sequence:
            <bos> <triplet> Barack Obama <subj> place of birth <rel> Hawaii <obj> <eos>

        Returns:
            list of token IDs
        """
        ids = [BOS_ID]

        for triplet in triplets:
            ids.append(self.encode("<triplet>"))

            for word in triplet["subject"].split():
                ids.append(self.encode(word))
            ids.append(self.encode("<subj>"))

            for word in triplet["relation"].split():
                ids.append(self.encode(word))
            ids.append(self.encode("<rel>"))

            for word in triplet["object"].split():
                ids.append(self.encode(word))
            ids.append(self.encode("<obj>"))

        ids.append(EOS_ID)
        return ids

    def delinearize(self, token_ids):
        """
        Convert a token ID sequence back to a list of triplet dicts.

        Robust to partial/malformed outputs — extracts as many valid
        triplets as possible.

        Returns:
            list of {"subject": str, "relation": str, "object": str}
        """
        tokens = [self.decode(i) for i in token_ids]
        triplets = []

        # State machine: look for <triplet> ... <subj> ... <rel> ... <obj>
        i = 0
        while i < len(tokens):
            if tokens[i] == "<triplet>":
                subj_tokens = []
                rel_tokens = []
                obj_tokens = []
                i += 1

                # Collect subject tokens until <subj>
                while i < len(tokens) and tokens[i] not in SPECIAL_TOKENS:
                    subj_tokens.append(tokens[i])
                    i += 1
                if i < len(tokens) and tokens[i] == "<subj>":
                    i += 1
                else:
                    continue

                # Collect relation tokens until <rel>
                while i < len(tokens) and tokens[i] not in SPECIAL_TOKENS:
                    rel_tokens.append(tokens[i])
                    i += 1
                if i < len(tokens) and tokens[i] == "<rel>":
                    i += 1
                else:
                    continue

                # Collect object tokens until <obj>
                while i < len(tokens) and tokens[i] not in SPECIAL_TOKENS:
                    obj_tokens.append(tokens[i])
                    i += 1
                if i < len(tokens) and tokens[i] == "<obj>":
                    i += 1

                if subj_tokens and rel_tokens and obj_tokens:
                    triplets.append({
                        "subject": " ".join(subj_tokens),
                        "relation": " ".join(rel_tokens),
                        "object": " ".join(obj_tokens),
                    })
            elif tokens[i] == "<eos>":
                break
            else:
                i += 1

        return triplets

    # ----- Persistence -----

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "token2id": self.token2id,
                "special_tokens": SPECIAL_TOKENS,
            }, f, indent=2)

    @classmethod
    def load(cls, path):
        vocab = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        vocab._frozen = True
        return vocab