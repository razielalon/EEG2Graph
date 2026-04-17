"""
Tests for the model/ module — logic correctness and data format integration.
"""
import os
import sys
import json
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from vocabulary import (
    STRUCT_TOKENS,
    build_tokenizer,
    linearize_triplets,
    delinearize,
    save_tokenizer,
    load_tokenizer,
)
from eeg_graph_model import EEGBartModel
from eeg_graph_dataset import EEGGraphDataset, collate_fn, build_dataloaders


# =============================================================================
# Helpers
# =============================================================================

SAMPLE_TRIPLETS_LIST = [
    {
        "text": "Barack Obama was born in Hawaii .",
        "triplets": [
            {"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}
        ],
    },
    {
        "text": "Albert Einstein developed the theory of relativity .",
        "triplets": [
            {"subject": "Albert Einstein", "relation": "developed", "object": "theory of relativity"}
        ],
    },
]

SAMPLE_TRIPLETS_DICT = {
    "Barack Obama was born in Hawaii .": {
        "triplets": [
            {"subject": "Barack Obama", "relation": "place of birth", "object": "Hawaii"}
        ]
    },
    "Albert Einstein developed the theory of relativity .": {
        "triplets": [
            {"subject": "Albert Einstein", "relation": "developed", "object": "theory of relativity"}
        ]
    },
}


# Module-level tokenizer so we don't re-download for every test.
_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = build_tokenizer()
    return _TOKENIZER


def make_test_data(tmpdir, triplets_format="dict"):
    n_samples = 4
    n_words = [5, 8, 5, 8]
    feat_dim = 840
    texts = [
        "Barack Obama was born in Hawaii .",
        "Albert Einstein developed the theory of relativity .",
        "Barack Obama was born in Hawaii .",
        "Albert Einstein developed the theory of relativity .",
    ]

    eeg_list = [np.random.randn(nw, feat_dim).astype(np.float32) for nw in n_words]
    np.save(os.path.join(tmpdir, "train_eeg.npy"), np.array(eeg_list, dtype=object), allow_pickle=True)

    meta = []
    for i in range(n_samples):
        words = texts[i].split()[:n_words[i]]
        meta.append({
            "text": texts[i],
            "words": words,
            "subject_id": "YAC",
            "task": "task1-NR",
            "has_fixation": [True] * n_words[i],
            "n_words": n_words[i],
        })
    with open(os.path.join(tmpdir, "train_meta.json"), "w") as f:
        json.dump(meta, f)

    if triplets_format == "dict":
        triplets = SAMPLE_TRIPLETS_DICT
    else:
        triplets = SAMPLE_TRIPLETS_LIST

    triplets_path = os.path.join(tmpdir, "triplets.json")
    with open(triplets_path, "w") as f:
        json.dump(triplets, f)

    return triplets_path


# =============================================================================
# Tokenizer Tests
# =============================================================================

def test_tokenizer_has_struct_tokens():
    """Tokenizer knows about the four structural marker tokens."""
    tok = get_tokenizer()
    for marker in STRUCT_TOKENS:
        ids = tok(marker, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"{marker} should tokenize to a single ID, got {ids}"
    print("  PASS: test_tokenizer_has_struct_tokens")


def test_linearize_delinearize_roundtrip():
    """linearize_triplets then delinearize recovers original triplets."""
    tok = get_tokenizer()
    triplets = SAMPLE_TRIPLETS_LIST[0]["triplets"]
    ids = linearize_triplets(triplets, tok)

    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id

    recovered = delinearize(ids, tok)
    assert len(recovered) == len(triplets), f"expected {len(triplets)}, got {len(recovered)}"
    assert recovered[0]["subject"] == "Barack Obama"
    assert recovered[0]["relation"] == "place of birth"
    assert recovered[0]["object"] == "Hawaii"
    print("  PASS: test_linearize_delinearize_roundtrip")


def test_linearize_multiple_triplets():
    """Multi-triplet linearization round-trips."""
    tok = get_tokenizer()
    triplets = [
        {"subject": "A", "relation": "r1", "object": "B"},
        {"subject": "C", "relation": "r2", "object": "D"},
    ]
    ids = linearize_triplets(triplets, tok)
    recovered = delinearize(ids, tok)
    assert len(recovered) == 2
    assert recovered[0]["subject"] == "A" and recovered[1]["subject"] == "C"
    print("  PASS: test_linearize_multiple_triplets")


def test_linearize_empty():
    """Linearizing no triplets gives just [bos, eos]."""
    tok = get_tokenizer()
    ids = linearize_triplets([], tok)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert delinearize(ids, tok) == []
    print("  PASS: test_linearize_empty")


def test_tokenizer_save_load():
    """save/load tokenizer roundtrip preserves structural tokens."""
    tok = get_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tokenizer(tok, tmpdir)
        loaded = load_tokenizer(tmpdir)
        assert len(loaded) == len(tok)
        for marker in STRUCT_TOKENS:
            assert loaded(marker, add_special_tokens=False)["input_ids"] == \
                   tok(marker, add_special_tokens=False)["input_ids"]
    print("  PASS: test_tokenizer_save_load")


# =============================================================================
# Dataset Tests
# =============================================================================

def test_dataset_dict_format():
    tok = get_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="dict")
        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, tok,
        )
        assert len(ds) == 4
        sample = ds[0]
        assert sample["eeg"].shape[1] == 840
        assert sample["target_ids"].dim() == 1
        assert sample["target_ids"][0].item() == tok.bos_token_id
    print("  PASS: test_dataset_dict_format")


def test_dataset_list_format():
    tok = get_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="list")
        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, tok,
        )
        assert len(ds) == 4
    print("  PASS: test_dataset_list_format")


def test_collate_fn():
    tok = get_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="dict")
        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, tok,
        )
        batch = collate_fn([ds[0], ds[1]], pad_id=tok.pad_token_id)
        assert batch["src"].shape[0] == 2
        assert batch["src"].shape[2] == 840
        assert batch["tgt"].shape[0] == 2
        assert batch["tgt_labels"].shape == batch["tgt"].shape
        assert batch["tgt_mask"].shape == batch["tgt"].shape
        max_tgt = max(ds[0]["n_tgt"], ds[1]["n_tgt"])
        assert batch["tgt"].shape[1] == max_tgt - 1
    print("  PASS: test_collate_fn")


def test_build_dataloaders_with_real_data():
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "processed_zuco")
    triplets_path = os.path.join(processed_dir, "sentence_triplets.json")

    if not os.path.exists(triplets_path):
        print("  SKIP: test_build_dataloaders_with_real_data (no processed_zuco)")
        return

    loaders, tokenizer = build_dataloaders(
        processed_dir, triplets_path,
        batch_size=4, max_src_len=32, max_tgt_len=32,
        tokenizer=get_tokenizer(),
    )

    assert "train" in loaders
    assert len(tokenizer) > len(STRUCT_TOKENS)

    batch = next(iter(loaders["train"]))
    B = batch["src"].shape[0]
    assert B <= 4
    assert batch["src"].shape[2] == 840
    assert batch["tgt"].dim() == 2
    assert batch["tgt_labels"].shape == batch["tgt"].shape
    print("  PASS: test_build_dataloaders_with_real_data")


# =============================================================================
# Model Tests
# =============================================================================

def test_model_forward():
    tok = get_tokenizer()
    model = EEGBartModel(tok, eeg_dim=840, dropout=0.0)
    model.eval()

    B, S, T = 2, 10, 8
    src = torch.randn(B, S, 840)
    src_mask = torch.ones(B, S, dtype=torch.bool)
    tgt = torch.randint(0, len(tok), (B, T))

    logits = model(src, src_mask, tgt)
    assert logits.shape == (B, T, len(tok))
    print("  PASS: test_model_forward")


def test_model_generate_greedy():
    tok = get_tokenizer()
    model = EEGBartModel(tok, eeg_dim=840, dropout=0.0)
    model.eval()

    src = torch.randn(1, 5, 840)
    src_mask = torch.ones(1, 5, dtype=torch.bool)

    out1 = model.generate(src, src_mask, max_len=10, num_beams=1)
    out2 = model.generate(src, src_mask, max_len=10, num_beams=1)
    assert torch.equal(out1, out2), "Greedy decoding should be deterministic"
    print("  PASS: test_model_generate_greedy")


def test_model_generate_beam():
    tok = get_tokenizer()
    model = EEGBartModel(tok, eeg_dim=840, dropout=0.0)
    model.eval()

    src = torch.randn(1, 5, 840)
    src_mask = torch.ones(1, 5, dtype=torch.bool)

    result = model.generate(src, src_mask, max_len=20, num_beams=4)
    assert result.shape[0] == 1
    assert result.dim() == 2
    print("  PASS: test_model_generate_beam")


def test_model_param_groups():
    tok = get_tokenizer()
    model = EEGBartModel(tok, eeg_dim=840, dropout=0.0)
    groups = model.param_groups(bridge_lr=3e-4, bart_lr=3e-5, weight_decay=0.01)
    assert len(groups) == 2
    assert groups[0]["lr"] == 3e-4
    assert groups[1]["lr"] == 3e-5
    # Bridge params are a strict subset; bart has many more.
    bridge_n = sum(p.numel() for p in groups[0]["params"])
    bart_n = sum(p.numel() for p in groups[1]["params"])
    assert bart_n > bridge_n * 10
    print("  PASS: test_model_param_groups")


# =============================================================================
# Integration: end-to-end with real data
# =============================================================================

def test_end_to_end_with_real_data():
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "processed_zuco")
    triplets_path = os.path.join(processed_dir, "sentence_triplets.json")

    if not os.path.exists(triplets_path):
        print("  SKIP: test_end_to_end_with_real_data (no processed_zuco)")
        return

    loaders, tokenizer = build_dataloaders(
        processed_dir, triplets_path,
        batch_size=2, max_src_len=32, max_tgt_len=32,
        tokenizer=get_tokenizer(),
    )

    model = EEGBartModel(tokenizer, eeg_dim=840, dropout=0.0)
    model.eval()

    batch = next(iter(loaders["train"]))

    logits = model(batch["src"], batch["src_mask"], batch["tgt"])
    assert logits.shape[0] == batch["tgt"].shape[0]
    assert logits.shape[1] == batch["tgt"].shape[1]
    assert logits.shape[2] == len(tokenizer)

    generated = model.generate(batch["src"], batch["src_mask"], max_len=20, num_beams=1)
    assert generated.shape[0] == batch["src"].shape[0]

    for i in range(generated.shape[0]):
        triplets = delinearize(generated[i].tolist(), tokenizer)
        assert isinstance(triplets, list)

    print("  PASS: test_end_to_end_with_real_data")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running model tests")
    print("=" * 60)

    tests = [
        # Tokenizer
        test_tokenizer_has_struct_tokens,
        test_linearize_delinearize_roundtrip,
        test_linearize_multiple_triplets,
        test_linearize_empty,
        test_tokenizer_save_load,
        # Dataset
        test_dataset_dict_format,
        test_dataset_list_format,
        test_collate_fn,
        test_build_dataloaders_with_real_data,
        # Model
        test_model_forward,
        test_model_generate_greedy,
        test_model_generate_beam,
        test_model_param_groups,
        # Integration
        test_end_to_end_with_real_data,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
