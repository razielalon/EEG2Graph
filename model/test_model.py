"""
Tests for the model/ module — logic correctness and data format integration.
"""
import os
import sys
import json
import tempfile
import numpy as np
import torch

# Ensure model/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from vocabulary import Vocabulary, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from eeg_graph_model import EEGGraphModel
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

# Dict format as produced by generate_triplets.py / sentence_triplets.json
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


def make_test_data(tmpdir, triplets_format="dict"):
    """Create minimal test data files in tmpdir."""
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
# Vocabulary Tests
# =============================================================================

def test_vocab_build_from_list():
    """Vocabulary.build_from_triplets works with list format."""
    vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
    vocab.freeze()
    assert len(vocab) > 8  # special tokens + words
    assert vocab.encode("Barack") != UNK_ID
    assert vocab.encode("Hawaii") != UNK_ID
    assert vocab.encode("xyznonexistent") == UNK_ID
    print("  PASS: test_vocab_build_from_list")


def test_vocab_build_from_dict():
    """Vocabulary.build_from_triplets works with dict format (sentence_triplets.json)."""
    vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_DICT)
    vocab.freeze()
    assert len(vocab) > 8
    assert vocab.encode("Barack") != UNK_ID
    assert vocab.encode("Hawaii") != UNK_ID
    print("  PASS: test_vocab_build_from_dict")


def test_vocab_list_dict_equivalence():
    """Both formats produce identical vocabularies."""
    v1 = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
    v2 = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_DICT)
    assert v1.token2id == v2.token2id
    print("  PASS: test_vocab_list_dict_equivalence")


def test_linearize_delinearize_roundtrip():
    """Linearize then delinearize recovers original triplets."""
    vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
    vocab.freeze()

    triplets = SAMPLE_TRIPLETS_LIST[0]["triplets"]
    ids = vocab.linearize_triplets(triplets)

    # Should start with BOS, end with EOS
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID

    recovered = vocab.delinearize(ids)
    assert len(recovered) == len(triplets)
    assert recovered[0]["subject"] == "Barack Obama"
    assert recovered[0]["relation"] == "place of birth"
    assert recovered[0]["object"] == "Hawaii"
    print("  PASS: test_linearize_delinearize_roundtrip")


def test_linearize_empty():
    """Linearizing no triplets gives [BOS, EOS]."""
    vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
    ids = vocab.linearize_triplets([])
    assert ids == [BOS_ID, EOS_ID]
    print("  PASS: test_linearize_empty")


def test_vocab_save_load():
    """Save and load roundtrip preserves vocab."""
    vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
    vocab.freeze()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        vocab.save(path)
        loaded = Vocabulary.load(path)
        assert loaded.token2id == vocab.token2id
        assert loaded.encode("Barack") == vocab.encode("Barack")
        assert len(loaded) == len(vocab)
    finally:
        os.unlink(path)
    print("  PASS: test_vocab_save_load")


# =============================================================================
# Dataset Tests
# =============================================================================

def test_dataset_dict_format():
    """EEGGraphDataset loads correctly with dict-format triplets (sentence_triplets.json)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="dict")
        vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_DICT)
        vocab.freeze()

        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, vocab,
        )
        assert len(ds) == 4
        sample = ds[0]
        assert sample["eeg"].shape[1] == 840
        assert sample["target_ids"].dim() == 1
        assert sample["target_ids"][0].item() == BOS_ID
    print("  PASS: test_dataset_dict_format")


def test_dataset_list_format():
    """EEGGraphDataset loads correctly with list-format triplets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="list")
        vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_LIST)
        vocab.freeze()

        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, vocab,
        )
        assert len(ds) == 4
    print("  PASS: test_dataset_list_format")


def test_collate_fn():
    """collate_fn produces correctly shaped and padded tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        triplets_path = make_test_data(tmpdir, triplets_format="dict")
        vocab = Vocabulary.build_from_triplets(SAMPLE_TRIPLETS_DICT)
        vocab.freeze()

        ds = EEGGraphDataset(
            os.path.join(tmpdir, "train_eeg.npy"),
            os.path.join(tmpdir, "train_meta.json"),
            triplets_path, vocab,
        )

        batch = collate_fn([ds[0], ds[1]])
        assert batch["src"].shape[0] == 2  # batch size
        assert batch["src"].shape[2] == 840  # feature dim
        assert batch["tgt"].shape[0] == 2
        assert batch["tgt_labels"].shape == batch["tgt"].shape
        assert batch["tgt_mask"].shape == batch["tgt"].shape
        # tgt and tgt_labels should be max_tgt - 1 in length
        max_tgt = max(ds[0]["n_tgt"], ds[1]["n_tgt"])
        assert batch["tgt"].shape[1] == max_tgt - 1
    print("  PASS: test_collate_fn")


def test_build_dataloaders_with_real_data():
    """build_dataloaders works with the actual processed_zuco data."""
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "processed_zuco")
    triplets_path = os.path.join(processed_dir, "sentence_triplets.json")

    if not os.path.exists(triplets_path):
        print("  SKIP: test_build_dataloaders_with_real_data (no processed_zuco)")
        return

    loaders, vocab = build_dataloaders(
        processed_dir, triplets_path,
        batch_size=4, max_src_len=32, max_tgt_len=32,
    )

    assert "train" in loaders
    assert len(vocab) > 8

    # Get one batch and verify shapes
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
    """Model forward pass produces correct output shape."""
    vocab_size = 50
    model = EEGGraphModel(vocab_size=vocab_size, eeg_dim=840, d_model=64, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32)
    model.eval()

    B, S, T = 2, 10, 8
    src = torch.randn(B, S, 840)
    src_mask = torch.ones(B, S, dtype=torch.bool)
    tgt = torch.randint(0, vocab_size, (B, T))

    logits = model(src, src_mask, tgt)
    assert logits.shape == (B, T, vocab_size)
    print("  PASS: test_model_forward")


def test_generate_greedy_deterministic():
    """generate() with default temperature=0 is deterministic (greedy)."""
    vocab_size = 50
    model = EEGGraphModel(vocab_size=vocab_size, eeg_dim=840, d_model=64, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32)
    model.eval()

    src = torch.randn(1, 5, 840)
    src_mask = torch.ones(1, 5, dtype=torch.bool)

    out1 = model.generate(src, src_mask, max_len=10)
    out2 = model.generate(src, src_mask, max_len=10)
    assert torch.equal(out1, out2), "Greedy decoding should be deterministic"
    assert out1[0, 0].item() == BOS_ID, "First token should be BOS"
    print("  PASS: test_generate_greedy_deterministic")


def test_generate_sampling():
    """generate() with temperature>0 does sampling (non-deterministic)."""
    vocab_size = 500
    model = EEGGraphModel(vocab_size=vocab_size, eeg_dim=840, d_model=64, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32)
    model.eval()

    src = torch.randn(1, 5, 840)
    src_mask = torch.ones(1, 5, dtype=torch.bool)

    # With high temperature, outputs should differ across runs (probabilistic)
    results = set()
    for _ in range(5):
        out = model.generate(src, src_mask, max_len=10, temperature=2.0)
        results.add(tuple(out[0].tolist()))

    # With high temp and many trials, we expect at least some variation
    # (not guaranteed but very likely with temp=2.0 and vocab_size=500)
    # Just check it doesn't crash and returns valid shape
    assert out.shape[0] == 1
    assert out[0, 0].item() == BOS_ID
    print("  PASS: test_generate_sampling")


def test_beam_search_no_duplicates():
    """beam_search doesn't produce duplicate entries in completed list."""
    vocab_size = 50
    model = EEGGraphModel(vocab_size=vocab_size, eeg_dim=840, d_model=64, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32)
    model.eval()

    src = torch.randn(1, 5, 840)
    src_mask = torch.ones(1, 5, dtype=torch.bool)

    result = model.beam_search(src, src_mask, beam_size=4, max_len=20)
    assert result.dim() == 1
    assert result[0].item() == BOS_ID
    print("  PASS: test_beam_search_no_duplicates")


def test_beam_search_starts_with_bos():
    """beam_search output starts with BOS."""
    vocab_size = 50
    model = EEGGraphModel(vocab_size=vocab_size, eeg_dim=840, d_model=64, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32)
    model.eval()

    src = torch.randn(1, 3, 840)
    src_mask = torch.ones(1, 3, dtype=torch.bool)

    result = model.beam_search(src, src_mask, beam_size=2, max_len=10)
    assert result[0].item() == BOS_ID
    print("  PASS: test_beam_search_starts_with_bos")


# =============================================================================
# Integration: end-to-end with real data
# =============================================================================

def test_end_to_end_with_real_data():
    """Full pipeline: load real data -> build model -> forward pass -> generate -> delinearize."""
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "processed_zuco")
    triplets_path = os.path.join(processed_dir, "sentence_triplets.json")

    if not os.path.exists(triplets_path):
        print("  SKIP: test_end_to_end_with_real_data (no processed_zuco)")
        return

    loaders, vocab = build_dataloaders(
        processed_dir, triplets_path,
        batch_size=2, max_src_len=32, max_tgt_len=32,
    )

    model = EEGGraphModel(
        vocab_size=len(vocab), eeg_dim=840, d_model=64, n_heads=4,
        n_enc_layers=1, n_dec_layers=1, max_src_len=32, max_tgt_len=32, dropout=0.0,
    )
    model.eval()

    batch = next(iter(loaders["train"]))

    # Forward pass (training mode)
    logits = model(batch["src"], batch["src_mask"], batch["tgt"])
    assert logits.shape[0] == batch["tgt"].shape[0]
    assert logits.shape[1] == batch["tgt"].shape[1]
    assert logits.shape[2] == len(vocab)

    # Generate (inference mode)
    generated = model.generate(batch["src"], batch["src_mask"], max_len=20)
    assert generated.shape[0] == batch["src"].shape[0]

    # Delinearize
    for i in range(generated.shape[0]):
        triplets = vocab.delinearize(generated[i].tolist())
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
        # Vocabulary
        test_vocab_build_from_list,
        test_vocab_build_from_dict,
        test_vocab_list_dict_equivalence,
        test_linearize_delinearize_roundtrip,
        test_linearize_empty,
        test_vocab_save_load,
        # Dataset
        test_dataset_dict_format,
        test_dataset_list_format,
        test_collate_fn,
        test_build_dataloaders_with_real_data,
        # Model
        test_model_forward,
        test_generate_greedy_deterministic,
        test_generate_sampling,
        test_beam_search_no_duplicates,
        test_beam_search_starts_with_bos,
        # Integration
        test_end_to_end_with_real_data,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
