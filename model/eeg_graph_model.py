"""
EEG-to-Graph: Generative Model (Bridge + REBEL)
================================================

Wraps a pretrained `Babelscape/rebel-large` (a BART-large checkpoint
already fine-tuned for relation extraction) with a Bridge projection
that maps EEG word-level features (batch, src_len, 840) into the
encoder embedding space (d_model=1024). REBEL's native triplet format
(`<triplet> subj <subj> obj <obj> rel`) is used as-is — the structural
tokens are already in REBEL's vocabulary, so no embedding resize is
needed.

    ┌──────────────┐        ┌──────────────────┐
    │  EEG Input   │        │  Target Tokens   │
    │ (B, L, 840)  │        │  (shifted right) │
    └──────┬───────┘        └────────┬─────────┘
           │                         │
    ┌──────▼───────┐                 │
    │    Bridge    │                 │
    │  840 → 1024  │                 │
    │    LN+DO     │                 │
    └──────┬───────┘                 │
           │                         │
           │   inputs_embeds         │
           ▼                         ▼
    ┌─────────────────────────────────────────┐
    │   BartForConditionalGeneration          │
    │   (REBEL = BART-large fine-tuned on RE) │
    └─────────────────────────────────────────┘
                       │
                       ▼
                 (B, T, vocab)
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration


class EEGBartModel(nn.Module):
    """
    Bridge + REBEL (BART-large) encoder-decoder for EEG → triplet generation.

    Args:
        tokenizer: REBEL tokenizer (structural tokens already native)
        eeg_dim:   Input EEG feature dim (840 for ZuCo)
        bart_name: HuggingFace model name (default "Babelscape/rebel-large")
        dropout:   Dropout rate applied inside the Bridge projection
    """

    def __init__(self, tokenizer, eeg_dim=840, bart_name="Babelscape/rebel-large", dropout=0.3):
        super().__init__()
        self.bart_name = bart_name
        self.pad_token_id = tokenizer.pad_token_id

        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        # No-op when vocab already matches; keeps us safe if the tokenizer
        # ever gets extra tokens added downstream.
        if self.bart.get_input_embeddings().num_embeddings != len(tokenizer):
            self.bart.resize_token_embeddings(len(tokenizer))

        d_model = self.bart.config.d_model
        self.bridge = nn.Sequential(
            nn.Linear(eeg_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src, src_mask, tgt):
        """
        Teacher-forced forward pass.

        Args:
            src:      (B, S, eeg_dim) — EEG features
            src_mask: (B, S) bool — True for real tokens
            tgt:      (B, T) — decoder input token IDs (shifted right)

        Returns:
            logits: (B, T, vocab_size)
        """
        inputs_embeds = self.bridge(src)
        out = self.bart(
            inputs_embeds=inputs_embeds,
            attention_mask=src_mask.long(),
            decoder_input_ids=tgt,
            use_cache=False,
        )
        return out.logits

    @torch.no_grad()
    def generate(self, src, src_mask, max_len=128, num_beams=1):
        """
        Autoregressive decoding via BART. num_beams=1 = greedy, >1 = beam.

        Returns:
            generated: (B, gen_len) token ID tensor
        """
        inputs_embeds = self.bridge(src)
        return self.bart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=src_mask.long(),
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=num_beams > 1,
            decoder_start_token_id=self.bart.config.decoder_start_token_id,
        )

    def freeze_bart(self):
        """Freeze all BART/REBEL params — only the Bridge trains."""
        for p in self.bart.parameters():
            p.requires_grad = False

    def param_groups(self, bridge_lr, bart_lr, weight_decay):
        """
        Differential learning rates: high for Bridge, low for BART.
        BART params that are frozen (requires_grad=False) are skipped —
        otherwise AdamW would still allocate optimizer state for them.
        """
        groups = [{
            "params": list(self.bridge.parameters()),
            "lr": bridge_lr, "weight_decay": weight_decay,
        }]
        trainable_bart = [p for p in self.bart.parameters() if p.requires_grad]
        if trainable_bart:
            groups.append({
                "params": trainable_bart,
                "lr": bart_lr, "weight_decay": weight_decay,
            })
        return groups
