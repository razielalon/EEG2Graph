"""
EEG-to-Graph: Generative Model (Bridge + BART)
===============================================

Wraps a pretrained `facebook/bart-base` with a Bridge projection that
maps EEG word-level features (batch, src_len, 840) into BART's encoder
embedding space (d_model=768). The structured triplet output format
(`<triplet> subj <subj> rel <rel> obj <obj>`) is preserved — the four
markers are registered as special tokens on the tokenizer, and
`resize_token_embeddings` extends BART's embedding table accordingly.

    ┌──────────────┐        ┌──────────────────┐
    │  EEG Input   │        │  Target Tokens   │
    │ (B, L, 840)  │        │  (shifted right) │
    └──────┬───────┘        └────────┬─────────┘
           │                         │
    ┌──────▼───────┐                 │
    │    Bridge    │                 │
    │ 840 → 768 +  │                 │
    │ LN+GELU+DO   │                 │
    └──────┬───────┘                 │
           │                         │
           │   inputs_embeds         │
           ▼                         ▼
    ┌─────────────────────────────────────────┐
    │   BartForConditionalGeneration          │
    │   (encoder + decoder + lm_head)         │
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
    Bridge + BART encoder-decoder for EEG → triplet generation.

    Args:
        tokenizer: BART tokenizer with structural tokens already added
        eeg_dim:   Input EEG feature dim (840 for ZuCo)
        bart_name: HuggingFace model name (default "facebook/bart-base")
        dropout:   Dropout rate applied inside the Bridge projection
    """

    def __init__(self, tokenizer, eeg_dim=840, bart_name="facebook/bart-base", dropout=0.3):
        super().__init__()
        self.bart_name = bart_name
        self.pad_token_id = tokenizer.pad_token_id

        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        self.bart.resize_token_embeddings(len(tokenizer))

        d_model = self.bart.config.d_model
        self.bridge = nn.Sequential(
            nn.Linear(eeg_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
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

    def param_groups(self, bridge_lr, bart_lr, weight_decay):
        """Differential learning rates: high for Bridge, low for BART."""
        return [
            {"params": self.bridge.parameters(), "lr": bridge_lr, "weight_decay": weight_decay},
            {"params": self.bart.parameters(),   "lr": bart_lr,   "weight_decay": weight_decay},
        ]
