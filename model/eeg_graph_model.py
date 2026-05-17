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

    def __init__(self, tokenizer, eeg_dim=840, bart_name="Babelscape/rebel-large", dropout=0.3,
                 bridge_layers=1, bart_dropout=None, bart_attention_dropout=None):
        super().__init__()
        self.bart_name = bart_name
        self.pad_token_id = tokenizer.pad_token_id

        bart_kwargs = {}
        if bart_dropout is not None:
            bart_kwargs["dropout"] = bart_dropout
            bart_kwargs["activation_dropout"] = bart_dropout
        if bart_attention_dropout is not None:
            bart_kwargs["attention_dropout"] = bart_attention_dropout
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name, **bart_kwargs)
        # No-op when vocab already matches; keeps us safe if the tokenizer
        # ever gets extra tokens added downstream.
        if self.bart.get_input_embeddings().num_embeddings != len(tokenizer):
            self.bart.resize_token_embeddings(len(tokenizer))

        d_model = self.bart.config.d_model
        layers = []
        in_dim = eeg_dim
        for i in range(bridge_layers):
            layers.append(nn.Linear(in_dim, d_model))
            if i < bridge_layers - 1:
                layers.append(nn.GELU())
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout))
            in_dim = d_model
        self.bridge = nn.Sequential(*layers)

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

    def unfreeze_bart(self):
        """Unfreeze all BART/REBEL params for full fine-tuning."""
        for p in self.bart.parameters():
            p.requires_grad = True

    def param_groups(self, bridge_lr, bart_lr, weight_decay):
        """
        Differential learning rates: high for Bridge, low for BART.
        BART params that are frozen (requires_grad=False) are skipped —
        otherwise AdamW would still allocate optimizer state for them.
        """
        groups = [{
            "name": "bridge",
            "params": list(self.bridge.parameters()),
            "lr": bridge_lr, "weight_decay": weight_decay,
        }]
        trainable_bart = [p for p in self.bart.parameters() if p.requires_grad]
        if trainable_bart:
            groups.append({
                "name": "bart",
                "params": trainable_bart,
                "lr": bart_lr, "weight_decay": weight_decay,
            })
        return groups

    def param_groups_llrd(self, bridge_lr, bart_lr, bridge_wd, bart_wd, llrd_gamma=0.8):
        """
        Layer-wise LR decay (LLRD) for BART, plus the bridge group.

        Encoder and decoder are treated as parallel stacks: each TOP layer
        gets full `bart_lr`; the layer `k` below the top gets
        `bart_lr * llrd_gamma**k`. Token/position embeddings (shared between
        encoder, decoder, and lm_head via weight tying) get the smallest LR.

        `lm_head.weight` is tied to `model.shared.weight`; deduplicated via
        `id(p)` so it lands in the embeddings group exactly once and AdamW
        doesn't see the same Parameter twice.

        Returns 1 group (bridge only) when BART is frozen; otherwise 1 +
        n_enc + n_dec + 1 ≈ 26 groups for BART-large.
        """
        seen = set()
        groups = []

        bridge_params = [p for p in self.bridge.parameters() if p.requires_grad]
        if bridge_params:
            groups.append({
                "name": "bridge",
                "params": bridge_params,
                "lr": bridge_lr,
                "weight_decay": bridge_wd,
            })
            for p in bridge_params:
                seen.add(id(p))

        trainable_bart = [p for p in self.bart.parameters() if p.requires_grad]
        if not trainable_bart:
            return groups

        enc_layers = self.bart.model.encoder.layers
        dec_layers = self.bart.model.decoder.layers
        n_enc = len(enc_layers)
        n_dec = len(dec_layers)
        max_depth = max(n_enc, n_dec)

        # Encoder layers — top (index n_enc-1) gets full bart_lr.
        for i, layer in enumerate(enc_layers):
            depth = n_enc - 1 - i
            lr_i = bart_lr * (llrd_gamma ** depth)
            params = [p for p in layer.parameters()
                      if p.requires_grad and id(p) not in seen]
            if params:
                groups.append({
                    "name": f"bart_enc_{i}",
                    "params": params,
                    "lr": lr_i,
                    "weight_decay": bart_wd,
                })
                for p in params:
                    seen.add(id(p))

        # Decoder layers — top (index n_dec-1) gets full bart_lr.
        for i, layer in enumerate(dec_layers):
            depth = n_dec - 1 - i
            lr_i = bart_lr * (llrd_gamma ** depth)
            params = [p for p in layer.parameters()
                      if p.requires_grad and id(p) not in seen]
            if params:
                groups.append({
                    "name": f"bart_dec_{i}",
                    "params": params,
                    "lr": lr_i,
                    "weight_decay": bart_wd,
                })
                for p in params:
                    seen.add(id(p))

        # Embeddings & lm_head — smallest LR.
        embed_lr = bart_lr * (llrd_gamma ** max_depth)
        embed_modules = [
            self.bart.model.shared,
            self.bart.model.encoder.embed_tokens,
            self.bart.model.encoder.embed_positions,
            self.bart.model.encoder.layernorm_embedding,
            self.bart.model.decoder.embed_tokens,
            self.bart.model.decoder.embed_positions,
            self.bart.model.decoder.layernorm_embedding,
            self.bart.lm_head,
        ]
        embed_params = []
        for m in embed_modules:
            if m is None:
                continue
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    embed_params.append(p)
                    seen.add(id(p))
        if embed_params:
            groups.append({
                "name": "bart_embed",
                "params": embed_params,
                "lr": embed_lr,
                "weight_decay": bart_wd,
            })

        # Catch-all for any trainable BART params not covered above (defensive
        # against future HF BART variants that add new submodules).
        remaining = [p for p in self.bart.parameters()
                     if p.requires_grad and id(p) not in seen]
        if remaining:
            groups.append({
                "name": "bart_other",
                "params": remaining,
                "lr": bart_lr,
                "weight_decay": bart_wd,
            })

        return groups
