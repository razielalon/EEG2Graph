"""
EEG-to-Graph: Generative Model
================================

Encoder-Decoder Transformer that translates EEG word-level features
into linearized knowledge graph triplets.

    Encoder: EEG (batch, src_len, 840) → hidden (batch, src_len, d_model)
    Decoder: Autoregressive, cross-attends to encoder output,
             generates <bos> <triplet> subj <subj> rel <rel> obj <obj> ... <eos>

Architecture:

    ┌──────────────┐        ┌──────────────────┐
    │  EEG Input   │        │  Target Tokens   │
    │ (B, L, 840)  │        │  (shifted right)  │
    └──────┬───────┘        └────────┬─────────┘
           │                         │
    ┌──────▼───────┐        ┌────────▼─────────┐
    │  Projection  │        │ Token Embedding  │
    │  + PosEmbed  │        │   + PosEmbed     │
    └──────┬───────┘        └────────┬─────────┘
           │                         │
    ┌──────▼───────┐        ┌────────▼─────────┐
    │  Transformer │ ──────►│   Transformer    │
    │   Encoder    │ cross  │     Decoder      │
    │  (N layers)  │ attn   │   (N layers)     │
    └──────────────┘        └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │   Output Linear  │
                            │   → vocab logits │
                            └──────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabulary import PAD_ID, BOS_ID, EOS_ID


class EEGGraphModel(nn.Module):
    """
    Encoder-Decoder Transformer for EEG → triplet generation.

    Args:
        vocab_size: Decoder vocabulary size
        eeg_dim: Input EEG feature dim (840 for ZuCo)
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_enc_layers: Encoder Transformer layers
        n_dec_layers: Decoder Transformer layers
        max_src_len: Max encoder sequence length
        max_tgt_len: Max decoder sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size,
        eeg_dim=840,
        d_model=256,
        n_heads=8,
        n_enc_layers=4,
        n_dec_layers=4,
        max_src_len=128,
        max_tgt_len=128,
        dropout=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # ---- Encoder ----
        self.enc_proj = nn.Sequential(
            nn.Linear(eeg_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.enc_pos = nn.Embedding(max_src_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # ---- Decoder ----
        self.dec_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.dec_pos = nn.Embedding(max_tgt_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # ---- Output head ----
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Weight tying: share decoder embedding and output projection
        self.output_proj.weight = self.dec_embed.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        # Re-init embeddings
        nn.init.normal_(self.dec_embed.weight, std=0.02)
        nn.init.zeros_(self.dec_embed.weight[PAD_ID])
        nn.init.normal_(self.enc_pos.weight, std=0.02)
        nn.init.normal_(self.dec_pos.weight, std=0.02)

    # ---- Encoder ----

    def encode(self, src, src_mask=None):
        """
        Encode EEG features.

        Args:
            src: (B, S, eeg_dim)
            src_mask: (B, S) — True for real tokens

        Returns:
            memory: (B, S, d_model)
            memory_key_padding_mask: (B, S) — True for pad positions
        """
        B, S, _ = src.shape
        h = self.enc_proj(src)
        positions = torch.arange(S, device=src.device).unsqueeze(0)
        h = h + self.enc_pos(positions)
        h = self.dropout(h)

        pad_mask = ~src_mask if src_mask is not None else None
        memory = self.encoder(h, src_key_padding_mask=pad_mask)

        return memory, pad_mask

    # ---- Decoder ----

    @staticmethod
    def _causal_mask(sz, device):
        """Generate upper-triangular causal mask."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def decode(self, tgt_ids, memory, memory_key_padding_mask=None):
        """
        Decode target token sequence with cross-attention to encoder memory.

        Args:
            tgt_ids: (B, T) token IDs
            memory: (B, S, d_model) encoder output
            memory_key_padding_mask: (B, S)

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tgt_ids.shape
        h = self.dec_embed(tgt_ids)
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        h = h + self.dec_pos(positions)
        h = self.dropout(h)

        causal = self._causal_mask(T, tgt_ids.device)
        tgt_pad = (tgt_ids == PAD_ID)

        h = self.decoder(
            h, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.output_proj(h)

    # ---- Forward (training) ----

    def forward(self, src, src_mask, tgt):
        """
        Full forward pass for training with teacher forcing.

        Args:
            src: (B, S, eeg_dim) — EEG features
            src_mask: (B, S) — True for real tokens
            tgt: (B, T) — decoder input token IDs (shifted right)

        Returns:
            logits: (B, T, vocab_size)
        """
        memory, mem_pad = self.encode(src, src_mask)
        logits = self.decode(tgt, memory, mem_pad)
        return logits

    # ---- Greedy Inference ----

    @torch.no_grad()
    def generate(self, src, src_mask, max_len=128, temperature=0.0):
        """
        Autoregressive decoding (greedy by default, sampling with temperature > 0).

        Args:
            src: (B, S, eeg_dim)
            src_mask: (B, S)
            max_len: Maximum tokens to generate
            temperature: 0.0 for greedy argmax, >0 for sampling

        Returns:
            generated: (B, max_len) token ID tensor
        """
        B = src.size(0)
        device = src.device

        memory, mem_pad = self.encode(src, src_mask)

        # Start with <bos>
        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decode(generated, memory, mem_pad)  # (B, T, V)
            next_logits = logits[:, -1, :]  # (B, V)

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced <eos>
            if (next_token.squeeze(-1) == EOS_ID).all():
                break

        return generated

    # ---- Beam Search ----

    @torch.no_grad()
    def beam_search(self, src, src_mask, beam_size=4, max_len=128):
        """
        Beam search decoding (single sample, B=1).

        Returns:
            best_seq: (seq_len,) token ID tensor — best hypothesis
        """
        assert src.size(0) == 1, "Beam search operates on single samples"
        device = src.device

        memory, mem_pad = self.encode(src, src_mask)

        # Each beam: (log_prob, token_ids list)
        beams = [(0.0, [BOS_ID])]
        completed = []

        for _ in range(max_len):
            candidates = []

            for score, seq in beams:
                if seq[-1] == EOS_ID:
                    completed.append((score, seq))
                    continue

                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.decode(tgt, memory, mem_pad)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)

                topk_lp, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((score + lp, seq + [tid]))

            if not candidates:
                break

            # Keep top-k beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            # Early stop if top beam ended (already in completed from the loop above)
            if beams[0][1][-1] == EOS_ID:
                break

        if completed:
            completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # length-normalized
            return torch.tensor(completed[0][1], device=device)
        else:
            return torch.tensor(beams[0][1], device=device)