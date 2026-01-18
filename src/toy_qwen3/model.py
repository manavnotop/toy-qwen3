"""Qwen3 model implementation for character-level language modeling."""

from typing import Any

import torch
import torch.nn as nn

from .layers import RMSNormalisation, TransformerBlock, compute_rope_params


class Qwen3Model(nn.Module):
    """Qwen3-style transformer model for character-level language modeling.

    Args:
        cfg: Configuration dictionary with keys:
            - vocab_size: Number of unique tokens
            - emb_dim: Token embedding dimension
            - n_layers: Number of transformer layers
            - n_heads: Number of attention heads
            - n_kv_groups: Number of key/value groups for GQA
            - hidden_dim: Feedforward hidden dimension
            - context_length: Maximum sequence length
            - rope_base: RoPE base frequency
            - qk_norm: Whether to apply QK normalization
            - dtype: Data type for weights
            - head_dim: Head dimension (auto-computed if None)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        )

        self.final_norm = RMSNormalisation(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg.get("head_dim") or cfg["emb_dim"] // cfg["n_heads"]

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Cache causal mask once during initialization
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    cfg["context_length"],
                    cfg["context_length"],
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

        self.cfg = cfg

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            in_idx: Input token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        tok_emb = self.tok_emb(in_idx)
        x = tok_emb

        seq_len = x.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.transformer_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
