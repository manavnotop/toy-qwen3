"""Qwen3 model implementation for character-level language modeling."""

from typing import Any

import torch
import torch.nn as nn

from .config import Qwen3Config
from .layers import RMSNormalisation, TransformerBlock, compute_rope_params_from_config


class Qwen3Model(nn.Module):
    """Qwen3-style transformer model for character-level language modeling.

    Args:
        cfg: Qwen3Config instance or dict with configuration keys.
    """

    def __init__(self, cfg: Qwen3Config | dict[str, Any]) -> None:
        super().__init__()

        # Support both Qwen3Config and dict for backward compatibility
        if isinstance(cfg, dict):
            self.cfg = Qwen3Config(**cfg)
        else:
            self.cfg = cfg

        cfg = self.cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)

        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(self._to_dict(cfg)) for _ in range(cfg.n_layers)
        )

        self.final_norm = RMSNormalisation(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        cos, sin = compute_rope_params_from_config(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Cache causal mask once during initialization
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    cfg.context_length,
                    cfg.context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

    def _to_dict(self, cfg: Qwen3Config) -> dict[str, Any]:
        """Convert Qwen3Config to dict for TransformerBlock compatibility."""
        return {
            "vocab_size": cfg.vocab_size,
            "emb_dim": cfg.emb_dim,
            "n_heads": cfg.n_heads,
            "n_kv_groups": cfg.n_kv_groups,
            "n_layers": cfg.n_layers,
            "hidden_dim": cfg.hidden_dim,
            "context_length": cfg.context_length,
            "rope_base": cfg.rope_base,
            "qk_norm": False,  # Not used in current implementation
            "dtype": cfg.dtype,
            "head_dim": cfg.head_dim,
        }

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
        logits = self.out_head(x.to(self.cfg.dtype))
        return logits
