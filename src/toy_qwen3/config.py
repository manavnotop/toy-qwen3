"""Configuration and path constants for Toy Qwen3."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelConfig:
    """Configuration for the Qwen3 model architecture.

    Args:
        vocab_size: Number of unique tokens in the vocabulary.
        emb_dim: Token embedding dimension.
        n_heads: Total number of attention heads.
        n_kv_groups: Number of key/value groups for group query attention.
        n_layers: Number of transformer layers.
        hidden_dim: Feedforward hidden dimension.
        context_length: Maximum input sequence length.
        rope_base: Base for rotary positional embeddings.
        qk_norm: Whether to apply RMSNorm to queries and keys.
        dtype: Data type for model weights (bfloat16 or float32).
        head_dim: Head dimension (auto-computed if None).
    """

    vocab_size: int
    emb_dim: int = 128
    n_heads: int = 4
    n_kv_groups: int = 2
    n_layers: int = 4
    hidden_dim: int = 128
    context_length: int = 128
    rope_base: float = 10000.0
    qk_norm: bool = False
    dtype: Any = torch.bfloat16
    head_dim: int | None = None

    def __post_init__(self) -> None:
        """Set head_dim if not specified."""
        if self.head_dim is None:
            self.head_dim = self.emb_dim // self.n_heads


def get_default_config(vocab_size: int, device: str = "cpu") -> dict[str, Any]:
    """Get default configuration dictionary with device-appropriate dtype.

    Args:
        vocab_size: Size of the vocabulary.
        device: Computation device ("cuda", "mps", or "cpu").

    Returns:
        Configuration dictionary compatible with Qwen3Model.
    """
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    return {
        "vocab_size": vocab_size,
        "emb_dim": 128,
        "n_heads": 4,
        "n_kv_groups": 2,
        "n_layers": 4,
        "hidden_dim": 128,
        "context_length": 128,
        "rope_base": 10000.0,
        "qk_norm": False,
        "dtype": dtype,
        "head_dim": None,
    }


# Path constants for data and model files
DATA_PATH = "data/tiny_shakespeare.txt"
VOCAB_PATH = "char_vocab.json"
DEFAULT_MODEL_PATH = "toy_qwen3_24epochs.pth"
