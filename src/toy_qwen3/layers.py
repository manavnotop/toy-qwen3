"""Transformer layer components for Qwen3 architecture."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Args:
        cfg: Configuration dictionary with 'emb_dim', 'hidden_dim', and 'dtype'.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, emb_dim)
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        x_fc3 = self.fc3(x)
        return x_fc3


class RMSNormalisation(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization).

    Args:
        emb_dim: Embedding dimension.
        eps: Epsilon for numerical stability.
        bias: Whether to include a shift parameter.
        qwen3_compatible: Use Qwen3-compatible upcast to float32.
    """

    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
        qwen3_compatible: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape

        Returns:
            Normalized tensor of same shape as input
        """
        input_type = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(mean_square + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_type)


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary positional embedding parameters.

    Args:
        head_dim: Head dimension (must be even).
        theta_base: Base for the rotary angle computation.
        context_length: Maximum context length.
        dtype: Data type for the output tensors.

    Returns:
        Tuple of (cos, sin) tensors of shape (context_length, head_dim)
    """
    if head_dim % 2 != 0:
        msg = "Embedding dimension must be even"
        raise ValueError(msg)

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    position = torch.arange(context_length, dtype=dtype)
    angles = position[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to query/key tensors.

    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        cos: Cosine values of shape (seq_len, head_dim).
        sin: Sine values of shape (seq_len, head_dim).

    Returns:
        Transformed tensor of same shape as input.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        msg = "Head dimensions must be even"
        raise ValueError(msg)

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupQueryAttention(nn.Module):
    """Group Query Attention (GQA) layer.

    Args:
        d_in: Input dimension.
        num_heads: Number of query heads.
        num_kv_groups: Number of key/value groups (must divide num_heads).
        head_dim: Head dimension (auto-computed if None).
        qk_norm: Whether to apply RMSNorm to queries and keys.
        dtype: Data type for the linear projections.
    """

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            msg = "num_heads must be divisible by num_kv_groups"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if d_in % num_heads != 0:
                msg = "d_in must be divisible by num_heads if head_dim is not set"
                raise ValueError(msg)
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNormalisation(head_dim, eps=1e-6)
            self.k_norm = RMSNormalisation(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply group query attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in).
            mask: Causal mask of shape (seq_len, seq_len).
            cos: RoPE cos values.
            sin: RoPE sin values.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_in).
        """
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture.

    Args:
        cfg: Configuration dictionary for the model.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.attn = GroupQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNormalisation(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNormalisation(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transformer block with residual connections.

        Args:
            x: Input tensor.
            mask: Causal attention mask.
            cos: RoPE cos values.
            sin: RoPE sin values.

        Returns:
            Transformed tensor of same shape as input.
        """
        original_x = x
        x = self.norm1(x)
        x = self.attn(x, mask, cos, sin)
        x = x + original_x

        original_x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + original_x

        return x
