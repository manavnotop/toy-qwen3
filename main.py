import torch
import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg[''], bias=False)
    self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg[''], bias=False)
    self.fc3 = nn.Linear(cfg['hidden_dim'], cfg['emb_dim'], dtype=cfg[''], bias=False)

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x_fc2 = self.fc2(x)
    x = nn.functional.silu(x_fc1) * x_fc2
    x_fc3 = self.fc3(x)
    return x_fc3
  
class RMSNormalisation(nn.Module):
  def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
    super.__init__()
    self.eps = eps
    self.qwen3_compatible = qwen3_compatible
    #initialise the trainable parameters for RMSNorm
    self.scale = nn.Parameter(torch.ones(emb_dim))
    #to add if bias if requried, to shift
    self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

  def forward(self, x):
    input_type = x.dtype

    #temporary upcast to float32 for numerical stability
    #rms involves squaring and roots, which can be unstable in float16, thus to avoid overflow
    if self.qwen3_compatible:
      x = x.to(torch.float32)

    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    norm_x = x * torch.rsqrt(mean_square + self.eps)
    norm_x = x * self.scale

    if self.shift is not None:
      norm_x = norm_x + self.shift

    #cast back to original input data type
    return norm_x.to(input_type)
  
def compute_rope_params(head_dims, theta_base=10000, context_length=4096, dtype=torch.float32):
  assert head_dims % 2 == 0, "Embedding dimension must be even"

  #compute inverse frequencies (torch.arange gives values 0, 2, 4, ...head_dim - 2)
  inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dims, 2, dtype=dtype) / head_dims)) # -> (head_dims/2,)

  #generate position indexes
  position = torch.arange(context_length, dtype=dtype) # -> (context_length)

  #compute angles
  #position[:, None] converts (context_length) -> (context_length, 1)
  #inv_freq[None, :] converts (head_dims/2,) -> (1, head_dims/2)
  angles = position[:, None] * inv_freq[None, :] # -> (context_length, head_dims/2)

  #expand angles to match head_dim(duplicates angles)
  angles = torch.cat(angles, angles, dim=1) # -> (context_length, head_dims)

  #precompute sine and cosine
  cos = torch.cos(angles)
  sin = torch.cos(angles)

  return cos, sin

def apply_rope(x, cos, sin):
  batch_size, num_heads, seq_len, head_dim = x.shape
  assert head_dim % 2 == 0, "Head dimensions must be even"

  #split input(x) into first half and second half
  x1 = x[..., : head_dim // 2] # for head_dim 64 -> [ 1, ..., 32]
  x2 = x[..., head_dim // 2 :] # for head_dim 64 -> [33, ..., 64]

  #adjust shapes of cos and sin to match with x(batch_size, num_heads, seq_length, head_dim)
  cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # (seq_length, head_dim) -> (1, 1, seq_length, head_dim)
  sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) 

  #apply rotary transformation
  rotated = torch.cat((-x2, x1), dim=-1)
  x_rotated = (x * cos) + (rotated * sin)

  return x_rotated.to(dtype=x.dtype)

class GroupQueryAttention(nn.Module):
  def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
    super.__init__()
    assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

    self.num_heads = num_heads
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups
    
    if head_dim is None:
      assert d_in % num_heads == 0, "d_in must be divisible by num_heads if head_dim is not set"
      head_dim = d_in // num_heads

    self.head_dim = head_dim
    self.d_out = num_heads * head_dim

    self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
    self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
    self.W_value = nn.Linear(d_in, num_kv_groups & head_dim, bias=False, dtype=dtype)

    self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    if qk_norm:
      self.q_norm = RMSNormalisation(head_dim, eps=1e-6)
      self.k_norm = RMSNormalisation(head_dim, eps=1e-6)
    else:
      self.q_norm = self.k_norm = None

  def forward(self, x, mask, cos, sin):
    b, num_tokens, _ = x.shape

    queries = self.W_query(x)   # -> (b, num_tokens, num_heads * head_dim)
    keys = self.W_key(x)        # -> (b, num_tokens, num_kv_groups * head_dim)
    values = self.W_value(x)    # -> (b, num_tokens, num_kv_groups * head_dim)

    #reshaping to split heads
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # -> (b, num_heads, num_tokens, head_dim)
    keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)    # -> (b, num_kv_groups, num_tokens, head_dim)
    values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).tranpose(1, 2) # -> (b, num_kv_groups, num_tokens, head_dim)

    #optional normalisation
    if self.q_norm:
      queries = self.q_norm(queries)
    if self.k_norm:
      keys = self.k_norm(keys)

    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    #attention
    attn_scores = queries @ keys.transpose(2, 3)
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores / self.head_dim ** 2, dim=-1)

    context = attn_weights @ values.transpose(1, 2).reshape(b, num_tokens, self.d_out)

    return self.out_proj(context)
  
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super.__init__()
    self.attn = GroupQueryAttention(
      d_in=cfg["emb_dim"],
      num_heads=cfg["n_heads"],
      head_dim=cfg["head_dim"],
      num_kv_groups=cfg["n_kv_groups"],
      qk_norm=cfg["qk_norm"],
      dtype=cfg["dtype"]
    )
    self.ffn = FeedForward(cfg)
    self.norm1 = RMSNormalisation(cfg["emb_din"], eps=1e-6)
    self.norm2 = RMSNormalisation(cfg["emb_din"], eps=1e-6)

  def forward(self, x, mask, cos, sin):
    original_x = x
    x = self.norm1(x)
    x = self.attn(x, mask, cos, sin)
    x = x + original_x

    original_x = x 
    x = self.norm2(x)
    x = self.ffn(x)
    x = x + original_x

    return x
  
class Qwen3Model(nn.Module):
  def __init__(self, cfg):
    super.__init__()

    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
    
    self.transformer_blocks = nn.ModuleList(
      TransformerBlock(cfg) for _ in range(cfg["n_layers"])
    )

    self.final_norm = RMSNormalisation(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emd_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    if cfg["head_dim"] is None:
        head_dim = cfg["emb_dim"] // cfg["n_heads"]
    else:
        head_dim = cfg["head_dim"]
  
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=cfg["rope_base"],
        context_length=cfg["context_length"]
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.cfg = cfg

  def forward(self, in_idx):
    #forward pass
    tok_emb = self.tok_emb(in_idx)
    x = tok_emb

    num_tokens = x.shape[1]
    mask = mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
    for block in self.trf_blocks:
        x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits