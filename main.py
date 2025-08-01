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

