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