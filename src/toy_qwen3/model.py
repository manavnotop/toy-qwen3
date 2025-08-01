import torch
from torch import nn
from .layers import TransformerBlock, RMSNormalisation, compute_rope_params
  
class Qwen3Model(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
    
    self.transformer_blocks = nn.ModuleList(
      TransformerBlock(cfg) for _ in range(cfg["n_layers"])
    )

    self.final_norm = RMSNormalisation(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

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
    mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
    for block in self.transformer_blocks:
        x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits