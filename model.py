# model.py
import math
import torch
import torch.nn as nn
from transformers import AutoModel

# ---------- Norms/blocks ----------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model * 2),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model * 2, d_model),
        )
        self.n1, self.n2 = RMSNorm(d_model), RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.n2(x)
        x = x + self.drop(self.ffn(h))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# ---------- Text encoder ----------
class HFTextEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", trainable: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.out_dim = self.encoder.config.hidden_size
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

# ---------- Main imputer ----------
class VectorConditionedImputerXL(nn.Module):
    """
    Same transformer, but conditioning comes from extra channels in x_in:
      x_in: (B, T, A*2 + A + R)   # xy + obs + region one-hot (masked agent)
      loss_mask: (B, T, A*2)
    """
    def __init__(self,
                 num_agents: int,
                 region_dim: int,              # R
                 d_model: int = 512,
                 nhead: int = 16,
                 num_layers: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.A = num_agents
        self.R = region_dim
        self.in_dim = self.A * 3 + self.R     # xy (A*2) + obs (A) + region (R)
        self.out_dim = self.A * 2

        self.frame_embed = nn.Linear(self.in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dropout=dropout, ffn_mult=4)
                                     for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x_in, loss_mask=None):
        # x_in: (B,T,in_dim)
        h = self.pos(self.frame_embed(x_in))
        for blk in self.blocks:
            h = blk(h)
        z = self.norm(h)            # (B,T,D)
        pred = self.head(z)         # (B,T,A*2)
        # For API compatibility with training loop, also return dummy tvec/pooled
        tvec = None
        pooled = z[:, -1, :]        # (B,D)
        return pred, tvec, pooled

# ---------- Losses ----------
def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den

def contrastive_loss(z_traj, z_text, temperature):
    zt = nn.functional.normalize(z_traj, dim=1)
    zl = nn.functional.normalize(z_text, dim=1)
    logits = zt @ zl.t() / temperature.clamp_min(1e-6)
    labels = torch.arange(zt.size(0), device=zt.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels) +
                  nn.functional.cross_entropy(logits.t(), labels))
