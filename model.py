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
class TextConditionedImputerXL(nn.Module):
    """
    Inputs:
      x_in: (B,T,A*3) = xy per agent + observed flags
      text_batch: dict with 'input_ids','attention_mask'
      loss_mask: (B,T,A*2) optional (used only for contrastive pooling)
    """
    def __init__(self,
                 num_agents: int,
                 text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 d_model: int = 512,
                 nhead: int = 16,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 use_contrastive: bool = True,
                 proj_dim: int = 256,
                 film_gating: bool = True,
                 freeze_text: bool = False):
        super().__init__()
        self.A = num_agents
        self.in_dim = self.A * 3
        self.out_dim = self.A * 2
        self.use_contrastive = use_contrastive
        self.film_gating = film_gating

        self.frame_embed = nn.Linear(self.in_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        self.text_enc = HFTextEncoder(text_model_name, trainable=not freeze_text)
        self.text_to_d = nn.Linear(self.text_enc.out_dim, d_model)
        if film_gating:
            self.gamma = nn.Linear(d_model, d_model)
            self.beta = nn.Linear(d_model, d_model)

        self.prefix = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dropout=dropout, ffn_mult=4) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

        if use_contrastive:
            self.traj_proj = nn.Linear(d_model, proj_dim)
            self.text_proj = nn.Linear(d_model, proj_dim)
            self.temperature = nn.Parameter(torch.tensor(0.05))

    def forward(self, x_in, text_batch, loss_mask=None):
        B, T, _ = x_in.shape
        h = self.pos(self.frame_embed(x_in))
        tvec = self.text_to_d(self.text_enc(text_batch["input_ids"], text_batch["attention_mask"]))  # (B,d)

        # prefix + optional FiLM
        prefix = self.prefix.expand(B, -1, -1) + tvec.unsqueeze(1)
        if self.film_gating:
            h = h * (1 + self.gamma(tvec).unsqueeze(1)) + self.beta(tvec).unsqueeze(1)

        z = torch.cat([prefix, h], dim=1)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        z_time = z[:, 1:, :]
        pred = self.head(z_time)

        # pooled features for contrastive (mean over masked timesteps; fallback to last)
        pooled = None
        if self.use_contrastive:
            if (loss_mask is not None) and (loss_mask.sum() > 0):
                masked_t = loss_mask.view(B, T, self.A, 2).any(dim=(2, 3)).float()
                w = masked_t / (masked_t.sum(dim=1, keepdim=True) + 1e-8)
                pooled = torch.bmm(w.unsqueeze(1), z_time).squeeze(1)
            else:
                pooled = z_time[:, -1, :]
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
