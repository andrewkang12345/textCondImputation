# experiment.py
# Compare Transformer variants and Diffusion imputer using per-agent, richly-augmented captions.
import os, csv, math, argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter as _TBWriter  # optional

from dataset_text_imputer import TextImputationDataset
from model import (
    TextConditionedImputerXL,
    masked_mse as masked_mse_xl,
    contrastive_loss as contrastive_xl,
    HFTextEncoder, RMSNorm, TransformerBlock
)

# ---------- Metrics ----------
def ade_fde_masked(pred, target, loss_mask, A):
    B, T, _ = pred.shape
    P = pred.view(B, T, A, 2); Y = target.view(B, T, A, 2); M = loss_mask.view(B, T, A, 2)
    err = torch.linalg.norm((P - Y) * M, dim=-1)  # (B,T,A)
    ADE = err.sum() / (M[...,0].sum() + 1e-8)
    # last masked timestep per seq
    mask_any_t = M[...,0].any(dim=2)
    fde_num, fde_den = 0.0, 0.0
    for b in range(B):
        t_idx = None
        for t in range(T-1, -1, -1):
            if mask_any_t[b, t]:
                t_idx = t; break
        if t_idx is None: t_idx = T-1
        e = torch.linalg.norm((P[b, t_idx] - Y[b, t_idx]) * M[b, t_idx], dim=-1).sum().item()
        d = M[b, t_idx, :, 0].sum().item()
        fde_num += e; fde_den += d
    FDE = fde_num / (fde_den + 1e-8)
    return float(ADE.item()), float(FDE)

def collision_rate(pred, A, thresh=1.5, loss_mask=None):
    B, T, _ = pred.shape
    P = pred.view(B, T, A, 2)
    vals = []
    for b in range(B):
        keep_t = torch.ones(T, dtype=torch.bool, device=pred.device)
        if loss_mask is not None:
            keep_t = loss_mask[b].view(T, A, 2).any(dim=(1,2))
        for t in torch.where(keep_t)[0]:
            X = P[b, t]  # (A,2)
            diff = X.unsqueeze(0) - X.unsqueeze(1)  # (A,A,2)
            dist = torch.linalg.norm(diff, dim=-1)
            iu = torch.triu_indices(A, A, offset=1, device=pred.device)
            vals.append((dist[iu[0], iu[1]] < (thresh/94.0 if P.max()<=1.0 else thresh)).float())
    if not vals: return 0.0
    return float(torch.cat(vals).mean().item())

def smoothness_penalty(pred, A):
    V = pred.view(*pred.shape[:2], A, 2).diff(dim=1)
    A2 = V.diff(dim=1)
    return float((A2**2).mean().item())

# ---------- Diffusion (self-contained) ----------
class PosEnc(nn.Module):
    def __init__(self, d, L=4096):
        super().__init__()
        pe = torch.zeros(L, d); pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TimeEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, 4*d); self.l2 = nn.Linear(4*d, d)
        self.act = nn.SiLU()
    def forward(self, t, d):
        device = t.device; half = d//2
        freqs = torch.exp(torch.arange(half, device=device).float() * (-math.log(10000.0)/half))
        ang = t[:, None]*freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if d % 2 == 1: emb = torch.nn.functional.pad(emb,(0,1))
        return self.l2(self.act(self.l1(emb)))

class DiffusionDenoiser(nn.Module):
    def __init__(self, A, text_model, d=512, h=16, L=12, dropout=0.1, film=True, freeze_text=False):
        super().__init__()
        self.A = A; self.in_dim = A*3 + A*2; self.out_dim = A*2; self.d = d
        self.text = HFTextEncoder(text_model, trainable=not freeze_text)
        self.text_proj = nn.Linear(self.text.out_dim, d)
        self.film = film
        if film:
            self.gamma = nn.Linear(d, d); self.beta = nn.Linear(d, d)
        self.time_emb = TimeEmbedding(d); self.frame = nn.Linear(self.in_dim, d); self.pos = PosEnc(d)
        self.prefix = nn.Parameter(torch.zeros(1,1,d))
        self.blocks = nn.ModuleList([TransformerBlock(d, h, dropout=dropout, ffn_mult=4) for _ in range(L)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d, self.out_dim)
    def forward(self, x_in, y_noisy, t_scalar, text):
        B, T, _ = x_in.shape
        txt = self.text_proj(self.text(text["input_ids"], text["attention_mask"]))
        temb = self.time_emb(t_scalar, self.d)
        z = torch.cat([x_in, y_noisy], dim=-1)
        h = self.pos(self.frame(z)); h = h + temb.unsqueeze(1)
        prefix = self.prefix.expand(B,-1,-1) + txt.unsqueeze(1)
        if self.film: h = h * (1 + self.gamma(txt).unsqueeze(1)) + self.beta(txt).unsqueeze(1)
        zcat = torch.cat([prefix, h], dim=1)
        for blk in self.blocks: zcat = blk(zcat)
        zcat = self.norm(zcat)
        return self.head(zcat[:,1:,:])

class DiffusionImputer:
    def __init__(self, A, text_model, d=512, h=16, L=12, dropout=0.1, film=True, freeze_text=False, steps=200, device="cuda"):
        self.A=A; self.device=device
        self.net = DiffusionDenoiser(A, text_model, d, h, L, dropout, film, freeze_text).to(device)
        self.steps = steps; self.ac = self._cosine_schedule(steps).to(device)
        self.sqrt_ac = torch.sqrt(self.ac); self.sqrt_om = torch.sqrt(1 - self.ac)
    def _cosine_schedule(self, S):
        s=0.008; t=torch.linspace(0,1,S+1); f=torch.cos((t+s)/(1+s)*math.pi/2)**2; a=f[1:]/f[:-1]; return a.cumprod(0)
    def training_step(self, batch, opt, scaler=None):
        x_in = batch["x_in"].to(self.device); y0 = batch["y_gt"].to(self.device); m = batch["loss_mask"].to(self.device)
        text = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
        B = x_in.size(0); t_idx = torch.randint(0, self.steps, (B,), device=self.device)
        sqrt_ac = self.sqrt_ac[t_idx].view(B,1,1); sqrt_om = self.sqrt_om[t_idx].view(B,1,1)
        eps = torch.randn_like(y0)
        y_noisy = y0 * (1 - m) + (sqrt_ac * y0 + sqrt_om * eps) * m
        t_scalar = (t_idx.float()+0.5)/float(self.steps)
        with torch.autocast(device_type="cuda", enabled=(self.device=="cuda")):
            eps_hat = self.net(x_in, y_noisy, t_scalar, text)
            loss = ((eps_hat - eps)**2 * m).sum() / (m.sum()+1e-8)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
        return loss.item()
    @torch.no_grad()
    def sample(self, x_in, text, m, ddim_steps=60):
        device=self.device; B,T,_=x_in.shape; y=torch.randn(B,T,self.A*2,device=device)*m
        ts=torch.linspace(self.steps-1,0,steps=ddim_steps,dtype=torch.long,device=device)
        for i,t in enumerate(ts):
            t = t.long(); t_scalar = (t.float()+0.5)/float(self.steps)
            s_ac=torch.sqrt(self.ac[t]).view(1,1,1); s_om=torch.sqrt(1-self.ac[t]).view(1,1,1)
            x_t = s_ac * y
            eps_hat = self.net(x_in, x_t, t_scalar.repeat(B), text)
            x0 = (x_t - s_om * eps_hat) / (s_ac + 1e-8)
            if i == len(ts)-1:
                y = x0; break
            t_next = ts[i+1].long()
            s_ac_n = torch.sqrt(self.ac[t_next]).view(1,1,1); s_om_n = torch.sqrt(1-self.ac[t_next]).view(1,1,1)
            y = s_ac_n * x0 + s_om_n * eps_hat
            y = y * m
        return y

# ---------- Experiments ----------
@dataclass
class Exp:
    name: str; arch: str
    text_model: str = "sentence-transformers/all-mpnet-base-v2"
    d: int = 512; h: int = 16; L: int = 12; dropout: float = 0.1
    film: bool = True; freeze_text: bool = False
    w_contra: float = 0.1; proj_dim: int = 256
    epochs: int = 30; batch: int = 64
    steps_diff: int = 200; ddim_steps: int = 60
    lr: float = 2e-4; wd: float = 1e-4

def make_runs(default: dict) -> List[Exp]:
    return [
        Exp(**{**default, "name":"xl_baseline", "arch":"xl"}),
        Exp(**{**default, "name":"xl_nofilm", "arch":"xl", "film":False}),
        Exp(**{**default, "name":"xl_freezetext", "arch":"xl", "freeze_text":True}),
        Exp(**{**default, "name":"xl_no_contra", "arch":"xl", "w_contra":0.0}),
        Exp(**{**default, "name":"xl_large", "arch":"xl", "d":768, "h":24, "L":16, "proj_dim":384}),
        Exp(**{**default, "name":"diffusion_mid", "arch":"diff", "w_contra":0.0, "steps_diff":400, "ddim_steps":60}),
        Exp(**{**default, "name":"diffusion_large", "arch":"diff", "d":640, "h":20, "L":16, "w_contra":0.0, "steps_diff":600, "ddim_steps":80}),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_traj", default="/mnt/data/mywork/nbaWork/prev/UniTraj-pytorch/Sports-Traj/datasets/pre-processed/basketball/train_clean.p")
    ap.add_argument("--val_traj",   default="/mnt/data/mywork/nbaWork/prev/UniTraj-pytorch/Sports-Traj/datasets/pre-processed/basketball/test_clean.p")
    ap.add_argument("--out_dir", default='outputs')
    ap.add_argument("--text_model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # datasets (per-agent; captions stochastically augmented for train, deterministic for val)
    train_ds = TextImputationDataset(args.train_traj,
                                     deterministic_mask=False,
                                     deterministic_caption=False,
                                     per_agent=True)
    val_ds   = TextImputationDataset(args.val_traj,
                                     deterministic_mask=True,
                                     deterministic_caption=True,
                                     mirror_prob=0.0,
                                     synonym_prob=0.0,
                                     per_agent=True)
    A, T = train_ds.A, train_ds.T
    print(f"Train {len(train_ds)} Val {len(val_ds)} A={A} T={T}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    def collate(batch):
        x = torch.stack([b["x_in"] for b in batch]); y = torch.stack([b["y_gt"] for b in batch]); m = torch.stack([b["loss_mask"] for b in batch])
        caps = [b["caption"] for b in batch]
        toks = tokenizer(caps, padding=True, truncation=True, max_length=128, return_tensors="pt")
        out = {"x_in":x, "y_gt":y, "loss_mask":m, "input_ids":toks["input_ids"], "attention_mask":toks["attention_mask"]}
        out["target_agent"] = torch.tensor([b["target_agent"] for b in batch], dtype=torch.long)
        return out

    default = dict(name="", arch="xl", text_model=args.text_model, d=512, h=16, L=12, dropout=0.1, film=True, freeze_text=False,
                   w_contra=0.1, proj_dim=256, epochs=30, batch=64, steps_diff=200, ddim_steps=60, lr=2e-4, wd=1e-4)
    runs = make_runs(default)

    # runs = [r for r in runs if r.name == "xl_baseline"] # for specified runs
    
    if args.epochs is not None:
        for r in runs: r.epochs = args.epochs

    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["name","arch","params(M)","best_val","ADE","FDE","Coll","Smooth"])

    # system perf tweaks
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    for cfg in runs:
        tqdm.write(f"\n=== {cfg.name} ({cfg.arch}) ===")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

        run_dir = os.path.join(args.out_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        try:
            writer = _TBWriter(log_dir=run_dir)
        except Exception:
            writer = None

        if cfg.arch == "xl":
            model = TextConditionedImputerXL(
                A, args.text_model, cfg.d, cfg.h, cfg.L, cfg.dropout,
                use_contrastive=(cfg.w_contra>0.0), proj_dim=cfg.proj_dim,
                film_gating=cfg.film, freeze_text=cfg.freeze_text
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)  # use all GPUs
            device = next(model.parameters()).device

            opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9,0.95))
            # GradScaler (PT2+ fallback to PT1.x)
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
            except TypeError:
                scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

            def step(loader, train, desc):
                base_model = model
                base_model.train(mode=train)
                tot = 0.0
                with tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) as it:
                    for b in it:
                        x=b["x_in"].to(device); y=b["y_gt"].to(device); m=b["loss_mask"].to(device)
                        text={"input_ids":b["input_ids"].to(device),"attention_mask":b["attention_mask"].to(device)}
                        with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                            p, tv, pooled = base_model(x, text, m); loss = masked_mse_xl(p, y, m)
                            if getattr(base_model, "module", None) is not None:
                                use_contra = base_model.module.use_contrastive
                                temperature = base_model.module.temperature
                                traj_proj = base_model.module.traj_proj
                                text_proj = base_model.module.text_proj
                            else:
                                use_contra = base_model.use_contrastive
                                temperature = base_model.temperature
                                traj_proj = base_model.traj_proj
                                text_proj = base_model.text_proj
                            if use_contrastive := use_contra and cfg.w_contra>0.0:
                                zt = traj_proj(pooled); zl = text_proj(tv)
                                loss = loss + cfg.w_contra * contrastive_xl(zt, zl, temperature)
                        if train:
                            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                            nn.utils.clip_grad_norm_(base_model.parameters(), 1.0); scaler.step(opt); scaler.update()
                        tot += loss.item()
                        it.set_postfix(avg_loss=f"{tot / max(1, it.n):.4f}")
                return tot / max(1, len(loader))

            best = float("inf")
            for ep in trange(1, cfg.epochs+1, desc=f"{cfg.name}:epochs", dynamic_ncols=True):
                tr_loss = step(train_loader, True, desc=f"{cfg.name}:train")
                va_loss = step(val_loader,   False, desc=f"{cfg.name}:val")
                if writer:
                    writer.add_scalar("loss/train", tr_loss, ep)
                    writer.add_scalar("loss/val", va_loss, ep)
                if va_loss < best:
                    best = va_loss
                tqdm.write(f"ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

                # checkpoints
                base = model.module if isinstance(model, nn.DataParallel) else model
                state = {"model": base.state_dict(), "cfg": cfg.__dict__, "epoch": ep, "val_loss": va_loss}
                torch.save(state, os.path.join(run_dir, "last.pt"))
                if va_loss <= best + 1e-12:
                    torch.save(state, os.path.join(run_dir, "best.pt"))

            # metrics
            model.eval(); allP=[]; allY=[]; allM=[]
            with torch.no_grad():
                for b in tqdm(val_loader, desc=f"{cfg.name}:metrics", leave=False, dynamic_ncols=True):
                    x=b["x_in"].to(device); y=b["y_gt"].to(device); m=b["loss_mask"].to(device)
                    text={"input_ids":b["input_ids"].to(device),"attention_mask":b["attention_mask"].to(device)}
                    p,_,_ = model(x, text, m); allP.append(p.cpu()); allY.append(y.cpu()); allM.append(m.cpu())
            P=torch.cat(allP); Y=torch.cat(allY); M=torch.cat(allM)
            ade,fde = ade_fde_masked(P,Y,M,A); coll=collision_rate(P,A,1.5,M); sm=smoothness_penalty(P,A)
            base = model.module if isinstance(model, nn.DataParallel) else model
            params = sum(p.numel() for p in base.parameters())/1e6

        else:  # diffusion
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dif = DiffusionImputer(A, args.text_model, cfg.d, cfg.h, cfg.L, cfg.dropout,
                                   film=cfg.film, freeze_text=cfg.freeze_text,
                                   steps=cfg.steps_diff, device=device.type)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                dif.net = nn.DataParallel(dif.net).to(device)

            opt = optim.AdamW(dif.net.parameters(), lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9,0.95))
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
            except TypeError:
                scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

            best=float("inf")
            for ep in trange(1, cfg.epochs+1, desc=f"{cfg.name}:epochs", dynamic_ncols=True):
                tot, steps = 0.0, 0
                with tqdm(train_loader, desc=f"{cfg.name}:train(diff)", leave=False, dynamic_ncols=True) as it:
                    for b in it:
                        loss = dif.training_step(b, opt, scaler); tot+=loss; steps+=1
                        it.set_postfix(avg_loss=f"{tot/max(1,steps):.4f}")
                avg = tot/max(1,steps)
                if writer: writer.add_scalar("loss/train_diff", avg, ep)
                if avg < best: best = avg

                # save checkpoints
                base = dif.net.module if isinstance(dif.net, nn.DataParallel) else dif.net
                state = {"model": base.state_dict(), "cfg": cfg.__dict__, "epoch": ep, "best_train": best}
                torch.save(state, os.path.join(run_dir, "last.pt"))
                if avg <= best + 1e-12:
                    torch.save(state, os.path.join(run_dir, "best.pt"))
                tqdm.write(f"ep {ep:03d} | train {avg:.4f} | best {best:.4f}")

            # sampling eval
            allP=[]; allY=[]; allM=[]
            dif.net.eval()
            with torch.no_grad():
                for b in tqdm(val_loader, desc=f"{cfg.name}:sample(diff)", leave=False, dynamic_ncols=True):
                    x=b["x_in"].to(device); y=b["y_gt"].to(device); m=b["loss_mask"].to(device)
                    text={"input_ids":b["input_ids"].to(device),"attention_mask":b["attention_mask"].to(device)}
                    p = dif.sample(x, text, m, ddim_steps=cfg.ddim_steps); allP.append(p.cpu()); allY.append(y.cpu()); allM.append(m.cpu())
            P=torch.cat(allP); Y=torch.cat(allY); M=torch.cat(allM)
            ade,fde = ade_fde_masked(P,Y,M,A); coll=collision_rate(P,A,1.5,M); sm=smoothness_penalty(P,A)
            base = dif.net.module if isinstance(dif.net, nn.DataParallel) else dif.net
            params = sum(p.numel() for p in base.parameters())/1e6

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([cfg.name, cfg.arch, f"{params:.2f}", f"{best:.6f}", f"{ade:.4f}", f"{fde:.4f}", f"{coll:.4f}", f"{sm:.6f}"])
        tqdm.write(f"✓ {cfg.name}: ADE {ade:.4f} FDE {fde:.4f} Coll {coll:.4f} Smooth {sm:.6f}")

        try:
            writer and writer.close()
        except Exception:
            pass

    print(f"\nAll done. Results at {csv_path}")

if __name__ == "__main__":
    main()
