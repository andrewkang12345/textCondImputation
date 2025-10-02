# inference.py (team-colored, grid-conditioned, with OpenAI text->grid ids)
# Vector-conditioned single-agent imputation for XL/Diffusion + GIF viz over court.png
# Conditioning comes from a per-timestep 8x5 grid one-hot (R=40) appended to x_in.
#
# Region sources (priority):
#   (1) --region_ids_json  -> JSON list of integers in [0, R_BINS-1], len T
#   (2) --prompt           -> natural language, mapped to per-timestep grid ids via OpenAI
#   (3) ORACLE             -> derived from GT masked agent (for testing)

import os, argparse, pickle, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import math

from model import VectorConditionedImputerXL, RMSNorm, TransformerBlock
from dataset_text_imputer import _normalize_xy, _agent_region_ids_grid, GRID_W, GRID_H, R_BINS

# ---------- metrics ----------
def ade_fde_masked(pred, target, loss_mask, A):
    B, T, _ = pred.shape
    P = pred.view(B, T, A, 2); Y = target.view(B, T, A, 2); M = loss_mask.view(B, T, A, 2)
    err = torch.linalg.norm((P - Y) * M, dim=-1)
    ADE = err.sum() / (M[...,0].sum() + 1e-8)
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
            X = P[b, t]
            diff = X.unsqueeze(0) - X.unsqueeze(1)
            dist = torch.linalg.norm(diff, dim=-1)
            iu = torch.triu_indices(A, A, offset=1, device=pred.device)
            vals.append((dist[iu[0], iu[1]] < (thresh/94.0 if P.max()<=1.0 else thresh)).float())
    if not vals: return 0.0
    return float(torch.cat(vals).mean().item())

def smoothness_penalty(pred, A):
    V = pred.view(*pred.shape[:2], A, 2).diff(dim=1)
    A2 = V.diff(dim=1)
    return float((A2**2).mean().item())

# ---------- minimal diffusion (sampling only, vector-conditioned) ----------
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
        super().__init__(); self.l1=nn.Linear(d,4*d); self.l2=nn.Linear(4*d,d); self.act=nn.SiLU()
    def forward(self, t, d):
        device=t.device; half=d//2
        freqs=torch.exp(torch.arange(half, device=device).float()*(-math.log(10000.0)/half))
        ang=t[:,None]*freqs[None,:]
        emb=torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if d%2==1: emb=torch.nn.functional.pad(emb,(0,1))
        return self.l2(self.act(self.l1(emb)))

class DiffusionDenoiserVec(nn.Module):
    def __init__(self, A, R, d=512, h=16, L=12, dropout=0.1):
        super().__init__()
        self.A=A; self.R=R; self.d=d
        self.in_dim = (A*3 + R) + (A*2)   # concat([x_in, y_t])
        self.out_dim = A*2
        self.time_emb=TimeEmbedding(d); self.frame=nn.Linear(self.in_dim,d); self.pos=PosEnc(d)
        self.blocks=nn.ModuleList([TransformerBlock(d,h,dropout=dropout,ffn_mult=4) for _ in range(L)])
        self.norm=RMSNorm(d); self.head=nn.Linear(d,self.out_dim)
    def forward(self, x_in, y_noisy, t_scalar):
        B,T,_=x_in.shape
        temb=self.time_emb(t_scalar, self.d)
        z=torch.cat([x_in, y_noisy], dim=-1)
        h=self.pos(self.frame(z)); h=h+temb.unsqueeze(1)
        for blk in self.blocks: h=blk(h)
        h=self.norm(h)
        return self.head(h)

class DiffusionImputerVec:
    def __init__(self, A, R, d=512, h=16, L=12, dropout=0.1, steps=200, device="cuda"):
        self.A=A; self.R=R; self.device=device
        self.net=DiffusionDenoiserVec(A,R,d,h,L,dropout).to(device)
        self.steps=steps; self.ac=self._cosine(steps).to(device)
    def _cosine(self,S):
        s=0.008; t=torch.linspace(0,1,S+1); f=torch.cos((t+s)/(1+s)*math.pi/2)**2; a=f[1:]/f[:-1]; return a.cumprod(0)
    @torch.no_grad()
    def sample(self, x_in, m, ddim_steps=60):
        device=self.device; B,T,_=x_in.shape; y=torch.randn(B,T,self.A*2,device=device)*m
        ts=torch.linspace(self.steps-1,0,steps=ddim_steps,dtype=torch.long,device=device)
        for i,t in enumerate(ts):
            t=t.long(); t_scalar=(t.float()+0.5)/float(self.steps)
            s_ac=torch.sqrt(self.ac[t]).view(1,1,1); s_om=torch.sqrt(1-self.ac[t]).view(1,1,1)
            x_t=s_ac*y
            eps_hat=self.net(x_in, x_t, t_scalar.repeat(B))
            x0=(x_t - s_om*eps_hat)/(s_ac+1e-8)
            if i==len(ts)-1: y=x0; break
            t_next=ts[i+1].long()
            s_ac_n=torch.sqrt(self.ac[t_next]).view(1,1,1); s_om_n=torch.sqrt(1-self.ac[t_next]).view(1,1,1)
            y=s_ac_n*x0 + s_om_n*eps_hat
            y=y*m
        return y

# ---------- io / prep ----------
def _load_traj(path, seq_idx=0):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        z = np.load(path); key = "traj" if "traj" in z.files else z.files[0]; arr = z[key]
    elif ext in [".pkl", ".pickle", ".p"]:
        with open(path, "rb") as f: obj = pickle.load(f)
        arr = obj[seq_idx] if hasattr(obj, "ndim") and obj.ndim == 4 else obj
    else:
        raise ValueError(f"Unsupported traj file: {path}")
    assert arr.ndim == 3 and arr.shape[-1] == 2
    return arr.astype(np.float32)

def _make_mask(T, A, masked_agent, mask_start=1, mask_end=None):
    M = np.zeros((T, A), dtype=bool)
    if mask_end is None: mask_end = T
    mask_start = max(1, int(mask_start)); mask_end = min(T, int(mask_end))
    if mask_start < mask_end: M[mask_start:mask_end, masked_agent] = True
    M[0,:] = False
    return M

def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def _prepare_inputs_base(traj, M, blank_masked=True):
    """Build base x_in/y_gt/loss_mask like training, without region one-hot."""
    T, A, _ = traj.shape
    xy = traj.reshape(T, A*2).astype(np.float32)
    if blank_masked:
        mask_flat = np.repeat(M.astype(bool), 2, axis=1)  # (T, A*2)
        xy = xy.copy()
        xy[mask_flat] = 0.0  # hide masked player's coords from the model
    obs = (~M).astype(np.float32)
    x_in = np.concatenate([xy, obs], axis=1).astype(np.float32)  # (T, A*3)
    y_gt = traj.reshape(T, A*2).astype(np.float32)
    loss_mask = np.repeat(M.astype(np.float32), 2, axis=1)
    return x_in, y_gt, loss_mask

# ---------- OpenAI: sentence -> per-timestep grid ids ----------
def sentence_to_grid_ids(prompt: str, T: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Returns np.ndarray of length T with integers in [0, grid_w*grid_h - 1].
    Requires OPENAI_API_KEY in environment. Raises RuntimeError on failure.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not available. Install `openai`.") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)

    # JSON schema ensures exactly T integers with bounds [0, R-1]
    R = grid_w * grid_h
    schema = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "minItems": T,
                "maxItems": T,
                "items": {"type": "integer", "minimum": 0, "maximum": R - 1}
            }
        },
        "required": ["ids"],
        "additionalProperties": False
    }

    system_instructions = (
        "You map a short movement description into a sequence of grid cell indices for a basketball court.\n"
        f"The court is discretized into a {grid_w}x{grid_h} grid (total {R} bins). "
        "Indices are row-major with y from bottom (0) to top and x from left (0) to right:\n"
        "  id = y * GRID_W + x, where 0 <= x < GRID_W and 0 <= y < GRID_H.\n"
        "The output must contain exactly T integers, one per time step (t=1..T), where T is provided by the caller.\n"
        "Interpret vague phrases by spreading steps across a reasonable path. "
        "Use gradual transitions; avoid jumping indices unless the text implies a sudden move.\n"
        "Return STRICT JSON only."
    )

    # We pass T explicitly to the model via the user message for clarity.
    user_msg = (
        f"T = {T} timesteps.\n"
        f"GRID_W = {grid_w}, GRID_H = {grid_h} (row-major id = y*{grid_w} + x).\n\n"
        f"Description:\n{prompt}\n\n"
        "Return JSON with an `ids` array of length T."
    )

    rsp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "grid_ids", "schema": schema, "strict": True}},
    )

    try:
        data = json.loads(rsp.output_text)
        ids = np.array(data["ids"], dtype=np.int64)
    except Exception as e:
        raise RuntimeError(f"Failed to parse OpenAI output: {e}")

    if ids.shape[0] != T or ids.min() < 0 or ids.max() >= R:
        raise RuntimeError("OpenAI returned invalid ids (wrong length or out-of-range).")
    return ids

# ---------- teams / viz helpers ----------
def _ensure_rgba(img): return img.convert("RGBA") if img.mode != "RGBA" else img
def _to_pixels(xy_norm, W, H):
    x = xy_norm[...,0]*W; y = (1.0-xy_norm[...,1])*H; return np.stack([x,y],axis=-1)
def _draw_circle(draw, xy, r, fill, outline=None, width=2):
    x,y=float(xy[0]), float(xy[1]); bbox=[x-r,y-r,x+r,y+r]
    draw.ellipse(bbox, fill=fill, outline=outline, width=width if outline else 0)
def _draw_line(draw, p0, p1, color, width=3):
    draw.line([float(p0[0]),float(p0[1]),float(p1[0]),float(p1[1])], fill=color, width=width)
def _try_font():
    try: return ImageFont.truetype("DejaVuSans.ttf", 18)
    except: return ImageFont.load_default()

def _parse_team_assign(s, A):
    """Parse '--team_assign' like '0,0,0,0,0,1,1,1,1,1' into a list[int] of len A."""
    if s is None:
        return [0]*(A//2) + [1]*(A - A//2)
    vals = [int(x.strip()) for x in s.split(",") if x.strip()!=""]
    if len(vals) != A:
        raise ValueError(f"--team_assign must have {A} entries, got {len(vals)}")
    return vals

def make_gif(court_png, traj_gt, traj_pred, mask_bool, masked_agent, caption, out_path,
             fps=10, trail=8, show_ids=False, team_assign=None):
    court = _ensure_rgba(Image.open(court_png)); W,H=court.size
    T,A,_=traj_gt.shape
    gt_px   = _to_pixels(_normalize_xy(traj_gt), W, H)
    pred_px = _to_pixels(_normalize_xy(traj_pred), W, H)

    TEAM_COLORS = [
        (220, 20, 60, 230),   # Team 0
        (153, 50, 204, 230),  # Team 1
        (34, 139, 34, 230),   # Team 2
        (255, 140, 0, 230),   # Team 3
    ]
    TEAM_TRAIL_ALPHA = 140

    # Masked-agent colors (GT vs Pred)
    COLOR_GT=(46, 204, 64, 230)         # green for GT
    COLOR_GT_TRAIL=(46, 204, 64, 140)
    COLOR_PRED=(66,133,244,230)         # blue for prediction
    COLOR_PRED_TRAIL=(66,133,244,140)
    COLOR_TEXT=(20,20,20,255); COLOR_WHITE=(255,255,255,255)
    R_OTHER=8; R_MASK=10
    font=_try_font(); frames=[]

    if team_assign is None:
        team_assign = [0]*(A//2) + [1]*(A - A//2)
    agent_fill = []
    agent_trail = []
    for a in range(A):
        t = team_assign[a] if a < len(team_assign) else 0
        base = TEAM_COLORS[t % len(TEAM_COLORS)]
        trail_col = tuple(list(base[:3]) + [TEAM_TRAIL_ALPHA])
        agent_fill.append(base)
        agent_trail.append(trail_col)

    for t in range(T):
        frame=court.copy(); draw=ImageDraw.Draw(frame,"RGBA")
        t0=max(0,t-trail)

        # Trails for non-masked agents (team-colored)
        for a in range(A):
            if a==masked_agent: continue
            for k in range(t0,t):
                _draw_line(draw, gt_px[k,a], gt_px[k+1,a], agent_trail[a], 2)

        # Trails for masked agent (GT first, then Pred on top)
        for k in range(t0,t):
            _draw_line(draw, gt_px[k,masked_agent],   gt_px[k+1,masked_agent],   COLOR_GT_TRAIL,   4)
        for k in range(t0,t):
            _draw_line(draw, pred_px[k,masked_agent], pred_px[k+1,masked_agent], COLOR_PRED_TRAIL, 4)

        # Dots for non-masked agents
        for a in range(A):
            if a==masked_agent: continue
            _draw_circle(draw, gt_px[t,a], R_OTHER, fill=agent_fill[a])
            if show_ids:
                draw.text((gt_px[t,a,0]+6, gt_px[t,a,1]-6), f"{a}", fill=COLOR_TEXT, font=font)

        # Masked agent GT dot (smaller), then predicted dot with white halo on top
        _draw_circle(draw, gt_px[t,masked_agent],   R_MASK-3, fill=COLOR_GT, outline=None)
        _draw_circle(draw, pred_px[t,masked_agent], R_MASK+2, fill=None, outline=COLOR_WHITE, width=5)
        _draw_circle(draw, pred_px[t,masked_agent], R_MASK,   fill=COLOR_PRED, outline=None)

        lines=[f"t={t+1}/{T}", f"masked agent={masked_agent}",
               f"{'masked' if mask_bool[t,masked_agent] else 'observed'} at t",
               f"{caption}", "legend: GT=green, Pred=blue"]
        yoff=8
        for s in lines:
            draw.text((10,yoff), s, fill=COLOR_TEXT, font=font); yoff+=20
        frames.append(frame)

    duration=1.0/max(1,fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    print(f"GIF saved to {out_path}")

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/vec_xl_baseline/best.pt")
    ap.add_argument("--traj_file", default="/mnt/data/mywork/nbaWork/prev/UniTraj-pytorch/Sports-Traj/datasets/pre-processed/basketball/train_clean.p")
    ap.add_argument("--masked_agent", type=int, default=1)
    ap.add_argument("--ddim_steps", type=int, default=None)
    ap.add_argument("--mask_start", type=int, default=1)
    ap.add_argument("--mask_end", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--save_npy", type=str, default=None)
    ap.add_argument("--save_npz", type=str, default=None)
    ap.add_argument("--seq_idx", type=int, default=0)
    # regions (choose one)
    ap.add_argument("--region_ids_json", type=str, default=None,
                    help='JSON list of integer grid ids per timestep (length T), each in [0, R_BINS-1].')
    ap.add_argument("--prompt", type=str, default=None,
                    help="Natural language to be mapped into per-timestep grid ids via OpenAI.")
    # viz
    ap.add_argument("--gif_out", type=str, default="test.gif")
    ap.add_argument("--court_png", type=str, default="court.png")
    ap.add_argument("--gif_fps", type=int, default=10)
    ap.add_argument("--gif_trail", type=int, default=8)
    ap.add_argument("--viz_show_ids", action="store_true")
    # debug: keep masked coords instead of blanking (NOT recommended)
    ap.add_argument("--keep_masked_input", action="store_true",
                    help="Do NOT blank masked coords in x_in (for debugging).")
    # teams
    ap.add_argument("--team_assign", type=str, default="-1,0,0,0,0,0,1,1,1,1,1",
                    help="Comma-separated team ids per agent, e.g. '0,0,0,0,0,1,1,1,1,1'. If omitted, auto-splits half/half.")

    args=ap.parse_args()

    ckpt=torch.load(args.checkpoint, map_location="cpu")
    cfg=ckpt.get("cfg", {})
    arch=cfg.get("arch","xl_vec")  # "xl_vec" or "diff_vec"
    d=cfg.get("d",512); h=cfg.get("h",16); L=cfg.get("L",12)
    dropout=cfg.get("dropout",0.1)
    steps_diff=cfg.get("steps_diff",200)

    device = (torch.device("cuda") if (args.device in ["auto","cuda"] and torch.cuda.is_available())
              else torch.device("cpu"))

    traj=_load_traj(args.traj_file, seq_idx=args.seq_idx)  # (T,A,2)
    T,A,_=traj.shape
    assert 0 <= args.masked_agent < A
    M=_make_mask(T,A,args.masked_agent,args.mask_start,args.mask_end)

    # parse teams
    team_assign = _parse_team_assign(args.team_assign, A)

    # base x_in/y_gt/loss_mask (no regions yet)
    x_in_np, y_gt_np, loss_mask_np = _prepare_inputs_base(
        traj, M, blank_masked=(not args.keep_masked_input)
    )

    # build region one-hot (T,R) with priority: json > prompt(OpenAI) > oracle
    R = R_BINS  # 40
    if args.region_ids_json:
        ids = np.array(json.loads(args.region_ids_json), dtype=np.int64)
        assert ids.shape[0] == T, f"region_ids_json length {len(ids)} must equal T={T}"
        assert ids.min() >= 0 and ids.max() < R, f"region ids must be in [0,{R-1}]"
        reg_onehot = np.eye(R, dtype=np.float32)[ids]
        caption = f"vector-conditioned: {GRID_W}x{GRID_H} grid (json)"
    elif args.prompt:
        try:
            ids = sentence_to_grid_ids(args.prompt, T, GRID_W, GRID_H)
            reg_onehot = np.eye(R, dtype=np.float32)[ids]
            caption = f"vector-conditioned: {GRID_W}x{GRID_H} grid (OpenAI)"
        except Exception as e:
            print(f"[warn] OpenAI mapping failed ({e}). Falling back to ORACLE grid labels.")
            xy_norm = _normalize_xy(traj)
            ids = _agent_region_ids_grid(xy_norm, args.masked_agent)
            reg_onehot = np.eye(R, dtype=np.float32)[ids]
            caption = f"vector-conditioned: {GRID_W}x{GRID_H} grid (oracle)"
    else:
        xy_norm = _normalize_xy(traj)
        ids = _agent_region_ids_grid(xy_norm, args.masked_agent)  # (T,)
        reg_onehot = np.eye(R, dtype=np.float32)[ids]
        caption = f"vector-conditioned: {GRID_W}x{GRID_H} grid (oracle)"

    # Optionally zero out region condition at t=0 (since initial pos is observed):
    # reg_onehot[0] = 0.0

    # append region one-hot to x_in
    x_in_np = np.concatenate([x_in_np, reg_onehot.astype(np.float32)], axis=1)  # (T, A*3 + R)

    # sanity: confirm masked coords zeroed
    if not args.keep_masked_input:
        flat = x_in_np[:, :A*2]
        mask_flat = np.repeat(M, 2, axis=1)
        zero_ok = np.allclose(flat[mask_flat], 0.0)
        print(f"[check] masked coords blanked: {zero_ok} (should be True)")

    # build and run model
    if arch=="xl_vec":
        model=VectorConditionedImputerXL(
            num_agents=A, region_dim=R,
            d_model=d, nhead=h, num_layers=L, dropout=dropout
        ).to(device)
        model.load_state_dict(_strip_module_prefix(ckpt["model"]), strict=True)
        model.eval()

        x_in=torch.from_numpy(x_in_np).unsqueeze(0).to(device)
        y_gt=torch.from_numpy(y_gt_np).unsqueeze(0).to(device)
        loss_mask=torch.from_numpy(loss_mask_np).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            pred,_,_=model(x_in, loss_mask)  # (1,T,A*2)
        P = pred.detach().cpu().numpy()[0].reshape(T,A,2)
        # Keep observed coords for masked agent from GT
        obs_agent = ~M[:, args.masked_agent]
        P[obs_agent, args.masked_agent, :] = traj[obs_agent, args.masked_agent, :]

    elif arch=="diff_vec":
        dif=DiffusionImputerVec(A, R, d,h,L,dropout,steps_diff,device.type)
        base=dif.net.module if isinstance(dif.net, nn.DataParallel) else dif.net
        base.load_state_dict(_strip_module_prefix(ckpt["model"]), strict=True)
        dif.net.eval()
        x_in=torch.from_numpy(x_in_np).unsqueeze(0).to(device)
        m=torch.from_numpy(loss_mask_np).unsqueeze(0).to(device)
        ddim_steps=args.ddim_steps if args.ddim_steps is not None else cfg.get("ddim_steps",60)
        with torch.no_grad():
            pred=dif.sample(x_in, m, ddim_steps=ddim_steps)  # (1,T,A*2)
        P = pred.detach().cpu().numpy()[0].reshape(T,A,2)
        obs_agent = ~M[:, args.masked_agent]
        P[obs_agent, args.masked_agent, :] = traj[obs_agent, args.masked_agent, :]
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # metrics vs ground truth on masked spans
    if args.metrics:
        P_t=torch.from_numpy(P.reshape(1,T,A*2))
        Y_t=torch.from_numpy(y_gt_np.reshape(1,T,A*2))
        M_t=torch.from_numpy(loss_mask_np.reshape(1,T,A*2))
        ade,fde=ade_fde_masked(P_t,Y_t,M_t,A)
        coll=collision_rate(P_t,A,1.5,M_t)
        sm=smoothness_penalty(P_t,A)
        print(f"ADE {ade:.4f} | FDE {fde:.4f} | Coll {coll:.4f} | Smooth {sm:.6f}")

    # save arrays
    if args.save_npy:
        np.save(args.save_npy, P); print(f"Saved pred P (T,A,2) to {args.save_npy}")
    if args.save_npz:
        np.savez(args.save_npz, pred_full=P, gt=traj, mask=M.astype(np.uint8),
                 team_assign=np.array(team_assign, dtype=np.int32))
        print(f"Saved NPZ to {args.save_npz}")

    # GIF: draw GT (green) vs Pred (blue) for masked agent + team-colored others
    if args.gif_out:
        if not os.path.isfile(args.court_png):
            raise FileNotFoundError(f"court.png not found at {args.court_png}")
        make_gif(args.court_png, traj_gt=traj, traj_pred=P, mask_bool=M,
                 masked_agent=args.masked_agent, caption=caption, out_path=args.gif_out,
                 fps=args.gif_fps, trail=args.gif_trail, show_ids=args.viz_show_ids,
                 team_assign=team_assign)

if __name__=="__main__":
    main()