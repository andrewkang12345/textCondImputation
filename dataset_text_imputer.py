# dataset_text_imputer.py
# Auto-captions + per-agent masked dataset for text-conditioned trajectory imputation

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

# ----------------------------
# Utilities
# ----------------------------
def _speed(vx, vy):
    return np.sqrt(vx**2 + vy**2)

def _finite_diff(x, axis=0):
    return np.diff(x, axis=axis, prepend=x.take([0], axis=axis))

def _normalize_xy(traj_xy: np.ndarray) -> np.ndarray:
    """If data is in feet, map to [0,1] by (x/94, y/50). If it's already in [0,1], this is harmless."""
    xy = traj_xy.copy()
    # Heuristic: if max > 1.5 assume feet
    if float(np.nanmax(xy)) > 1.5:
        xy[..., 0] = xy[..., 0] / 94.0
        xy[..., 1] = xy[..., 1] / 50.0
    return np.clip(xy, 0.0, 1.0)

# ----------------------------
# Regions / helpers
# ----------------------------
_REGION_CENTERS = {
    # Right hoop near x≈0.97 (offense left→right already aligned)
    "right_corner_low":   (0.97, 0.08),
    "right_corner_high":  (0.97, 0.92),
    "left_corner_low":    (0.03, 0.08),
    "left_corner_high":   (0.03, 0.92),
    "right_wing_low":     (0.78, 0.20),
    "right_wing_high":    (0.78, 0.80),
    "left_wing_low":      (0.22, 0.20),
    "left_wing_high":     (0.22, 0.80),
    "short_corner_low":   (0.94, 0.20),
    "short_corner_high":  (0.94, 0.80),
    "elbow_right_low":    (0.85, 0.28),
    "elbow_right_high":   (0.85, 0.72),
    "foul_line":          (0.85, 0.50),
    "top_of_key":         (0.60, 0.50),
    "slot_right":         (0.70, 0.50),
    "slot_left":          (0.30, 0.50),
    "center_circle":      (0.50, 0.50),
}

_REGION_PARENTS = {
    # Map fine → coarse when we want lower granularity
    "right_corner_low": "right_corner",
    "right_corner_high":"right_corner",
    "left_corner_low":  "left_corner",
    "left_corner_high": "left_corner",
    "right_wing_low":   "right_wing",
    "right_wing_high":  "right_wing",
    "left_wing_low":    "left_wing",
    "left_wing_high":   "left_wing",
    "short_corner_low": "short_corner",
    "short_corner_high":"short_corner",
    "elbow_right_low":  "elbow_right",
    "elbow_right_high": "elbow_right",
    "slot_right":       "slot_right",
    "slot_left":        "slot_left",
    "top_of_key":       "top_of_key",
    "foul_line":        "foul_line",
    "center_circle":    "center_circle",
}

def _nearest_region(p: np.ndarray) -> str:
    best, bd = None, 1e9
    for k, c in _REGION_CENTERS.items():
        d = float(np.linalg.norm(p - np.array(c, dtype=np.float32)))
        if d < bd:
            best, bd = k, d
    # tags
    tags = []
    if p[0] >= 0.72:  # crude "perimeter" tag on right half
        tags.append("perimeter")
    if p[1] <= 0.08:
        tags.append("near_low_sideline")
    elif p[1] >= 0.92:
        tags.append("near_high_sideline")
    if 0.48 <= p[0] <= 0.52:
        tags.append("half_court_line")
    return best + ("+" + "+".join(tags) if tags else "")

# ----------------------------
# Lexical pools for augmentation
# ----------------------------
_SYNONYMS = {
    "stationary": ["standing still", "mostly stationary", "static", "hardly moving"],
    "jogging":    ["jogging", "moving at a jog", "cruising"],
    "running":    ["running", "sprinting", "moving fast"],
    "accelerated":["then accelerated", "then picked up speed", "then burst forward"],
    "straight":   ["straight", "direct", "linear"],
    "slightly curved": ["slightly curved", "gentle curve", "subtle arc"],
    "curvy":      ["curvy", "winding", "meandering"],
    "circular":   ["circular", "looping", "arc-like"],
    "on_route":   ["on a", "following a", "along a"],
    "while_mark": ["while closely marking another player", "while tight to a defender", "while staying glued to a matchup"],
    "moved_through": ["moved through", "traveled via", "passed through", "visited"],
    "visited_regions": ["visited regions", "covered areas", "area sequence"],
    "then": ["then", "and then", "before"],
    "commas": [", ", "; ", " — "],
}

# ----------------------------
# Rule-based captioner (agent-centric)
# ----------------------------
class RuleCaptioner:
    """
    Generates a detailed, agent-centric caption per sequence/player.
    Works on normalized coordinates [0,1] and assumes offense goes left->right.
    """

    def __init__(self,
                 min_run_speed=0.12,         # ~6 ft/s (unused directly; kept for reference)
                 screen_stationary_spd=0.04, # ~2 ft/s (kept for future extensions)
                 pass_by_distance=0.03,      # ~1.5 ft
                 roll_window=10,             # frames after contact
                 min_roll_speed=0.16,        # ~8 ft/s
                 seed:int=1337):
        self.pass_by_distance = pass_by_distance
        self.roll_window = roll_window
        self.min_roll_speed = min_roll_speed
        self.rng = np.random.RandomState(seed)

    # -------- Agent-centric analytics --------
    def _speed_profile(self, p: np.ndarray) -> Tuple[str, bool]:
        """Return (speed_label, started_after_idle)."""
        v = np.diff(p, axis=0, prepend=p[[0]])
        spd = np.linalg.norm(v, axis=-1)
        med = float(np.median(spd))
        # thresholds in normalized units (~ ft/s mapped)
        if med < 0.015:
            lab = "stationary"
        elif med < 0.05:
            lab = "jogging"
        else:
            lab = "running"
        start = (np.mean(spd[:max(2, len(spd)//6)]) < 0.02) and (np.mean(spd[-max(2, len(spd)//6):]) > 0.05)
        return lab, start

    def _route_shape(self, p: np.ndarray) -> str:
        v = np.diff(p, axis=0)
        if len(v) < 3:
            return "straight"
        a1 = np.arctan2(v[:,1], v[:,0])
        da = np.unwrap(a1)[1:] - np.unwrap(a1)[:-1]
        total_turn = float(np.sum(np.abs(da)))
        mean_sign = float(np.sign(np.sum(da)))
        if total_turn < 0.3:
            return "straight"
        if total_turn > 2.0 and abs(mean_sign) > 0.6:
            return "circular"
        if total_turn > 1.2:
            return "curvy"
        return "slightly curved"

    def _marking(self, xy: np.ndarray, agent: int) -> Optional[str]:
        """Heuristic 'closely marking' if nearest neighbor <~1.5 ft for ≥30% frames."""
        T, A, _ = xy.shape
        p = xy[:, agent]
        dmin = []
        for t in range(T):
            others = np.delete(xy[t], agent, axis=0)
            d = np.linalg.norm(others - p[t], axis=-1).min() if len(others) else 1e3
            dmin.append(d)
        dmin = np.array(dmin)
        thr = 1.5 / 94.0
        frac = float(np.mean(dmin < thr))
        if frac >= 0.30:
            return "close_mark"
        return None

    def _region_path(self, p: np.ndarray, rng, stochastic=True) -> str:
        labels = [_nearest_region(q) for q in p]
        # collapse consecutive duplicates
        seq = []
        for lab in labels:
            if not seq or seq[-1] != lab:
                seq.append(lab)
        if not seq:
            return "stayed_put"
        # stochastic granularity: fine→coarse
        if stochastic and rng.rand() < 0.5:
            seq = [_REGION_PARENTS.get(s.split("+")[0], s) for s in seq]
            collapsed = []
            for s in seq:
                if not collapsed or collapsed[-1] != s:
                    collapsed.append(s)
            seq = collapsed
        # cap length
        if len(seq) > 7:
            seq = seq[:3] + ["…"] + seq[-3:]
        # final tokens
        return "|".join(seq)

    # -------- Template rendering with textual augmentation --------
    def _phrase(self, key: str, rng) -> str:
        pool = _SYNONYMS.get(key, [])
        if not pool:
            return key
        return pool[int(rng.randint(0, len(pool)))]

    def _render(self, speed_lab: str, started: bool, route: str, mark: Optional[str], path_str: str, rng, deterministic: bool) -> str:
        # Map canonical tokens to surface forms
        speed_txt = self._phrase(speed_lab, rng)  # stationary/jogging/running
        route_txt = self._phrase(route, rng)
        on_route = self._phrase("on_route", rng)
        moved_through = self._phrase("moved_through", rng)
        visited_regions = self._phrase("visited_regions", rng)
        then_kw = self._phrase("then", rng)
        comma = self._phrase("commas", rng)
        mark_txt = self._phrase("while_mark", rng) if mark == "close_mark" else None

        # Turn region tokens into readable text
        regions = path_str.split("|")
        pretty_regions = [r.replace("_", " ").replace("+", " + ") for r in regions]
        if "…" in regions:
            region_clause = f"{visited_regions}: " + ", ".join(pretty_regions)
        else:
            region_clause = f"{moved_through} " + ", ".join(pretty_regions)

        # Candidate templates (vary clause order & connectors)
        templates = [
            # T0: start->speed->route->mark->regions
            lambda: (("was stationary, " + then_kw + " ") if started else "")
                    + f"{speed_txt}{comma}{on_route} {route_txt} route"
                    + (f"{comma}{mark_txt}" if mark_txt else "")
                    + f"{comma}{region_clause}",
            # T1: speed->regions->route->mark
            lambda: f"{speed_txt}{comma}{region_clause}{comma}{on_route} {route_txt} route"
                    + (f"{comma}{mark_txt}" if mark_txt else ""),
            # T2: regions first
            lambda: f"{region_clause}{comma}{speed_txt}{comma}{on_route} {route_txt} route"
                    + (f"{comma}{mark_txt}" if mark_txt else ""),
            # T3: compact
            lambda: (("stationary " + then_kw + " ") if started else "")
                    + f"{speed_txt} {on_route} {route_txt} route; {region_clause}"
                    + (f"; {mark_txt}" if mark_txt else ""),
        ]
        idx = 0 if deterministic else int(rng.randint(0, len(templates)))
        out = templates[idx]()
        # light cleanup: collapse doubles
        return " ".join(out.split())

    def caption_agent(self, traj_xy: np.ndarray, agent: int, rng: np.random.RandomState, allow_screen: bool = True, deterministic_text: bool = False) -> str:
        """
        Detailed, agent-centric caption for a single player, with textual augmentation.
        """
        xy = _normalize_xy(traj_xy)
        p = xy[:, agent]
        speed_lab, started = self._speed_profile(p)
        route = self._route_shape(p)
        mark = self._marking(xy, agent)
        path = self._region_path(p, rng, stochastic=not deterministic_text)
        return self._render(speed_lab, started, route, mark, path, rng, deterministic_text)

# ----------------------------
# Dataset with per-agent masking + captions
# ----------------------------
class TextImputationDataset(Dataset):
    """
    Returns (per sample):
      x_in: (T, A*3)   [xy per agent (A*2)] + [obs_flag per agent (A)]
      y_gt: (T, A*2)
      loss_mask: (T, A*2)  1 where coords are masked (we supervise those)
      caption: str   (agent-centric)
      target_agent: int
    """

    def __init__(self,
                 traj_file: str,
                 mask_ratio: float = 0.30,
                 min_span: int = 3,
                 max_span: int = 12,
                 deterministic_mask: bool = False,
                 # captioning / augmentation
                 mirror_prob: float = 0.15,
                 synonym_prob: float = 0.50,
                 deterministic_caption: bool = False,
                 seed: int = 1337,
                 per_agent: bool = True):
        import pickle
        with open(traj_file, "rb") as f:
            data = pickle.load(f)  # (N, T, A, 2) float32
        assert data.ndim == 4 and data.shape[-1] == 2
        self.data = data.astype(np.float32)
        self.N, self.T, self.A, _ = self.data.shape

        self.mask_ratio = float(mask_ratio)
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        self.deterministic_mask = bool(deterministic_mask)

        self.mirror_prob = float(mirror_prob)
        self.synonym_prob = float(synonym_prob)
        self.det_caption = bool(deterministic_caption)
        self.seed = int(seed)
        self.captioner = RuleCaptioner(seed=seed)
        self.per_agent = bool(per_agent)

    def __len__(self) -> int:
        return self.N * self.A if self.per_agent else self.N

    def _index_map(self, idx: int) -> Tuple[int, int]:
        if not self.per_agent:
            return idx, -1
        seq_idx = idx // self.A
        agent_idx = idx % self.A
        return seq_idx, agent_idx

    def _rng(self, seq_idx: int):
        return np.random.RandomState(self.seed + seq_idx) if (self.det_caption or self.deterministic_mask) else np.random

    def _mask_plan_single(self, rng, target_agent: int) -> np.ndarray:
        T, A = self.T, self.A
        M = np.zeros((T, A), dtype=bool)
        # mask only the target agent, after the first frame
        target = max(1, int(self.mask_ratio * T))
        guard = 0
        while (M[:, target_agent].sum() < target) and guard < 64:
            span_len = int(rng.randint(self.min_span, self.max_span + 1))
            start = int(rng.randint(0, max(1, T - span_len + 1)))
            end = start + span_len
            M[start:end, target_agent] = True
            guard += 1
        # first frame always observed (everyone)
        M[0, :] = False
        return M

    def _maybe_mirror(self, seq_idx: int, traj: np.ndarray, cap: str) -> Tuple[np.ndarray, str]:
        rng = self._rng(seq_idx)
        if rng.rand() >= self.mirror_prob:
            return traj, cap
        xy = _normalize_xy(traj.copy())
        xy[..., 0] = 1.0 - xy[..., 0]
        cap = cap.replace("left", "<L>").replace("right", "left").replace("<L>", "right")
        cap = cap.replace("weak-side", "<W>").replace("strong-side", "weak-side").replace("<W>", "strong-side")
        return xy.astype(np.float32), cap

    def _synonymize(self, seq_idx: int, cap: str) -> str:
        rng = self._rng(seq_idx)
        if self.det_caption or rng.rand() >= self.synonym_prob:
            return cap
        # light paraphrase: randomly swap some tokens
        swaps = {
            "mostly stationary": ["standing still", "static", "hardly moving"],
            "jogging": ["moving at a jog", "cruising"],
            "running": ["moving fast", "sprinting"],
            "moved through": ["traveled via", "passed through", "visited"],
            "visited regions": ["covered areas", "area sequence"],
            "then": ["and then", "before"],
            "route": ["path", "track"],
        }
        out = cap
        for k, pool in swaps.items():
            if k in out:
                out = out.replace(k, pool[int(rng.randint(0, len(pool)))])
        return out

    def __getitem__(self, idx: int):
        seq_idx, agent_idx = self._index_map(idx)
        traj = self.data[seq_idx]  # (T, A, 2)
        rng = self._rng(seq_idx)

        # caption (agent-centric, before mirror/synonyms for consistency)
        cap = self.captioner.caption_agent(traj, agent_idx if agent_idx >= 0 else 0, rng,
                                           deterministic_text=self.det_caption)

        # optional geometric + text augmentation (train only; val uses deterministic flags)
        traj_aug, cap_aug = self._maybe_mirror(seq_idx, traj, cap)
        cap_aug = self._synonymize(seq_idx, cap_aug) if not self.det_caption else cap

        # masking plan: only target agent; first frame observed for all
        target_agent = agent_idx if agent_idx >= 0 else 0
        M = self._mask_plan_single(rng, target_agent)  # (T, A)
        obs = (~M).astype(np.float32)                  # (T, A)

        xy = traj_aug.reshape(self.T, self.A * 2).astype(np.float32).copy()
        mask_flat = np.repeat(M.astype(bool), 2, axis=1)  # (T, A*2)
        xy[mask_flat] = 0.0  # hide masked coords from inputs (matches inference)

        obs_flag = (~M).astype(np.float32)  # (T, A)
        x_in = np.concatenate([xy, obs_flag], axis=1).astype(np.float32)

        y_gt = xy                                                     # (T, A*2)
        loss_mask = np.repeat(M.astype(np.float32), 2, axis=1)        # (T, A*2)

        out = {
            "x_in": torch.from_numpy(x_in),
            "y_gt": torch.from_numpy(y_gt),
            "loss_mask": torch.from_numpy(loss_mask),
            "caption": cap_aug,
            "target_agent": int(target_agent),
        }
        return out

# ----------------------------
# Inference-time helper (per agent)
# ----------------------------
def auto_caption_per_agent(traj: np.ndarray, deterministic: bool = True, seed: int = 1337) -> List[List[str]]:
    """
    Produce a caption per agent for each sequence.
    Returns: caps[N][A] strings.
    """
    N, T, A, _ = traj.shape
    capper = RuleCaptioner(seed=seed)
    out: List[List[str]] = []
    for i in range(N):
        rng = np.random.RandomState(seed + i) if deterministic else np.random
        row = [capper.caption_agent(traj[i], a, rng, deterministic_text=deterministic) for a in range(A)]
        out.append(row)
    return out
