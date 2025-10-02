# dataset_text_imputer.py
# Vector-conditioned dataset for trajectory imputation with discretized grid regions (no text labels)

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

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
# Discretized grid regions (no text labels)
# ----------------------------
# Choose a grid that multiplies to 40. 8 x 5 works well with a 94x50 court.
GRID_W = 8   # bins along X (left→right)
GRID_H = 5   # bins along Y (bottom→top)
R_BINS = GRID_W * GRID_H  # = 40

def _point_to_grid_id(p: np.ndarray) -> int:
    """
    p in [0,1]^2. Map to a single integer id in [0, R_BINS-1] using row-major order:
    id = y * GRID_W + x
    """
    ix = int(np.clip(np.floor(p[0] * GRID_W), 0, GRID_W - 1))
    iy = int(np.clip(np.floor(p[1] * GRID_H), 0, GRID_H - 1))
    return iy * GRID_W + ix

def _agent_region_ids_grid(xy_norm: np.ndarray, agent: int) -> np.ndarray:
    """
    Return per-timestep integer grid ids for one agent.
    xy_norm: (T, A, 2) in [0,1]; agent: index; returns ids[T] in [0, R_BINS-1].
    """
    p = xy_norm[:, agent]  # (T,2)
    ids = np.array([_point_to_grid_id(q) for q in p], dtype=np.int64)
    return ids

# ----------------------------
# Dataset with per-agent masking + region grid
# ----------------------------
class TextImputationDataset(Dataset):
    """
    Returns (per sample):
      x_in: (T, A*3 + R_BINS)   [xy per agent (A*2, masked zeros)] + [obs_flag per agent (A)] + [region one-hot (R_BINS)]
      y_gt: (T, A*2)
      loss_mask: (T, A*2)  1 where coords are masked (we supervise those)
      target_agent: int
      region_ids: (T,) integer ids in [0, R_BINS-1] for the masked agent
    """

    def __init__(self,
                 traj_file: str,
                 mask_ratio: float = 0.30,
                 min_span: int = 3,
                 max_span: int = 12,
                 deterministic_mask: bool = False,
                 # augmentation
                 mirror_prob: float = 0.15,
                 deterministic_caption: bool = False,  # kept for RNG control only
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
        self.det_caption = bool(deterministic_caption)  # only affects RNG choice
        self.seed = int(seed)
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
        # Use deterministic RNG when either captioning (deprecated) or mask determinism is requested
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

    def _maybe_mirror(self, seq_idx: int, traj: np.ndarray) -> np.ndarray:
        rng = self._rng(seq_idx)
        if rng.rand() >= self.mirror_prob:
            return traj
        xy = _normalize_xy(traj.copy())
        xy[..., 0] = 1.0 - xy[..., 0]  # mirror horizontally in normalized coords
        return xy.astype(np.float32)

    def __getitem__(self, idx: int):
        seq_idx, agent_idx = self._index_map(idx)
        traj = self.data[seq_idx]  # (T, A, 2)
        rng = self._rng(seq_idx)

        traj_aug = self._maybe_mirror(seq_idx, traj)  # geometry-only augmentation

        target_agent = agent_idx if agent_idx >= 0 else 0
        M = self._mask_plan_single(rng, target_agent)  # (T, A)
        obs_flag = (~M).astype(np.float32)             # (T, A)

        # inputs: xy with masked coords zeroed
        xy = traj_aug.reshape(self.T, self.A * 2).astype(np.float32).copy()
        mask_flat = np.repeat(M.astype(bool), 2, axis=1)
        xy[mask_flat] = 0.0

        # per-timestep region one-hot for the masked agent using 8x5=40 grid
        xy_norm = _normalize_xy(traj_aug)
        reg_ids = _agent_region_ids_grid(xy_norm, target_agent)          # (T,)
        R = R_BINS
        reg_onehot = np.eye(R, dtype=np.float32)[reg_ids]                # (T, R)

        # x_in = [xy, obs_flag, region_onehot]
        x_in = np.concatenate([xy, obs_flag, reg_onehot], axis=1).astype(np.float32)

        y_gt = traj_aug.reshape(self.T, self.A * 2).astype(np.float32)
        loss_mask = np.repeat(M.astype(np.float32), 2, axis=1)

        return {
            "x_in": torch.from_numpy(x_in),
            "y_gt": torch.from_numpy(y_gt),
            "loss_mask": torch.from_numpy(loss_mask),
            "target_agent": int(target_agent),
            "region_ids": torch.from_numpy(reg_ids),   # (T,)
        }