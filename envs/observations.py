"""Turn racecar_gym's Dict sensor observation into a flat, pre-normalized
Box vector for SAC.

Design choices that matter:
  * The lidar is subsampled from 1080 beams to ``lidar_beams`` -- this is the
    single biggest lever on replay-buffer size (and therefore on how cheap it
    is to checkpoint the buffer to Google Drive).
  * Everything is normalized to roughly [-1, 1] *here*, with fixed constants
    (not a running VecNormalize).  That keeps frozen self-play opponents valid:
    a past snapshot sees exactly the same observation distribution it trained
    on, with no running-statistics drift to synchronize.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

# Per-component flat dimensions (lidar handled separately; last_action uses action_dim).
_COMPONENT_DIMS = {"velocity": 6, "pose": 6, "acceleration": 6, "progress": 1}


class ObservationBuilder:
    def __init__(self, obs_cfg, action_dim: int = 2):
        self.lidar_beams = int(obs_cfg.lidar_beams)
        self.lidar_max_range = float(obs_cfg.lidar_max_range)
        self.include = [str(c) for c in obs_cfg.include]
        self.max_lin = float(obs_cfg.max_linear_velocity)
        self.max_ang = float(obs_cfg.max_angular_velocity)
        self.action_dim = int(action_dim)
        self._lidar_idx: np.ndarray | None = None  # lazily built once raw size is known
        self.dim = self._compute_dim()

    def _compute_dim(self) -> int:
        d = 0
        if "lidar" in self.include:
            d += self.lidar_beams
        if "last_action" in self.include:
            d += self.action_dim
        for c in self.include:
            if c in _COMPONENT_DIMS:
                d += _COMPONENT_DIMS[c]
        return d

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32)

    def _lidar(self, raw_lidar) -> np.ndarray:
        lidar = np.asarray(raw_lidar, dtype=np.float32)
        if self._lidar_idx is None or len(self._lidar_idx) != self.lidar_beams:
            self._lidar_idx = np.linspace(0, len(lidar) - 1, self.lidar_beams).astype(np.int64)
        sub = lidar[self._lidar_idx] / max(self.lidar_max_range, 1e-6)
        return np.clip(sub, 0.0, 1.0)

    def build(self, raw_obs: dict, last_action, signals: dict | None = None) -> np.ndarray:
        parts = []
        if "lidar" in self.include:
            parts.append(self._lidar(raw_obs["lidar"]))
        if "velocity" in self.include:
            v = np.asarray(raw_obs["velocity"], dtype=np.float32).copy()
            v[:3] /= self.max_lin
            v[3:] /= self.max_ang
            parts.append(np.clip(v, -1.0, 1.0))
        if "pose" in self.include:
            p = np.asarray(raw_obs["pose"], dtype=np.float32).copy()
            p[:3] /= 100.0     # position bounds from racecar.yml (100, 100, 3)
            p[3:] /= np.pi     # angles
            parts.append(np.clip(p, -1.0, 1.0))
        if "acceleration" in self.include:
            a = np.asarray(raw_obs["acceleration"], dtype=np.float32).copy()
            parts.append(np.clip(a / self.max_lin, -1.0, 1.0))
        if "progress" in self.include:
            pr = 0.0 if signals is None else float(signals.get("progress", 0.0))
            parts.append(np.array([np.clip(pr, 0.0, 1.0)], dtype=np.float32))
        if "last_action" in self.include:
            parts.append(np.clip(np.asarray(last_action, dtype=np.float32).reshape(-1), -1.0, 1.0))
        return np.concatenate(parts).astype(np.float32)
