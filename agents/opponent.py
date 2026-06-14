"""Frozen opponents for self-play.

``OpponentPolicy`` wraps a loaded SB3/SBX model for fast, deterministic CPU
inference inside SubprocVecEnv workers.  Models are cached per process so we
never re-deserialize a multi-MB checkpoint on every episode reset.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# Per-process cache: snapshot path -> loaded model.
_MODEL_CACHE: Dict[str, object] = {}


def load_model(path: str, backend: str = "sb3", device: str = "cpu"):
    """Load (and cache) a frozen policy for inference only."""
    key = f"{backend}:{device}:{path}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if backend == "sbx":
        from sbx import SAC as _SAC  # type: ignore
    else:
        from stable_baselines3 import SAC as _SAC
    model = _SAC.load(path, device=device)
    _MODEL_CACHE[key] = model
    return model


class OpponentPolicy:
    """A frozen policy that produces actions for one opponent car."""

    def __init__(self, path: str, backend: str = "sb3", device: str = "cpu",
                 deterministic: bool = True, snapshot_id: str = ""):
        self.path = path
        self.snapshot_id = snapshot_id or path
        self.deterministic = deterministic
        self._model = load_model(path, backend=backend, device=device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(np.asarray(obs, dtype=np.float32),
                                        deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)


class ConstantOpponent:
    """Fallback opponent (drives gently forward) used before any league
    snapshot exists, so a self-play stage can start immediately."""

    def __init__(self, throttle: float = 0.3, steering: float = 0.0):
        self.snapshot_id = "constant"
        self._a = np.array([float(steering), float(throttle)], dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._a.copy()
