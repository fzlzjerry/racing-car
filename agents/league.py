"""AlphaStar-lite self-play league with PFSP opponent sampling.

The league is *filesystem-backed* so it works across processes: the main
training process writes frozen snapshots + a ``league.json`` metadata file;
SubprocVecEnv workers (which can't share Python objects) read that directory to
sample and load opponents.  Atomic writes mean workers never read a torn file.

PFSP: opponents the learner rarely beats are sampled more often (hard but
winnable first), encouraging robust overtaking/defending rather than exploiting
one weak past self.
"""
from __future__ import annotations

import math
import os
from typing import List

import numpy as np

from common import atomic_write, read_json, write_json
from .opponent import ConstantOpponent, OpponentPolicy

META_NAME = "league.json"


def _pfsp_weights(snaps: List[dict], mode: str, temperature: float) -> np.ndarray:
    n = len(snaps)
    if n == 0:
        return np.ones(0)
    if mode == "uniform":
        return np.ones(n)
    if mode == "latest":
        w = np.zeros(n)
        w[-1] = 1.0
        return w
    # pfsp: weight ~ (1 - P(learner beats opp))^temperature
    w = np.empty(n, dtype=np.float64)
    for i, s in enumerate(snaps):
        g, wins = int(s.get("games", 0)), int(s.get("wins", 0))
        wr = (wins / g) if g > 0 else 0.5
        w[i] = (1.0 - wr) + 1e-3
    w = np.power(w, float(temperature))
    return w


class League:
    """Main-process snapshot manager."""

    def __init__(self, league_dir: str, agent_cfg, league_cfg):
        self.dir = league_dir
        os.makedirs(self.dir, exist_ok=True)
        self.meta_path = os.path.join(self.dir, META_NAME)
        self.backend = str(agent_cfg.backend)
        self.cfg = league_cfg
        self._last_snapshot_step = -math.inf
        if not os.path.exists(self.meta_path):
            write_json({"snapshots": []}, self.meta_path)

    def _meta(self) -> dict:
        try:
            return read_json(self.meta_path)
        except Exception:
            return {"snapshots": []}

    def size(self) -> int:
        return len(self._meta().get("snapshots", []))

    def should_snapshot(self, step: int) -> bool:
        return (step - self._last_snapshot_step) >= int(self.cfg.snapshot_every)

    def snapshot(self, model, step: int) -> str:
        """Freeze ``model`` into the league at ``step`` (atomic)."""
        sid = f"snapshot_{int(step):09d}"
        fname = f"{sid}.zip"
        path = os.path.join(self.dir, fname)
        atomic_write(lambda p: model.save(p), path)

        meta = self._meta()
        meta.setdefault("snapshots", []).append(
            {"id": sid, "step": int(step), "file": fname, "games": 0, "wins": 0})
        # Evict oldest beyond max_size.
        max_size = int(self.cfg.max_size)
        while len(meta["snapshots"]) > max_size:
            old = meta["snapshots"].pop(0)
            try:
                os.remove(os.path.join(self.dir, old["file"]))
            except OSError:
                pass
        write_json(meta, self.meta_path)
        self._last_snapshot_step = step
        return sid

    def record_result(self, snapshot_id: str, learner_won: bool) -> None:
        meta = self._meta()
        for s in meta.get("snapshots", []):
            if s["id"] == snapshot_id:
                s["games"] = int(s.get("games", 0)) + 1
                s["wins"] = int(s.get("wins", 0)) + (1 if learner_won else 0)
                break
        write_json(meta, self.meta_path)

    def state(self) -> dict:
        return {"last_snapshot_step": (None if self._last_snapshot_step == -math.inf
                                       else self._last_snapshot_step)}

    def load_state(self, state: dict) -> None:
        v = (state or {}).get("last_snapshot_step")
        self._last_snapshot_step = -math.inf if v is None else float(v)


class LeagueOpponentProvider:
    """Worker-process opponent sampler.  Re-reads the league dir each call so
    new snapshots from the main process become visible at the next episode."""

    def __init__(self, league_dir: str, cfg):
        self.dir = league_dir
        self.meta_path = os.path.join(self.dir, META_NAME)
        self.backend = str(cfg.agent.backend)
        self.device = str(cfg.league.opponent_device)
        self.sample_mode = str(cfg.league.sample)
        self.latest_fraction = float(cfg.league.latest_fraction)
        self.temperature = float(cfg.league.pfsp_temperature)
        self.rng = np.random.default_rng()

    def _snapshots(self) -> List[dict]:
        try:
            return read_json(self.meta_path).get("snapshots", [])
        except Exception:
            return []

    def __call__(self, n_opponents: int):
        snaps = self._snapshots()
        if not snaps:
            return [ConstantOpponent() for _ in range(n_opponents)]
        weights = _pfsp_weights(snaps, self.sample_mode, self.temperature)
        weights = weights / weights.sum()
        out = []
        for _ in range(n_opponents):
            if self.rng.random() < self.latest_fraction:
                s = snaps[-1]
            else:
                s = snaps[int(self.rng.choice(len(snaps), p=weights))]
            path = os.path.join(self.dir, s["file"])
            try:
                out.append(OpponentPolicy(path, backend=self.backend, device=self.device,
                                          snapshot_id=s["id"]))
            except Exception:
                out.append(ConstantOpponent())
        return out
