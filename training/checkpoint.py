"""Atomic, fully-resumable checkpointing to a (possibly slow Drive) folder.

Every checkpoint is a *consistent set* described by ``latest.json`` (written
last, atomically): model (weights + optimizer + num_timesteps), RNG sidecar,
a JSON training-state (global step, curriculum stage, league state), and -- on
a rarer cadence, since it is the big file -- the replay buffer.  Resume reads
``latest.json`` and restores all of it, then training continues with
``reset_num_timesteps=False``.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from typing import Callable, Optional, Tuple

from common import atomic_write, get_rng_state, read_json, set_rng_state, write_json

_CKPT_RE = re.compile(r"ckpt_(\d+)_")


class CheckpointManager:
    def __init__(self, cfg):
        c = cfg.checkpoint
        self.dir = str(c.dir)
        os.makedirs(self.dir, exist_ok=True)
        self.save_every = int(c.save_every)
        self.buffer_every = int(c.buffer_every)
        self.keep_last = int(c.keep_last)
        self.save_rb = bool(c.save_replay_buffer)
        self.manifest_path = os.path.join(self.dir, "latest.json")
        self._last_save_step = -math.inf
        self._last_buffer_step = -math.inf

    # -- queries ---------------------------------------------------------------
    def latest(self) -> Optional[dict]:
        try:
            return read_json(self.manifest_path)
        except Exception:
            return None

    def has_checkpoint(self) -> bool:
        m = self.latest()
        return bool(m and os.path.exists(os.path.join(self.dir, m.get("model", ""))))

    def latest_state(self) -> Optional[dict]:
        """Read just the training-state JSON of the latest checkpoint (used to
        pick the curriculum stage before the env/model are built)."""
        m = self.latest()
        if not m:
            return None
        try:
            return read_json(os.path.join(self.dir, m["state"]))
        except Exception:
            return None

    def should_save(self, step: int) -> bool:
        return (step - self._last_save_step) >= self.save_every

    # -- save ------------------------------------------------------------------
    def save(self, model, training_state: dict, force_buffer: bool = False) -> dict:
        step = int(training_state["global_step"])
        prefix = f"ckpt_{step:09d}"

        model_file = f"{prefix}_model.zip"
        atomic_write(lambda p: model.save(p), os.path.join(self.dir, model_file))

        rng_file = f"{prefix}_rng.pkl"
        atomic_write(lambda p: pickle.dump(get_rng_state(), open(p, "wb"), protocol=4),
                     os.path.join(self.dir, rng_file))

        state_file = f"{prefix}_state.json"
        write_json(training_state, os.path.join(self.dir, state_file))

        manifest = {"global_step": step, "model": model_file, "rng": rng_file, "state": state_file}

        want_buffer = self.save_rb and (force_buffer or (step - self._last_buffer_step) >= self.buffer_every)
        if want_buffer:
            buf_file = f"{prefix}_buffer.pkl"
            atomic_write(lambda p: model.save_replay_buffer(p), os.path.join(self.dir, buf_file))
            manifest["buffer"] = buf_file
            self._last_buffer_step = step
        else:
            prev = self.latest()
            if prev and "buffer" in prev and os.path.exists(os.path.join(self.dir, prev["buffer"])):
                manifest["buffer"] = prev["buffer"]  # carry forward last good buffer

        write_json(manifest, self.manifest_path)  # LAST: defines the consistent set
        self._last_save_step = step
        self._rotate()
        return manifest

    def _rotate(self) -> None:
        manifest = self.latest() or {}
        protected = {manifest.get(k) for k in ("model", "rng", "state", "buffer") if k in manifest}
        by_step: dict[int, list] = {}
        for f in os.listdir(self.dir):
            m = _CKPT_RE.match(f)
            if m:
                by_step.setdefault(int(m.group(1)), []).append(f)
        keep = set(sorted(by_step)[-self.keep_last:])
        for s, files in by_step.items():
            if s in keep:
                continue
            for f in files:
                if f in protected:
                    continue
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass

    # -- restore ---------------------------------------------------------------
    def restore(self, model_loader: Callable[[str], object]) -> Tuple[object, dict]:
        """``model_loader(model_path)`` must return a learner with its env attached.
        Returns ``(model, training_state)``; also restores buffer + RNG + cadence."""
        manifest = self.latest()
        if not manifest:
            raise FileNotFoundError(f"No checkpoint manifest at {self.manifest_path}")
        model = model_loader(os.path.join(self.dir, manifest["model"]))
        if "buffer" in manifest:
            buf_path = os.path.join(self.dir, manifest["buffer"])
            if os.path.exists(buf_path):
                model.load_replay_buffer(buf_path)
        rng_path = os.path.join(self.dir, manifest["rng"])
        if os.path.exists(rng_path):
            with open(rng_path, "rb") as f:
                set_rng_state(pickle.load(f))
        training_state = read_json(os.path.join(self.dir, manifest["state"]))
        step = int(training_state["global_step"])
        self._last_save_step = step
        self._last_buffer_step = step if "buffer" in manifest else -math.inf
        return model, training_state
