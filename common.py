"""Shared utilities: atomic checkpoint writes, RNG state capture/restore,
device/VRAM detection, and checkpoint discovery.  Kept dependency-light and
import-cheap so SubprocVecEnv workers can import it.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Atomic writes to a (possibly slow Google Drive FUSE) filesystem.
# Writer must create *exactly* the path it is given; we then os.replace() it,
# which is atomic within a single filesystem, so readers never see a partial file.
# ---------------------------------------------------------------------------
def atomic_write(writer: Callable[[str], Any], final_path: str) -> str:
    d = os.path.dirname(os.path.abspath(final_path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp_{os.getpid()}_{os.path.basename(final_path)}")
    if os.path.exists(tmp):
        os.remove(tmp)
    writer(tmp)
    os.replace(tmp, final_path)
    return final_path


def write_json(obj: dict, final_path: str) -> str:
    return atomic_write(lambda p: open(p, "w").write(json.dumps(obj, indent=2)), final_path)


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RNG state -- SB3 does NOT persist live RNG, so we capture it ourselves for
# resumable training.  (Env-level PyBullet determinism is best-effort.)
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if state is None:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device / VRAM detection (drives hardware auto-scaling).
# ---------------------------------------------------------------------------
def detect_device(pref: str = "auto") -> str:
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if cuda else "cpu"
    return "cuda" if cuda else "cpu"


def detect_vram_gib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_latest_manifest(ckpt_dir: str) -> Optional[str]:
    """Return path to ``latest.json`` if a usable checkpoint manifest exists."""
    manifest = os.path.join(ckpt_dir, "latest.json")
    return manifest if os.path.exists(manifest) else None


def list_snapshots(snapshot_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.zip")))
