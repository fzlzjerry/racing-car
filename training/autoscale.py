"""Hardware auto-scaling: pick the largest VRAM profile that fits the detected
GPU and stamp net/batch/buffer/UTD/n_envs into the config so an A100 (80 GB)
and an RTX 6000 Pro (96 GB) both run hot without hand-editing YAML.

Env throughput is CPU-bound (PyBullet), so we never set n_envs above the CPU
count -- GPU saturation comes from net size + batch + gradient_steps, not from
spawning more sims than there are cores.
"""
from __future__ import annotations

import multiprocessing

from omegaconf import OmegaConf

from common import detect_vram_gib, gpu_name


def apply_autoscale(cfg) -> dict:
    info = {"gpu": gpu_name(), "vram_gib": round(detect_vram_gib(), 1),
            "auto_scale": bool(cfg.hardware.auto_scale)}
    if not cfg.hardware.auto_scale:
        info["profile"] = "disabled"
        return info

    vram = detect_vram_gib()
    profiles = OmegaConf.to_container(cfg.hardware.vram_profiles, resolve=True)
    items = sorted((int(k), v) for k, v in profiles.items())
    chosen_thresh, prof = items[0]
    for thresh, p in items:
        if vram >= thresh:
            chosen_thresh, prof = thresh, p

    cfg.agent.net_arch = list(prof["net_arch"])
    cfg.agent.batch_size = int(prof["batch_size"])
    cfg.agent.buffer_size = int(prof["buffer_size"])
    cfg.agent.gradient_steps = int(prof["gradient_steps"])

    cpu = max(1, multiprocessing.cpu_count())
    cfg.env.n_envs = min(int(prof["n_envs"]), cpu)

    info.update({"profile": chosen_thresh, "net_arch": list(cfg.agent.net_arch),
                 "batch_size": cfg.agent.batch_size, "buffer_size": cfg.agent.buffer_size,
                 "gradient_steps": cfg.agent.gradient_steps, "n_envs": cfg.env.n_envs,
                 "cpu_count": cpu})
    return info
