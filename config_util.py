"""OmegaConf config loading with layered overrides.

Usage:
    cfg = load_config("configs/config.yaml",
                      extra="configs/smoke.yaml",          # optional YAML layered on top
                      overrides=["experiment.seed=1", ...]) # optional CLI dotlist
"""
from __future__ import annotations

from typing import List, Optional

from omegaconf import OmegaConf


def load_config(base: str = "configs/config.yaml",
                extra: Optional[str] = None,
                overrides: Optional[List[str]] = None):
    cfg = OmegaConf.load(base)
    if extra:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def stage_list(cfg) -> List[dict]:
    return [OmegaConf.to_container(s, resolve=True) for s in cfg.curriculum.stages]
