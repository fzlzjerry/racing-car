#!/usr/bin/env python3
"""Evaluate a checkpoint: best/median lap time, % laps completed, and win-rate
vs the league.  Prints a JSON summary (and optionally appends to TensorBoard).

  python eval.py --checkpoint latest --episodes 10
  python eval.py --config configs/smoke.yaml --episodes 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from agents.sac_factory import load_sac
from common import detect_device, read_json
from config_util import load_config
from envs.make_env import make_single_env


def resolve_model_path(ckpt_dir: str, which: str) -> str:
    if which in ("latest", "", None):
        return os.path.join(ckpt_dir, read_json(os.path.join(ckpt_dir, "latest.json"))["model"])
    return which if os.path.exists(which) else os.path.join(ckpt_dir, f"{which}.zip")


def lap_metrics(cfg, model, episodes):
    env = make_single_env(cfg, num_agents=1, seed=2024)
    target = int(cfg.env.laps)
    lap_times, laps_fracs, collisions = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        last_lap, lap_start, ep_laps, collided, done = None, 0.0, 0, False, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            if info["collision"]:
                collided = True
            t, lap = info["time"], info["lap"]
            if last_lap is None:
                last_lap, lap_start = lap, t
            elif lap > last_lap:
                lap_times.append(t - lap_start)
                lap_start, last_lap, ep_laps = t, lap, ep_laps + 1
            done = term or trunc
        laps_fracs.append(min(ep_laps / max(target, 1), 1.0))
        collisions.append(1.0 if collided else 0.0)
    env.close()
    return lap_times, laps_fracs, collisions


def winrate(cfg, model, episodes, league_dir):
    env = make_single_env(cfg, num_agents=2, league_dir=league_dir, use_league=True, seed=321)
    wins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        rank, done = 1, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            rank = info["rank"]
            done = term or trunc
        wins += 1 if rank == 1 else 0
    env.close()
    return wins / max(episodes, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--base", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--no-winrate", action="store_true")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_config(args.base, extra=args.config, overrides=args.overrides)
    device = detect_device(str(cfg.hardware.device))
    ckpt_dir = str(cfg.checkpoint.dir)
    league_dir = os.path.join(ckpt_dir, "league")

    model = load_sac(cfg, resolve_model_path(ckpt_dir, args.checkpoint), env=None, device=device)
    lap_times, laps_fracs, collisions = lap_metrics(cfg, model, args.episodes)

    summary = {
        "episodes": args.episodes,
        "best_lap_time": float(np.min(lap_times)) if lap_times else None,
        "median_lap_time": float(np.median(lap_times)) if lap_times else None,
        "laps_completed_frac": float(np.mean(laps_fracs)) if laps_fracs else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "completed_laps_total": int(len(lap_times)),
    }
    if not args.no_winrate and os.path.isdir(league_dir):
        try:
            summary["win_rate_vs_league"] = winrate(cfg, model, args.episodes, league_dir)
        except Exception as e:
            summary["win_rate_vs_league"] = f"n/a ({e})"

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
