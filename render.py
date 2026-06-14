#!/usr/bin/env python3
"""Render a trained agent racing, headless -> MP4/GIF.

  python render.py --checkpoint latest --mode solo --out race.mp4
  python render.py --mode h2h --latest latest --early snapshot_000000300 --out duel.gif

'solo' renders one car (the racing line); 'h2h' pits the latest checkpoint
(car A, chase-cam) against an earlier checkpoint/snapshot (car B).  Frames come
from PyBullet's headless renderer; encoding uses imageio's bundled ffmpeg, so no
system ffmpeg/apt is required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import imageio
import numpy as np

from agents.opponent import OpponentPolicy
from agents.sac_factory import load_sac
from common import detect_device, read_json
from config_util import load_config
from envs.make_env import make_single_env


def resolve_model_path(ckpt_dir: str, which: str) -> str:
    if which in ("latest", "", None):
        manifest = read_json(os.path.join(ckpt_dir, "latest.json"))
        return os.path.join(ckpt_dir, manifest["model"])
    if os.path.isabs(which) or os.path.exists(which):
        return which
    # try as a snapshot id under the league dir or a model name under ckpt dir
    for cand in (os.path.join(ckpt_dir, "league", f"{which}.zip"),
                 os.path.join(ckpt_dir, f"{which}.zip"),
                 os.path.join(ckpt_dir, which)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not resolve checkpoint '{which}' under {ckpt_dir}")


def rollout(cfg, model, num_agents, opponent_provider, max_steps):
    env = make_single_env(cfg, num_agents=num_agents, seed=7)
    env.opponent_provider = opponent_provider
    obs, _ = env.reset()
    frames, info = [], {}
    for _ in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        f = env.render()
        if f is not None:
            frames.append(np.asarray(f, dtype=np.uint8))
        if term or trunc:
            break
    env.close()
    return frames, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--base", default="configs/config.yaml")
    p.add_argument("--mode", choices=["solo", "h2h"], default="solo")
    p.add_argument("--checkpoint", default="latest", help="model for solo / car A in h2h")
    p.add_argument("--latest", default="latest", help="(h2h) car A checkpoint")
    p.add_argument("--early", default=None, help="(h2h) car B checkpoint/snapshot id")
    p.add_argument("--out", default="outputs/race.mp4")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_config(args.base, extra=args.config, overrides=args.overrides)
    device = detect_device(str(cfg.hardware.device))
    ckpt_dir = str(cfg.checkpoint.dir)
    max_steps = int(cfg.render.max_steps)
    fps = int(cfg.render.fps)

    if args.mode == "solo":
        model = load_sac(cfg, resolve_model_path(ckpt_dir, args.checkpoint), env=None, device=device)
        frames, info = rollout(cfg, model, num_agents=1, opponent_provider=None, max_steps=max_steps)
    else:
        a_path = resolve_model_path(ckpt_dir, args.latest)
        early = args.early
        if early is None:  # default: earliest league snapshot
            snaps = sorted(os.path.join(ckpt_dir, "league", f)
                           for f in os.listdir(os.path.join(ckpt_dir, "league"))
                           if f.startswith("snapshot_") and f.endswith(".zip"))
            if not snaps:
                raise FileNotFoundError("No league snapshots for h2h; pass --early")
            b_path = snaps[0]
        else:
            b_path = resolve_model_path(ckpt_dir, early)
        model = load_sac(cfg, a_path, env=None, device=device)
        opp = OpponentPolicy(b_path, backend=str(cfg.agent.backend), device="cpu")
        frames, info = rollout(cfg, model, num_agents=2,
                               opponent_provider=lambda n: [opp for _ in range(n)], max_steps=max_steps)

    if not frames:
        print("No frames captured (headless render returned None). Is racecar_gym installed?")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.out.lower().endswith(".gif"):
        imageio.mimsave(args.out, frames, fps=fps)
    else:
        imageio.mimsave(args.out, frames, fps=fps, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {args.out}  ({len(frames)} frames, {len(frames)/max(fps,1):.1f}s) "
          f"final progress={info.get('progress', 0):.3f} lap={info.get('lap', 0)} rank={info.get('rank', 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
