#!/usr/bin/env python3
"""Entry point: curriculum self-play SAC training, fully resumable.

  python train.py                       # fresh run with configs/config.yaml
  python train.py --config configs/smoke.yaml
  python train.py agent.batch_size=2048 env.track=austria   # dotlist overrides

On start it auto-detects the latest checkpoint under ``checkpoint.dir`` and
resumes seamlessly (model + optimizer + replay buffer + step count + RNG +
curriculum stage + league).  Designed to be killed and restarted at any time.
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3.common.logger import configure

from agents.league import League
from agents.sac_factory import build_sac, load_sac
from common import detect_device, set_global_seed
from config_util import load_config, stage_list
from envs.make_env import make_vec_env
from training.autoscale import apply_autoscale
from training.callbacks import (CheckpointCallback, EvalCallback, LeagueSnapshotCallback,
                                RacingMetricsCallback, ThroughputCallback)
from training.checkpoint import CheckpointManager
from training.curriculum import stage_for_step, stage_window, total_steps


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="extra YAML layered on configs/config.yaml")
    p.add_argument("--base", default="configs/config.yaml")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after reaching this global step (cap a session; resume continues)")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. agent.batch_size=1024")
    return p.parse_args(argv)


def apply_mutable_hparams(model, cfg) -> None:
    """Re-apply GPU-saturation knobs after a resume (net_arch is fixed at creation)."""
    try:
        model.batch_size = int(cfg.agent.batch_size)
        model.gradient_steps = int(cfg.agent.gradient_steps)
    except Exception:
        pass


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.base, extra=args.config, overrides=args.overrides)

    hw = apply_autoscale(cfg)
    set_global_seed(int(cfg.experiment.seed))
    device = detect_device(str(cfg.hardware.device))

    ckpt_dir = str(cfg.checkpoint.dir)
    tb_dir = str(cfg.logging.tb_dir)
    league_dir = os.path.join(ckpt_dir, "league")
    for d in (ckpt_dir, tb_dir, league_dir, str(cfg.env.scenario_dir)):
        os.makedirs(d, exist_ok=True)

    print(f"[hw] {hw}")
    print(f"[cfg] device={device} backend={cfg.agent.backend} algo={cfg.agent.algo} "
          f"net={list(cfg.agent.net_arch)} batch={cfg.agent.batch_size} "
          f"UTD(gradient_steps)={cfg.agent.gradient_steps} buffer={cfg.agent.buffer_size} "
          f"n_envs={cfg.env.n_envs} track={cfg.env.track}")

    stages = stage_list(cfg)
    grand_total = total_steps(stages)
    ckpt_mgr = CheckpointManager(cfg)
    league = League(league_dir, cfg.agent, cfg.league)

    resume = ckpt_mgr.has_checkpoint()
    state0 = ckpt_mgr.latest_state() if resume else None
    global_step0 = int(state0["global_step"]) if state0 else 0
    if state0 and "league" in state0:
        league.load_state(state0["league"])
    start_stage = stage_for_step(stages, global_step0)

    seed = int(cfg.experiment.seed)
    vec = make_vec_env(cfg, stages[start_stage], league_dir, seed=seed)

    if resume:
        print(f"[resume] global_step={global_step0} -> stage {start_stage} "
              f"({stages[start_stage]['name']}); league size={league.size()}")
        model, _ = ckpt_mgr.restore(lambda mp: load_sac(cfg, mp, env=vec, device=device))
        apply_mutable_hparams(model, cfg)
        is_first_learn = False
    else:
        print("[fresh] starting new run")
        model = build_sac(cfg, vec, device=device, tensorboard_log=tb_dir, seed=seed)
        is_first_learn = True

    # One continuous logger across all stages / resumes -> a single TB run.
    model.set_logger(configure(tb_dir, ["stdout", "tensorboard", "csv"]))

    def state_provider():
        return {"stage_idx": current_stage_idx[0], "stage_name": stages[current_stage_idx[0]]["name"],
                "league": league.state(), "grand_total": grand_total}

    current_stage_idx = [start_stage]

    for i in range(start_stage, len(stages)):
        current_stage_idx[0] = i
        stage = stages[i]
        start, end = stage_window(stages, i)
        if model.num_timesteps >= end:
            continue
        num_agents = int(stage["num_agents"])
        use_league = bool(stage.get("opponents_from_league", False)) and num_agents > 1

        # Seed the league so a self-play stage always has at least one opponent.
        if use_league and league.size() == 0:
            league.snapshot(model, model.num_timesteps)
            print(f"[league] seeded with current policy (size={league.size()})")

        if i != start_stage or not resume:
            # (Re)build the vec env for this stage (different car count).
            vec.close()
            vec = make_vec_env(cfg, stage, league_dir, seed=seed)
            model.set_env(vec)

        callbacks = [
            RacingMetricsCallback(n_envs=int(cfg.env.n_envs), log_freq=int(cfg.logging.log_interval) * 200),
            ThroughputCallback(log_freq=2000),
            CheckpointCallback(ckpt_mgr, state_provider, verbose=1),
            EvalCallback(cfg, league_dir, num_agents, use_league,
                         eval_every=int(cfg.logging.eval_every),
                         n_episodes=int(cfg.logging.eval_episodes)),
        ]
        if use_league and bool(cfg.league.enabled):
            callbacks.append(LeagueSnapshotCallback(league, verbose=1))

        effective_end = min(end, args.max_steps) if args.max_steps else end
        remaining = effective_end - model.num_timesteps
        print(f"\n=== stage {i}: {stage['name']} | cars={num_agents} | "
              f"steps {model.num_timesteps}->{effective_end} (remaining {remaining}) ===")
        if remaining > 0:
            model.learn(total_timesteps=int(remaining), reset_num_timesteps=is_first_learn,
                        callback=callbacks, progress_bar=False)
            is_first_learn = False

        # End-of-stage snapshot + checkpoint so the next stage/resume is clean.
        league.snapshot(model, model.num_timesteps)
        ckpt_mgr.save(model, {"global_step": int(model.num_timesteps), **state_provider()},
                      force_buffer=True)

        if args.max_steps and model.num_timesteps >= args.max_steps:
            print(f"[max-steps] reached {model.num_timesteps} >= {args.max_steps}; stopping (resume to continue)")
            break

    print(f"\n[done] trained to {model.num_timesteps}/{grand_total} steps")
    vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
