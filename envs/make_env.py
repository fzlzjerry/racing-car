"""Vectorized-env factory.

Each racecar_gym env owns its own PyBullet (DIRECT) connection, so parallel
envs MUST live in separate processes -> SubprocVecEnv for n_envs > 1.  A single
env uses DummyVecEnv.  Self-play opponents are sampled per-worker from the
filesystem-backed league (so the GPU-side learner never blocks on opponents).
"""
from __future__ import annotations

from typing import Optional

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .self_play_env import SelfPlayRacingEnv


def _make_env_fn(cfg, num_agents, league_dir, use_league, rank, seed):
    def _init():
        provider = None
        if use_league:
            from agents.league import LeagueOpponentProvider
            provider = LeagueOpponentProvider(league_dir, cfg)
        env = SelfPlayRacingEnv(cfg, num_agents=num_agents,
                                opponent_provider=provider, seed=int(seed) + rank)
        return Monitor(env)
    return _init


def make_vec_env(cfg, stage, league_dir: str, seed: int = 0,
                 n_envs: Optional[int] = None, force_dummy: bool = False):
    n_envs = int(n_envs if n_envs is not None else cfg.env.n_envs)
    num_agents = int(stage["num_agents"])
    use_league = bool(stage.get("opponents_from_league", False)) and num_agents > 1

    fns = [_make_env_fn(cfg, num_agents, league_dir, use_league, i, seed) for i in range(n_envs)]
    if force_dummy or n_envs == 1:
        return DummyVecEnv(fns)
    start_method = str(getattr(cfg.env, "vec_start_method", "fork"))
    return SubprocVecEnv(fns, start_method=start_method)


def make_single_env(cfg, num_agents: int, league_dir: Optional[str] = None,
                    use_league: bool = False, seed: int = 0) -> SelfPlayRacingEnv:
    """An un-vectorized env for eval / rendering (lets us call .render())."""
    provider = None
    if use_league and league_dir is not None and num_agents > 1:
        from agents.league import LeagueOpponentProvider
        provider = LeagueOpponentProvider(league_dir, cfg)
    return SelfPlayRacingEnv(cfg, num_agents=num_agents, opponent_provider=provider, seed=seed)
