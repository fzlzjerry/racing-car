"""SB3 callbacks: training metrics, periodic eval (lap times / laps completed /
win-rate vs the league), league snapshotting, throughput/GPU, and atomic
checkpointing.  All metrics are recorded to the SB3 logger -> TensorBoard.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RacingMetricsCallback(BaseCallback):
    """Aggregate per-episode racing stats from step infos."""

    def __init__(self, n_envs: int, log_freq: int = 2000, window: int = 100):
        super().__init__()
        self.n_envs = n_envs
        self.log_freq = log_freq
        self.window = window
        self.ep_progress: List[float] = []
        self.ep_laps: List[float] = []
        self.ep_collision: List[float] = []
        self.ep_rank: List[float] = []
        self._collided = [False] * n_envs
        self._maxprog = [0.0] * n_envs
        self._next_log = log_freq

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            if info.get("collision"):
                self._collided[i] = True
            self._maxprog[i] = max(self._maxprog[i], info.get("progress", 0.0) + info.get("lap", 0))
            if done:
                self.ep_progress.append(self._maxprog[i])
                self.ep_laps.append(info.get("lap", 0))
                self.ep_collision.append(1.0 if self._collided[i] else 0.0)
                self.ep_rank.append(info.get("rank", 1))
                self._collided[i] = False
                self._maxprog[i] = 0.0
        if self.num_timesteps >= self._next_log and self.ep_progress:
            self._next_log = self.num_timesteps + self.log_freq
            w = self.window
            self.logger.record("racing/ep_total_progress", float(np.mean(self.ep_progress[-w:])))
            self.logger.record("racing/ep_laps", float(np.mean(self.ep_laps[-w:])))
            self.logger.record("racing/collision_rate", float(np.mean(self.ep_collision[-w:])))
            self.logger.record("racing/mean_rank", float(np.mean(self.ep_rank[-w:])))
        return True


class LeagueSnapshotCallback(BaseCallback):
    """Freeze the current policy into the league on the configured cadence."""

    def __init__(self, league, verbose: int = 0):
        super().__init__(verbose)
        self.league = league

    def _on_step(self) -> bool:
        if self.league is not None and self.league.should_snapshot(self.num_timesteps):
            sid = self.league.snapshot(self.model, self.num_timesteps)
            self.logger.record("league/size", self.league.size())
            if self.verbose:
                print(f"[league] snapshot {sid} @ step {self.num_timesteps} (size={self.league.size()})")
        return True


class ThroughputCallback(BaseCallback):
    """Log steps-per-second and (if available) GPU/CPU utilization."""

    def __init__(self, log_freq: int = 2000):
        super().__init__()
        self.log_freq = log_freq
        self._t0 = None
        self._s0 = 0
        self._next = log_freq
        self._nvml = None

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._s0 = self.num_timesteps
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._nvml = None

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.log_freq
            dt = max(time.time() - self._t0, 1e-6)
            sps = (self.num_timesteps - self._s0) / dt
            self.logger.record("perf/sps", float(sps))
            if self._nvml is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml)
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml)
                    self.logger.record("perf/gpu_util", float(util.gpu))
                    self.logger.record("perf/gpu_mem_gb", float(mem.used) / 1e9)
                except Exception:
                    pass
            try:
                import psutil
                self.logger.record("perf/ram_pct", psutil.virtual_memory().percent)
            except Exception:
                pass
            self._t0 = time.time()
            self._s0 = self.num_timesteps
        return True


class EvalCallback(BaseCallback):
    """Periodic deterministic eval: best/median lap time, % laps completed, and
    (in self-play stages) win-rate vs league opponents.  Builds its own single
    env in the main process so it never interferes with the SubprocVecEnv."""

    def __init__(self, cfg, league_dir: str, num_agents: int, use_league: bool,
                 eval_every: int, n_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.league_dir = league_dir
        self.num_agents = num_agents
        self.use_league = use_league
        self.eval_every = int(eval_every)
        self.n_episodes = int(n_episodes)
        self._next = self.eval_every
        self._tt_env = None    # time-trial (1 car) env for lap times
        self._race_env = None  # multi-car env for win-rate

    def _tt(self):
        if self._tt_env is None:
            from envs.make_env import make_single_env
            self._tt_env = make_single_env(self.cfg, num_agents=1, seed=12345)
        return self._tt_env

    def _race(self):
        if self._race_env is None and self.use_league and self.num_agents > 1:
            from envs.make_env import make_single_env
            self._race_env = make_single_env(self.cfg, num_agents=self.num_agents,
                                             league_dir=self.league_dir, use_league=True, seed=999)
        return self._race_env

    def _lap_metrics(self):
        env = self._tt()
        lap_times, laps_fracs = [], []
        target = int(self.cfg.env.laps)
        for _ in range(self.n_episodes):
            obs, _ = env.reset()
            last_lap, lap_start, ep_laps, done = None, 0.0, 0, False
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                t, lap = info["time"], info["lap"]
                if last_lap is None:
                    last_lap, lap_start = lap, t
                elif lap > last_lap:
                    lap_times.append(t - lap_start)
                    lap_start, last_lap, ep_laps = t, lap, ep_laps + 1
                done = term or trunc
            laps_fracs.append(min(ep_laps / max(target, 1), 1.0))
        return lap_times, laps_fracs

    def _winrate(self):
        env = self._race()
        if env is None:
            return None
        wins = 0
        for _ in range(self.n_episodes):
            obs, _ = env.reset()
            final_rank, done = 1, False
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                final_rank = info["rank"]
                done = term or trunc
            wins += 1 if final_rank == 1 else 0
        return wins / max(self.n_episodes, 1)

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.eval_every
        lap_times, laps_fracs = self._lap_metrics()
        if lap_times:
            self.logger.record("eval/best_lap_time", float(np.min(lap_times)))
            self.logger.record("eval/median_lap_time", float(np.median(lap_times)))
        self.logger.record("eval/laps_completed_frac", float(np.mean(laps_fracs)) if laps_fracs else 0.0)
        wr = self._winrate()
        if wr is not None:
            self.logger.record("eval/win_rate_vs_league", float(wr))
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] best_lap="
                  f"{(min(lap_times) if lap_times else float('nan')):.2f} "
                  f"laps_frac={np.mean(laps_fracs) if laps_fracs else 0:.2f} "
                  f"win_rate={wr}")
        return True

    def _on_training_end(self) -> None:
        for e in (self._tt_env, self._race_env):
            if e is not None:
                try:
                    e.close()
                except Exception:
                    pass


class CheckpointCallback(BaseCallback):
    """Atomic, resumable checkpointing on the configured cadence."""

    def __init__(self, ckpt_mgr, state_provider: Callable[[], dict], verbose: int = 0):
        super().__init__(verbose)
        self.mgr = ckpt_mgr
        self.state_provider = state_provider

    def _on_step(self) -> bool:
        if self.mgr.should_save(self.num_timesteps):
            state = {"global_step": int(self.num_timesteps), **self.state_provider()}
            self.mgr.save(self.model, state)
            if self.verbose:
                print(f"[ckpt] saved @ step {self.num_timesteps}")
        return True
