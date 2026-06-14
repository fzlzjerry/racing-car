"""Sim-agnostic adapter.

This is the single place that knows racecar_gym specifics (env construction,
agent ids, where the reward signals live).  A future ``F1TenthAdapter`` would
implement the same surface -- note f1tenth_gym puts ``collisions`` / ``lap_*``
in the *observation* and leaves ``info`` empty, and has no built-in centerline
``progress``, so it would need a waypoint-projection here.  Everything above
this module (obs builder, reward, self-play wrapper) is sim-independent.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from .scenarios import AGENT_IDS, generate_scenario

# racecar_gym only renders headless (PyBullet DIRECT) for non-'human' modes.
HEADLESS_RENDER_MODE = "rgb_array_follow"


class RacecarGymAdapter:
    """Builds racecar_gym's MultiAgentRaceEnv and exposes its reward signals."""

    def __init__(self, env_cfg):
        self.cfg = env_cfg

    def scenario_path(self, num_agents: int) -> str:
        return os.path.join(str(self.cfg.scenario_dir),
                            f"{self.cfg.track}_{num_agents}agents.yml")

    def make_raw(self, num_agents: int, render_mode: str = HEADLESS_RENDER_MODE):
        """Return ``(raw_env, agent_ids)``.  Always the multi-agent env so a
        single code path covers 1..N cars; agent 'A' is the learner."""
        # Import here so SubprocVecEnv workers import racecar_gym in-process.
        from racecar_gym.envs.gym_api import MultiAgentRaceEnv

        path = self.scenario_path(num_agents)
        if not os.path.exists(path):
            generate_scenario(
                track=str(self.cfg.track),
                num_agents=num_agents,
                laps=int(self.cfg.laps),
                time_limit=float(self.cfg.time_limit),
                out_path=path,
            )
        env = MultiAgentRaceEnv(scenario=path, render_mode=render_mode)
        return env, list(AGENT_IDS[:num_agents])

    @staticmethod
    def extract_signals(state_for_agent: dict) -> dict:
        """racecar_gym's per-agent state already holds progress/lap/collision/
        rank/velocity/wrong_way -- pass through (the contract for other sims)."""
        return state_for_agent


def build_adapter(env_cfg):
    backend = str(getattr(env_cfg, "backend", "racecar_gym"))
    if backend == "racecar_gym":
        return RacecarGymAdapter(env_cfg)
    raise NotImplementedError(
        f"Sim backend '{backend}' not implemented. Only 'racecar_gym' is wired up; "
        "f1tenth would implement the same adapter surface (see module docstring)."
    )
