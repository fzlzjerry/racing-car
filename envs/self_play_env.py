"""SelfPlayRacingEnv: a single-agent Gymnasium env over racecar_gym's
multi-agent sim.

The learner controls car 'A'.  Cars 'B', 'C', ... are driven by *frozen*
opponent policies supplied (at reset) by an ``opponent_provider`` -- typically
the self-play league.  With ``num_agents == 1`` (or no provider) this is a pure
time-trial env.  Opponent inference happens on whatever device the policy was
loaded on (CPU by default) so the training GPU is reserved for the learner.

The raw multi-agent env steps *all* cars every tick and never drops an agent
from its dicts, so there is no PettingZoo-style dead-agent bookkeeping here.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import gymnasium as gym

from .adapter import build_adapter, HEADLESS_RENDER_MODE
from .observations import ObservationBuilder
from .reward import RewardFunction

# An opponent provider maps n_opponents -> list of objects with ``.act(obs)->action``.
OpponentProvider = Callable[[int], List["SupportsAct"]]


class SelfPlayRacingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg, num_agents: int, opponent_provider: Optional[OpponentProvider] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.num_agents = int(num_agents)
        self.learner_id = "A"
        self.opponent_provider = opponent_provider

        self.adapter = build_adapter(cfg.env)
        self._raw, self.agent_ids = self.adapter.make_raw(self.num_agents, render_mode=HEADLESS_RENDER_MODE)

        self.obs_builder = ObservationBuilder(cfg.obs, action_dim=2)
        self.observation_space = self.obs_builder.space
        # SAC acts in a flat [steering, motor] box; mapped to the sim's action dict.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reward_fn = RewardFunction(cfg.reward, num_agents=self.num_agents,
                                        max_speed=float(cfg.obs.max_linear_velocity))

        self._max_steps = int(cfg.env.max_episode_steps)
        self._opponents: dict = {}
        self._last_actions: dict = {}
        self._last_raw_obs: dict = {}
        self._prev_learner_action: Optional[np.ndarray] = None
        self._step_count = 0
        self._seed = seed
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _to_sim_action(flat) -> dict:
        a = np.clip(np.asarray(flat, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return {"steering": np.array([a[0]], dtype=np.float32),
                "motor": np.array([a[1]], dtype=np.float32)}

    def _assign_opponents(self) -> None:
        self._opponents = {}
        n_opp = self.num_agents - 1
        if n_opp > 0 and self.opponent_provider is not None:
            policies = self.opponent_provider(n_opp)
            for aid, pol in zip(self.agent_ids[1:], policies):
                self._opponents[aid] = pol

    # -- gym API ----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        self._step_count = 0
        self._prev_learner_action = None
        self.reward_fn.reset()
        raw_obs, state = self._raw.reset(seed=seed, options={"mode": "grid"})
        self._last_raw_obs = raw_obs
        self._last_actions = {aid: np.zeros(2, dtype=np.float32) for aid in self.agent_ids}
        self._assign_opponents()
        obs = self.obs_builder.build(raw_obs[self.learner_id],
                                     self._last_actions[self.learner_id],
                                     state[self.learner_id])
        return obs, {}

    def step(self, action):
        learner_action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

        joint = {self.learner_id: self._to_sim_action(learner_action)}
        for aid in self.agent_ids[1:]:
            pol = self._opponents.get(aid)
            if pol is not None:
                opp_obs = self.obs_builder.build(self._last_raw_obs[aid], self._last_actions[aid])
                opp_action = np.asarray(pol.act(opp_obs), dtype=np.float32).reshape(-1)
            else:
                opp_action = np.zeros(2, dtype=np.float32)
            joint[aid] = self._to_sim_action(opp_action)
            self._last_actions[aid] = np.clip(opp_action, -1.0, 1.0)

        raw_obs, _rewards, dones, _trunc, state = self._raw.step(joint)
        self._last_raw_obs = raw_obs
        self._last_actions[self.learner_id] = learner_action
        self._step_count += 1

        signals = self.adapter.extract_signals(state[self.learner_id])
        learner_done = bool(dones[self.learner_id])
        reward, terms = self.reward_fn.compute(signals, learner_action,
                                               self._prev_learner_action, learner_done)
        self._prev_learner_action = learner_action

        obs = self.obs_builder.build(raw_obs[self.learner_id], learner_action, signals)
        truncated = self._step_count >= self._max_steps
        opp = signals.get("opponent_collisions", []) or []
        info = {
            "progress": float(signals.get("progress", 0.0)),
            "lap": int(signals.get("lap", 0)),
            "rank": int(signals.get("rank", 1)),
            "time": float(signals.get("time", 0.0)),
            "collision": bool(signals.get("wall_collision", False)) or len(opp) > 0,
            "reward_terms": terms,
        }
        if learner_done or truncated:
            info["episode_total_progress"] = float(self.reward_fn._prev_total or 0.0)
            info["episode_laps"] = int(signals.get("lap", 0))
        return obs, reward, learner_done, truncated, info

    def render(self):
        frame = self._raw.render()
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        try:
            self._raw.close()
        except Exception:
            pass
