"""Configurable reward shaping computed from racecar_gym's per-agent state.

Available signals (validated live): ``progress`` in [0,1] within a lap,
``lap`` counter, ``wall_collision`` bool, ``opponent_collisions`` list,
``wrong_way`` bool, ``rank`` int, ``velocity`` (6,), ``obstacle`` distance.

reward = w_progress * delta_progress + w_speed * speed
         - w_collision * collided - w_wrong_way * wrong_way - w_reverse * reverse
         - w_time - w_smooth * ||a - a_prev||^2
         + w_overtake * (rank improvement) + w_finish * finishing_position
"""
from __future__ import annotations

import numpy as np


class RewardFunction:
    def __init__(self, reward_cfg, num_agents: int = 1, max_speed: float = 14.0):
        self.c = reward_cfg
        self.num_agents = int(num_agents)
        self.max_speed = float(max_speed)
        self.reset()

    def reset(self) -> None:
        self._prev_total = None   # monotonic lap+progress
        self._prev_rank = None

    def compute(self, signals: dict, action, prev_action, done: bool):
        c = self.c
        progress = float(signals.get("progress", 0.0))
        lap = float(signals.get("lap", 0))
        total = lap + progress  # monotonic across laps (progress resets, lap increments)

        if self._prev_total is None:
            delta = 0.0
        else:
            delta = total - self._prev_total
            # Ignore spurious large backward jumps (reset glitches / start-line wrap noise).
            if delta < -float(c.progress_wrap_threshold):
                delta = 0.0
        self._prev_total = total

        vel = np.asarray(signals.get("velocity", np.zeros(6)), dtype=np.float32)
        speed_norm = float(min(np.linalg.norm(vel[:3]) / max(self.max_speed, 1e-6), 1.0))

        wall = bool(signals.get("wall_collision", False))
        opp = signals.get("opponent_collisions", []) or []
        collided = wall or (len(opp) > 0)
        wrong_way = bool(signals.get("wrong_way", False))
        rank = int(signals.get("rank", 1))

        terms = {
            "progress": float(c.progress) * delta,
            "speed": float(c.speed) * speed_norm,
            "collision": -float(c.collision) if collided else 0.0,
            "wrong_way": -float(c.wrong_way) if wrong_way else 0.0,
            "reverse": -float(c.reverse) if delta < 0.0 else 0.0,
            "time": -float(c.time_penalty),
        }

        if prev_action is not None:
            d = np.asarray(action, dtype=np.float32) - np.asarray(prev_action, dtype=np.float32)
            terms["action_smoothness"] = -float(c.action_smoothness) * float(np.sum(d * d))
        else:
            terms["action_smoothness"] = 0.0

        if self._prev_rank is not None and rank < self._prev_rank:
            terms["overtake"] = float(c.overtake) * (self._prev_rank - rank)
        else:
            terms["overtake"] = 0.0
        self._prev_rank = rank

        terms["finish"] = float(c.finish) * (self.num_agents - rank + 1) if done else 0.0

        reward = float(sum(terms.values()))
        return reward, terms
