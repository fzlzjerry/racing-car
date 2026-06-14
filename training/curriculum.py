"""Curriculum bookkeeping: convert the list of stages into cumulative global-step
budgets so a single global step counter (model.num_timesteps) maps to a stage,
which makes mid-stage resume trivial.
"""
from __future__ import annotations

from typing import List, Tuple


def cumulative_budgets(stages: List[dict]) -> List[int]:
    out, acc = [], 0
    for s in stages:
        acc += int(s["steps"])
        out.append(acc)
    return out


def total_steps(stages: List[dict]) -> int:
    return sum(int(s["steps"]) for s in stages)


def stage_for_step(stages: List[dict], step: int) -> int:
    """Index of the stage that ``step`` falls into (clamped to the last stage)."""
    for i, end in enumerate(cumulative_budgets(stages)):
        if step < end:
            return i
    return len(stages) - 1


def stage_window(stages: List[dict], idx: int) -> Tuple[int, int]:
    """[start, end) global-step window for stage ``idx``."""
    cum = cumulative_budgets(stages)
    start = 0 if idx == 0 else cum[idx - 1]
    return start, cum[idx]
