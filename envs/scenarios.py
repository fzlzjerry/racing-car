"""Generate racecar_gym scenario YAMLs with a configurable number of cars.

racecar_gym selects the number of agents on a track from a scenario file
(``world.name`` + a list of ``agents``).  The bundled scenarios are fixed at
4 cars, so for the curriculum (1 -> 2 -> N cars) we synthesize scenarios on
the fly.  Agent ids are 'A', 'B', 'C', ... ; 'A' is always the learner.
"""
from __future__ import annotations

import os
from typing import List, Optional

import yaml

# Up to 26 cars (A..Z); racecar_gym colors cycle, which is fine for our use.
AGENT_IDS: List[str] = [chr(ord("A") + i) for i in range(26)]

DEFAULT_SENSORS = ["lidar", "pose", "velocity", "acceleration"]


def generate_scenario(
    track: str,
    num_agents: int,
    laps: int,
    time_limit: float,
    out_path: str,
    sensors: Optional[List[str]] = None,
    terminate_on_collision: bool = False,
) -> str:
    """Write a scenario YAML for ``num_agents`` cars on ``track`` and return the path."""
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")
    if num_agents > len(AGENT_IDS):
        raise ValueError(f"at most {len(AGENT_IDS)} agents supported")
    sensors = sensors or DEFAULT_SENSORS

    agents = []
    for i in range(num_agents):
        agents.append({
            "id": AGENT_IDS[i],
            # actuators default to [steering, motor] via VehicleSpec -> action = {steering, motor}
            "vehicle": {"name": "racecar", "sensors": list(sensors)},
            "task": {
                "task_name": "maximize_progress",
                "params": {
                    "laps": int(laps),
                    "time_limit": float(time_limit),
                    "terminate_on_collision": bool(terminate_on_collision),
                },
            },
        })
    spec = {"world": {"name": track}, "agents": agents}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return out_path
