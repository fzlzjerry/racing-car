"""Agents: SAC factory, self-play league (PFSP), and frozen opponents."""
from .sac_factory import build_sac, ALGO_REGISTRY  # noqa: F401
from .league import League, LeagueOpponentProvider  # noqa: F401
from .opponent import OpponentPolicy, ConstantOpponent  # noqa: F401
