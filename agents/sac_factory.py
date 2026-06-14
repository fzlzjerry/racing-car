"""Build a SAC learner from config.

Default backend is Stable-Baselines3 (PyTorch) -- robust and a clean version
match with racecar_gym's gymnasium 0.28.1.  The ``sbx`` backend (JAX) is an
opt-in "fast mode" exposing CrossQ / TQC / DroQ for high update-to-data ratios;
it mirrors the SB3 save/load/learn API so the checkpoint code is unchanged.

GPU saturation with a CPU-bound sim comes from net size + batch size + a high
``gradient_steps`` (update-to-data ratio), all set here from (auto-scaled) cfg.
"""
from __future__ import annotations

from typing import Dict

ALGO_REGISTRY = {
    "sb3": ["sac"],
    "sbx": ["sac", "crossq", "tqc", "droq"],
}


def _common_kwargs(cfg, env, device, tensorboard_log, seed) -> Dict:
    a = cfg.agent
    policy_kwargs = {"net_arch": list(a.net_arch)}
    kwargs = dict(
        policy=str(a.policy),
        env=env,
        learning_rate=float(a.learning_rate),
        buffer_size=int(a.buffer_size),
        learning_starts=int(a.learning_starts),
        batch_size=int(a.batch_size),
        tau=float(a.tau),
        gamma=float(a.gamma),
        train_freq=int(a.train_freq),
        gradient_steps=int(a.gradient_steps),
        ent_coef=a.ent_coef if isinstance(a.ent_coef, (int, float)) else str(a.ent_coef),
        target_entropy=a.target_entropy if isinstance(a.target_entropy, (int, float)) else str(a.target_entropy),
        use_sde=bool(a.use_sde),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device=device,
        seed=seed,
        verbose=0,
    )
    # optimize_memory_usage can't coexist with handle_timeout_termination in SB3.
    if bool(a.optimize_memory_usage):
        kwargs["optimize_memory_usage"] = True
        kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
    return kwargs


def _algo_class(backend: str, algo: str):
    backend, algo = backend.lower(), algo.lower()
    if backend == "sbx":
        if algo in ("sac", "droq"):
            from sbx import SAC as C
        elif algo == "crossq":
            from sbx import CrossQ as C
        elif algo == "tqc":
            from sbx import TQC as C
        else:
            raise ValueError(f"Unknown sbx algo '{algo}'")
        return C
    from stable_baselines3 import SAC as C
    return C


def load_sac(cfg, path: str, env, device: str = "cpu"):
    """Reload a SAC learner for resume (restores weights, optimizer, num_timesteps)."""
    cls = _algo_class(str(cfg.agent.backend), str(cfg.agent.algo))
    return cls.load(path, env=env, device=device)


def build_sac(cfg, env, device: str = "cpu", tensorboard_log=None, seed=None):
    backend = str(cfg.agent.backend).lower()
    algo = str(cfg.agent.algo).lower()
    kwargs = _common_kwargs(cfg, env, device, tensorboard_log, seed)

    if backend == "sbx":
        return _build_sbx(algo, cfg, kwargs)
    if backend != "sb3":
        raise ValueError(f"Unknown agent.backend '{backend}' (expected sb3|sbx)")
    if algo != "sac":
        raise ValueError(f"SB3 backend only supports algo 'sac' (got '{algo}'). Use backend=sbx for {algo}.")
    from stable_baselines3 import SAC
    return SAC(**kwargs)


def _build_sbx(algo: str, cfg, kwargs):
    a = cfg.agent
    if algo in ("droq",):
        # DroQ = SAC + dropout + layer norm + (usually) higher critic lr & policy delay.
        from sbx import SAC
        kwargs["policy_kwargs"] = {**kwargs["policy_kwargs"],
                                   "dropout_rate": float(a.dropout_rate),
                                   "layer_norm": bool(a.layer_norm)}
        kwargs["policy_delay"] = int(a.policy_delay)
        kwargs["qf_learning_rate"] = float(a.qf_learning_rate)
        return SAC(**kwargs)
    if algo == "crossq":
        from sbx import CrossQ
        return CrossQ(**kwargs)
    if algo == "tqc":
        from sbx import TQC
        return TQC(**kwargs)
    if algo == "sac":
        from sbx import SAC
        return SAC(**kwargs)
    raise ValueError(f"Unknown sbx algo '{algo}' (expected one of {ALGO_REGISTRY['sbx']})")
