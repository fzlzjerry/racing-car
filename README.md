# 🏎️ Self-Play RL Racing

Wheel-to-wheel racing agents trained by **self-play** on a single big NVIDIA GPU,
headless, fully resumable, built to keep improving for *days* of A100 / RTX 6000
Pro time. The agent learns racing lines, overtaking and defending by racing a
growing league of its own frozen past selves (AlphaStar-lite / PFSP).

- **Sim:** [racecar_gym](https://github.com/axelbr/racecar_gym) (PyBullet, multi-agent, headless)
- **Algo:** SAC (off-policy) on Stable-Baselines3 — or SBX/JAX (CrossQ/DroQ) "fast mode"
- **Curriculum:** time-trial → head-to-head self-play → full grid
- **Resumability:** atomic checkpoints to Google Drive (model + optimizer + replay buffer + step + RNG + curriculum + league)

> Why these choices, and the install gotchas that were solved, are in **[SETUP.md](SETUP.md)**.
> Entry point for Colab is **[`train.ipynb`](train.ipynb)**.

---

## Quickstart

```bash
pip install -r requirements.txt
# torch: CPU -> pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/install_racecar_gym.py    # clone + py3.11 patch + install
python smoke_test.py                     # end-to-end proof on CPU (~1-2 min)

# Real training (auto-detects GPU, auto-scales, auto-resumes):
python train.py checkpoint.dir=/content/drive/MyDrive/racing/ckpts \
                logging.tb_dir=/content/drive/MyDrive/racing/tb \
                env.track=austria

# Visual payoff:
python render.py --mode solo --out outputs/solo.mp4
python render.py --mode h2h  --out outputs/duel.mp4     # latest vs earliest snapshot
python eval.py   --episodes 20
```

Kill it any time — re-running the same `train.py` command resumes from the
latest checkpoint seamlessly.

---

## Repository layout

```
envs/        sim adapter, obs/reward construction, the self-play env, vec-env factory
  adapter.py            racecar_gym specifics isolated here (f1tenth = stub)
  observations.py       Dict sensor obs -> flat, pre-normalized Box (lidar subsampled)
  reward.py             configurable reward from progress/speed/collision/overtake/...
  self_play_env.py      single-agent learner over the multi-agent sim + frozen opponents
  make_env.py           SubprocVecEnv factory (parallel, league-aware)
agents/      sac_factory.py (SB3/SBX), league.py (PFSP), opponent.py (frozen CPU policies)
training/    autoscale.py, checkpoint.py (atomic resume), callbacks.py, curriculum.py
configs/     config.yaml (full defaults) + smoke.yaml (tiny end-to-end)
train.py  eval.py  render.py  smoke_test.py  train.ipynb
scripts/install_racecar_gym.py    SETUP.md  requirements.txt
```

## How self-play works

The learner always drives car **A**. The other cars are driven by **frozen**
policies sampled at each episode from a filesystem-backed **league** of past
snapshots. Opponent inference runs on **CPU inside each env worker**, so the GPU
is reserved for the learner's (big-net, high-UTD) updates. PFSP samples
opponents the learner rarely beats more often — pushing it to actually overtake
and defend rather than exploit one weak past self. New snapshots are frozen into
the league on a cadence, so the opponents keep getting tougher the longer you
train. See `agents/league.py` and `envs/self_play_env.py`.

## Curriculum (configurable in `configs/config.yaml`)

| Stage | Cars | Goal |
|---|---|---|
| 1. time_trial | 1 | racing line, minimize lap time |
| 2. head_to_head | 2 | overtaking & defending vs a sampled past self |
| 3. full_grid | N | racecraft in traffic |

Each stage resumes from the previous stage's policy; the league persists across
stages. Advance by step budget or a performance threshold.

## Reward (all weights in `configs/reward` / `config.yaml > reward`)

progress along the centerline + speed; heavy penalties for collisions, going the
wrong way, and reversing; per-step time penalty; action-smoothness; **+overtake**
(rank improvement) and **+finish-position** bonuses.

## Metrics (TensorBoard, to Drive)

`eval/best_lap_time`, `eval/median_lap_time`, `eval/laps_completed_frac`,
`eval/win_rate_vs_league`, `racing/collision_rate`, `racing/ep_total_progress`,
`racing/mean_rank`, `league/size`, `perf/sps`, `perf/gpu_util`.

## "More compute = better"

Large nets + big batches + high update-to-data ratio keep the GPU busy despite a
CPU-bound sim; a league that keeps producing tougher opponents and a curriculum
that escalates car count mean the task difficulty rises *with* training — so it
should still be improving after 24h, not plateau in 30 minutes. All of it
auto-scales to the detected GPU (`hardware.vram_profiles`).
```
