# SETUP — choices, install, and Colab gotchas

This document explains *why* the stack is what it is, how to install it on a
headless Colab (Pro+) runtime, and the concrete traps that were hit and solved
while building it. If you just want to run, see the **Quick install** section
and `train.ipynb`.

---

## 1. Simulator choice — racecar_gym

**Picked: [`racecar_gym`](https://github.com/axelbr/racecar_gym)** (PyBullet, F1Tenth-like).

| Candidate | Verdict | Why |
|---|---|---|
| **racecar_gym** | ✅ chosen | Native multi-agent (1..N cars on a track), headless by default (PyBullet **DIRECT**, no display), Gymnasium API, lidar+pose observations, and — critically — its per-agent state exposes everything reward shaping needs: continuous centerline `progress`∈[0,1], `lap`, `time`, `wall_collision`, `opponent_collisions`, `wrong_way`, `rank`. |
| f1tenth_gym | ⛳ fallback | Good physics, but its Gymnasium migration was still in flight (legacy `f110-v0` API in the docs as of mid-2025), `info` is empty, and there is **no built-in centerline progress** (you'd project onto a waypoint centerline yourself). Wired as the `F1TenthAdapter` stub behind `envs/adapter.py`. |
| CARLA | ❌ | Unreal Engine; heavy off-screen GPU rendering, large server, fragile on a single Colab GPU. Wrong fit. |
| GPUDrive | ❌ | Blazing fast and GPU-native, but it's **Waymo traffic-navigation**, not wheel-to-wheel racing (no racing line / lap time / overtaking task). |

**It is CPU-physics.** PyBullet steps on the CPU, so you *cannot* saturate an
80–96 GB GPU by spawning more envs (you'd just oversubscribe the cores). That is
exactly why the algorithm is **SAC with a high update-to-data ratio** (below).

### racecar_gym is unmaintained — two patches were required
Last upstream commit is **2023-09-18**, and it does **not** install/run on
Python 3.11 as-is. `scripts/install_racecar_gym.py` handles both automatically:

1. **`numpy==1.22.3` pin has no cp311 wheel** → we install our own deps and add
   the package with `--no-deps` (see the lockfile rationale below).
2. **Dataclass `mutable default` (`ValueError: ... use default_factory`)** in
   `racecar_gym/core/specs.py` — Python 3.11 forbids dataclass-instance field
   defaults. The installer rewrites the two offending fields to
   `field(default_factory=...)`.

Tracks (Austria, Berlin, Montreal, Torino, Circle, Plechaty, …) **auto-download
on first use** from the `tracks-v1.0.0` GitHub release into the package's
`models/scenes/`. The first run needs network; cache the package dir to Drive on
Colab to avoid re-downloading.

---

## 2. Algorithm choice — SAC (SB3 default, SBX opt-in)

**SAC** (off-policy) is the right tool for a CPU-bound sim: you decouple GPU load
from env throughput by doing many gradient steps per env step (`gradient_steps`,
the **update-to-data ratio**) on a large replay buffer with big nets and big
batches. That is how a 96 GB GPU stays busy here, not env parallelism.

- **Default backend: Stable-Baselines3 (PyTorch), SB3 `2.0.0`.** Robust, and it
  is the SB3 release that targets gymnasium `0.28.1` — a *clean version match*
  with racecar_gym, no shim.
- **Opt-in "fast mode": SBX (SB3 + JAX)** exposing `SAC / CrossQ / TQC / DroQ`
  for high-UTD training (`agent.backend=sbx agent.algo=droq`). SBX mirrors the
  SB3 save/load/learn API, so the checkpoint/resume code is identical. Enable it
  with `pip install sbx-rl` + a CUDA JAX (see §5). DroQ/CrossQ are specifically
  designed to tolerate UTD ratios of 10–20, which is what keeps the GPU pinned.

GPU-saturation knobs live in `configs/config.yaml > agent` and are auto-scaled to
detected VRAM by `training/autoscale.py` (`hardware.vram_profiles`): `net_arch`,
`batch_size`, `buffer_size`, `gradient_steps`, and `n_envs` (capped at CPU count).

---

## 3. The dependency lockfile (and why every pin matters)

Installing this stack is a real-world dependency-resolution puzzle; the
**validated, mutually-compatible set** is in `requirements.txt`. The traps:

- **`numpy==1.23.5` (must be <1.24).** racecar_gym uses the removed `np.float`
  / `np.int` aliases in nptyping annotations (evaluated at import), so numpy
  1.24+ breaks it. 1.23.5 is the newest numpy with cp311 wheels that still has
  those aliases.
- **`pandas==2.0.3`, `matplotlib==3.7.5`, `contourpy==1.1.1`.** SB3 pulls pandas
  + matplotlib; their *latest* releases demand numpy ≥1.25/1.26 and silently
  upgrade numpy out from under racecar_gym. Pinned to their last numpy-1.23-
  compatible versions.
- **`gymnasium==0.28.1`** — the exact version both racecar_gym and SB3 2.0.0
  target.
- **`setuptools<80`** — newer setuptools dropped `pkg_resources`, which
  `imageio-ffmpeg==0.4.9` imports.
- **torch is not pinned in requirements.txt** — install the right build first
  (CPU vs CUDA), then `pip install -r requirements.txt`.

> Install everything in **one** `pip` pass (as the notebook does) so the resolver
> sees all the pins together — installing piecemeal lets it re-upgrade numpy.

---

## 4. Quick install (headless Linux / Colab)

```bash
# (Colab usually ships a CUDA torch; otherwise install one — see §5)
pip install -r requirements.txt
python scripts/install_racecar_gym.py     # clone @ pinned SHA, patch py3.11, install --no-deps
python scripts/install_racecar_gym.py --no-install   # (re)apply patch only

# Prove the whole loop end-to-end on CPU (a few hundred steps):
python smoke_test.py
```

No apt packages are needed for **headless DIRECT rendering** (software
TinyRenderer). For **EGL GPU-accelerated** offscreen rendering you need the
NVIDIA EGL/GL userspace libs present (they are, on Colab GPU runtimes).

---

## 5. Colab gotchas

- **Headless rendering works with no GPU.** PyBullet in DIRECT mode renders
  `rgb_array_follow` frames via the software TinyRenderer — verified producing
  real frames on a CPU-only box. `render.py` encodes them with imageio's
  **bundled ffmpeg** (`imageio-ffmpeg`), so you don't need system ffmpeg/apt.
  On a GPU runtime you can switch to EGL for faster recording.
- **Drive checkpoint cadence.** The replay buffer is the big file. With lidar
  subsampled to ~108 beams and `buffer_size≈5e5` it's only a few hundred MB, but
  Drive's FUSE mount is slow, so `checkpoint.buffer_every` is decoupled from
  `checkpoint.save_every` (weights/RNG saved often, buffer rarely). All writes
  are atomic (temp file → `os.replace`) with a `latest.json` manifest written
  last, so a runtime that dies mid-write never corrupts a checkpoint. Point
  `checkpoint.dir` and `logging.tb_dir` at a `/content/drive/...` path.
- **Background execution (Pro+).** Launch training as a subprocess writing logs
  to Drive (the notebook does this with `nohup`) so it survives closing the tab;
  re-running the launch cell auto-resumes from the latest checkpoint.
- **SBX / JAX on different GPUs.** `pip install "jax[cuda12]"` (needs CUDA ≥12.1,
  cuDNN ≥9.8). **A100 (sm_80)** works out of the box. **RTX 6000 Pro / Blackwell
  (sm_120)** needs a *recent* jaxlib — older builds throw "No visible GPU
  devices"; pin the newest jax and confirm `jax.devices()` shows the GPU.
- **GPU auto-scaling.** `hardware.auto_scale=true` picks the largest
  `vram_profiles` entry ≤ detected VRAM. An A100 80 GB and a 96 GB Blackwell get
  progressively larger nets/batches/buffers/UTD automatically.
- **Determinism is best-effort at the env level.** We checkpoint and restore
  python/numpy/torch (or JAX-key) RNG, but PyBullet has its own internal
  nondeterminism, so exact bit-reproducibility across a resume is not guaranteed
  (training continuity and the step counter are).
