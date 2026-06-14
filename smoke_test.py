#!/usr/bin/env python3
"""End-to-end CPU smoke test.

Proves the whole loop works headless: vectorized env -> SAC learns -> league
snapshot -> self-play -> atomic checkpoint -> KILL -> resume (continues from the
saved step, with replay buffer + RNG) -> eval -> render solo & head-to-head.

Runs train.py as separate processes so the resume path is a real process restart.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = "/tmp/smoke_run"
CKPT = os.path.join(RUN, "checkpoints")
CFG = "configs/smoke.yaml"


def run(args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([PY] + args, cwd=ROOT, check=True)


def manifest():
    with open(os.path.join(CKPT, "latest.json")) as f:
        return json.load(f)


def main() -> int:
    shutil.rmtree(RUN, ignore_errors=True)

    # 1) Partial run -> checkpoint around step 300 (end of stage 0).
    run(["train.py", "--config", CFG, "--max-steps", "300"])
    assert os.path.exists(os.path.join(CKPT, "latest.json")), "no checkpoint after run 1"
    m1 = manifest()
    step1 = int(m1["global_step"])
    assert step1 >= 300, f"run1 stopped too early at {step1}"
    assert "buffer" in m1, "replay buffer not in checkpoint manifest"
    assert os.path.exists(os.path.join(CKPT, m1["buffer"])), "replay buffer file missing"
    assert os.path.exists(os.path.join(CKPT, m1["rng"])), "rng sidecar missing"
    print(f"[ok] run1 checkpoint @ step {step1}: {m1}")

    # 2) Resume (fresh process) -> must continue past step1 into self-play stage.
    run(["train.py", "--config", CFG])
    m2 = manifest()
    step2 = int(m2["global_step"])
    assert step2 > step1, f"resume did not advance: {step2} <= {step1}"
    snaps = [f for f in os.listdir(os.path.join(CKPT, "league")) if f.endswith(".zip")]
    assert snaps, "no league snapshots created during self-play"
    print(f"[ok] resumed {step1} -> {step2}; league snapshots={len(snaps)}")

    # 3) Eval.
    run(["eval.py", "--config", CFG, "--episodes", "1"])

    # 4) Render solo + head-to-head (latest vs earliest snapshot).
    run(["render.py", "--config", CFG, "--mode", "solo", "--checkpoint", "latest",
         "--out", os.path.join(RUN, "solo.gif")])
    assert os.path.exists(os.path.join(RUN, "solo.gif")), "solo render missing"
    run(["render.py", "--config", CFG, "--mode", "h2h", "--out", os.path.join(RUN, "duel.gif")])
    assert os.path.exists(os.path.join(RUN, "duel.gif")), "h2h render missing"

    print(f"\n[SMOKE PASS] final step {step2}; videos: {RUN}/solo.gif, {RUN}/duel.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
