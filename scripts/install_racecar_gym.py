#!/usr/bin/env python3
"""Reproducible installer for racecar_gym on a modern (Python 3.11) stack.

racecar_gym (https://github.com/axelbr/racecar_gym) is unmaintained since
2023-09 and does NOT install/run cleanly on Python 3.11:

  * Its ``requirements.txt`` pins ``numpy==1.22.3`` which has no cp311 wheel,
    so we install our own (relaxed) deps first and add the package with
    ``--no-deps``.
  * ``racecar_gym/core/specs.py`` uses *dataclass instances* as field
    defaults (``vehicle: VehicleSpec = VehicleSpec()``).  Python 3.11 rejects
    this with ``ValueError: mutable default ... use default_factory``.  We
    patch those two fields to use ``default_factory``.

This script is idempotent: re-running it re-clones at the pinned commit,
re-applies the patch, and re-installs.  It is invoked by ``train.ipynb`` and
documented in ``SETUP.md``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Last commit on master (2023-09-18). The ``gym-api`` *tag* is the OLD pre-
# gymnasium API -- do not use it. This SHA is the gymnasium-compatible HEAD.
PINNED_SHA = "70496767d80397010ac9b5ba722647ca67e3f85c"
REPO_URL = "https://github.com/axelbr/racecar_gym.git"

SPECS_REL = "racecar_gym/core/specs.py"
PATCHES = [
    ("vehicle: VehicleSpec = VehicleSpec()",
     "vehicle: VehicleSpec = field(default_factory=VehicleSpec)"),
    ("task: TaskSpec = TaskSpec()",
     "task: TaskSpec = field(default_factory=TaskSpec)"),
]


def run(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, **kw)


def patch_specs(src: Path) -> None:
    """Apply the Python-3.11 ``default_factory`` fix (idempotent)."""
    p = src / SPECS_REL
    text = p.read_text()
    changed = False
    for old, new in PATCHES:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        p.write_text(text)
        print(f"patched {SPECS_REL} for Python 3.11 (default_factory)")
    else:
        print(f"{SPECS_REL} already patched (or upstream changed) -- skipping")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="/tmp/racecar_gym_src",
                    help="where to clone the source")
    ap.add_argument("--sha", default=PINNED_SHA, help="commit to pin")
    ap.add_argument("--no-install", action="store_true",
                    help="clone + patch only, skip pip install")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not (dest / ".git").exists():
        run(["git", "clone", "--quiet", REPO_URL, str(dest)])
    run(["git", "-C", str(dest), "checkout", "--quiet", args.sha])
    patch_specs(dest)

    if not args.no_install:
        # --no-deps: our requirements.txt already provides a py3.11-compatible
        # superset (relaxed numpy, matching gymnasium/pettingzoo).
        run([sys.executable, "-m", "pip", "install", "-e", str(dest), "--no-deps"])
        # Smoke-check the import path that previously failed.
        run([sys.executable, "-c",
             "import racecar_gym.envs.gym_api; print('racecar_gym import OK')"])
    print("\nracecar_gym ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
