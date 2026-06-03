#!/usr/bin/env python3
"""update_golden.py — (re)baseline Tier-B golden JAX trajectories (REFACTOR.md §2, Phase 0 P0.3).

DELIBERATE, reviewed step (never automatic): runs the JAX driver for a case, stores the stats under
`clubb_jax/output/golden/<case>/` (gitignored), and updates the tracked manifest
(`clubb_jax/golden_manifest.json`) with provenance (steps, git commit) + per-variable checksums.

Usage:
    python run_scripts/update_golden.py --case arm [--max-iters 30]
    python run_scripts/update_golden.py --all  [--max-iters 30]
    python run_scripts/update_golden.py --case arm --seed-from clubb_jax/output/arm_stats.nc  # copy, don't run
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys

import golden as G

# Default cases to baseline with --all: the 18 bit-faithful + the characterized cases.
DEFAULT_CASES = [
    "arm", "bomex", "dycoms2_rf01", "wangara", "gabls2", "gabls3_night", "fire", "neutral",
    "ekman", "cobra", "dycoms2_rf02_nd", "dycoms2_rf01_fixed_sst", "atex_long", "dycoms2_rf02_so",
    "jun25_altocu", "gabls3", "mpace_a", "atex",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=G.JAX_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def baseline_case(case: str, max_iters: int, seed_from: str | None) -> bool:
    import netCDF4 as nc
    dest_dir = os.path.join(G.GOLDEN_DIR, case)
    os.makedirs(dest_dir, exist_ok=True)
    dest = G.golden_stats_path(case)

    if seed_from:
        if not os.path.isfile(seed_from):
            print(f"  {case}: seed file not found: {seed_from}")
            return False
        shutil.copyfile(seed_from, dest)
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [G.JAX_ROOT, G.JAX_ROOT + "/clubb_release", G.JAX_ROOT + "/clubb_release/clubb_python_api"])
        cmd = [sys.executable, os.path.join(G.RUN_SCRIPTS, "run_scm.py"), case, "-jax",
               "-max_iters", str(max_iters), "-out_dir", dest_dir]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0 or not os.path.isfile(dest):
            print(f"  {case}: JAX run failed (rc={rc})")
            return False

    manifest = G.load_manifest()
    ds = nc.Dataset(dest)
    n_records = int(ds.dimensions["time"].size) if "time" in ds.dimensions else None
    manifest[case] = {"max_iters": max_iters, "n_records": n_records,
                      "git_commit": _git_commit(), "vars": G.checksums(ds)}
    ds.close()
    G.save_manifest(manifest)
    print(f"  {case}: golden baselined ({len(manifest[case]['vars'])} vars) → {dest}")
    return True


def main():
    ap = argparse.ArgumentParser(description="(Re)baseline Tier-B golden JAX trajectories.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="single case to baseline")
    g.add_argument("--all", action="store_true", help="baseline all default cases")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--seed-from", default=None,
                    help="copy an existing stats .nc instead of running (single --case only)")
    args = ap.parse_args()

    cases = DEFAULT_CASES if args.all else [args.case]
    if args.seed_from and args.all:
        sys.exit("--seed-from only valid with a single --case")

    ok = True
    for c in cases:
        ok &= baseline_case(c, args.max_iters, args.seed_from)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
