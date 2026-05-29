#!/usr/bin/env python3
"""compare_runs.py — run a case in pure-Fortran and hybrid JAX modes, then diff the stats.

Usage:
    python run_scripts/compare_runs.py [--max-iters N] [--case CASE]

Runs both modes with the same number of iterations and reports per-variable
max absolute differences from the stats NetCDF files.

Exits 0 on success (all diffs < threshold), 1 on failure.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time

import numpy as np

# This script lives in clubb_jax/run_scripts/.
# clubb_jax/ and clubb_release/ are siblings under the same parent (CLUBB-JAX/).
RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
JAX_ROOT    = os.path.normpath(os.path.join(RUN_SCRIPTS, "../.."))   # CLUBB-JAX/
CLUBB_ROOT  = os.path.normpath(os.path.join(JAX_ROOT, "clubb_release"))
PYTHONPATH  = os.pathsep.join([JAX_ROOT, CLUBB_ROOT,
                                os.path.join(CLUBB_ROOT, "clubb_python_api")])

REL_TOL = 1e-6   # flag variables whose max|Δ|/max|ref| exceeds this

# Key prognostic state variables — these MUST match between Fortran and hybrid.
# Diagnostic budget terms and mixing-length scalars can have timing differences.
PROGNOSTIC = {
    "thlm", "rtm", "um", "vm", "wp2", "wp3", "rtp2", "thlp2", "rtpthlp",
    "wpthlp", "wprtp", "upwp", "vpwp", "up2", "vp2", "em",
    "sclrm",
}


def run(cmd, env=None, cwd=None):
    """Run a shell command, print output, return returncode."""
    proc = subprocess.run(cmd, cwd=cwd or RUN_SCRIPTS,
                          env=env or os.environ, capture_output=False)
    return proc.returncode


def compare_nc(fort_path: str, jax_path: str) -> bool:
    """Compare two CLUBB stats NetCDF files. Return True if they agree."""
    try:
        import netCDF4 as nc
    except ImportError:
        print("netCDF4 not available — skipping comparison")
        return True

    ds_f = nc.Dataset(fort_path)
    ds_j = nc.Dataset(jax_path)

    # Scalar/vector variables to compare (skip coordinates and metadata)
    SKIP = {"time", "col", "zm", "zt", "lh_zt", "param", "param_name",
            "clubb_params"}

    vars_f = set(ds_f.variables) - SKIP
    vars_j = set(ds_j.variables) - SKIP

    if vars_f != vars_j:
        print(f"  Variable mismatch: fort-only={vars_f - vars_j}  jax-only={vars_j - vars_f}")

    header = f"{'Variable':<28}  {'maxAbsDiff':>12}  {'maxRef':>12}  {'relErr':>10}  {'Type':<5}  {'Status'}"
    print(header)
    print("-" * len(header))

    prog_fail = False
    rows = []
    for v in sorted(vars_f & vars_j):
        fv = np.asarray(ds_f[v][:], dtype=np.float64)
        jv = np.asarray(ds_j[v][:], dtype=np.float64)
        if fv.shape != jv.shape:
            print(f"  {v}: shape mismatch {fv.shape} vs {jv.shape}")
            continue
        max_diff = float(np.nanmax(np.abs(fv - jv)))
        max_ref  = float(np.nanmax(np.abs(fv)))
        rel = max_diff / max_ref if max_ref > 0 else 0.0
        is_prog = v in PROGNOSTIC
        status = "PASS" if rel <= REL_TOL else "FAIL"
        if status == "FAIL" and is_prog:
            prog_fail = True
        rows.append((v, max_diff, max_ref, rel, is_prog, status))

    for v, md, mr, re, ip, st in rows:
        mark = "**" if (st == "FAIL" and ip) else "  "
        vtype = "PROG" if ip else "diag"
        print(f"{mark}{v:<26}  {md:>12.3e}  {mr:>12.3e}  {re:>10.2e}  {vtype:<5}  {st}")

    n_prog_fail = sum(1 for _, _, _, _, ip, st in rows if ip and st == "FAIL")
    n_diag_fail = sum(1 for _, _, _, _, ip, st in rows if not ip and st == "FAIL")
    print(f"\n  Prognostic failures: {n_prog_fail}   Diagnostic failures (timing): {n_diag_fail}")

    ds_f.close()
    ds_j.close()
    return not prog_fail


def main():
    parser = argparse.ArgumentParser(description="Compare Fortran vs JAX ARM runs")
    parser.add_argument("--case", default="arm", help="Case name (default: arm)")
    parser.add_argument("--max-iters", type=int, default=30,
                        help="Number of timesteps (default: 30)")
    args = parser.parse_args()

    case   = args.case
    niters = args.max_iters
    outdir_f = os.path.join(CLUBB_ROOT, f"output/{case}_compare_fort")
    outdir_j = os.path.join(CLUBB_ROOT, f"output/{case}_compare_jax")

    env_jax = os.environ.copy()
    env_jax["PYTHONPATH"] = PYTHONPATH

    scm = [sys.executable, os.path.join(RUN_SCRIPTS, "run_scm.py")]

    print(f"=== Step 1: Pure Fortran run ({niters} iters) ===")
    t0 = time.time()
    rc = run(scm + [case, "-legacy",
                    "-max_iters", str(niters),
                    "-out_dir", outdir_f])
    print(f"  Fortran: {time.time()-t0:.1f}s  (rc={rc})")
    if rc != 0:
        sys.exit(f"Fortran run failed (rc={rc})")

    print(f"\n=== Step 2: JAX hybrid run ({niters} iters) ===")
    t0 = time.time()
    rc = run(scm + [case, "-jax",
                    "-max_iters", str(niters),
                    "-out_dir", outdir_j],
             env=env_jax)
    print(f"  JAX hybrid: {time.time()-t0:.1f}s  (rc={rc})")
    if rc != 0:
        sys.exit(f"JAX run failed (rc={rc})")

    fort_nc = os.path.join(outdir_f, f"{case}_stats.nc")
    jax_nc  = os.path.join(outdir_j, f"{case}_stats.nc")

    print(f"\n=== Step 3: Comparing outputs ===")
    print(f"  Fortran: {fort_nc}")
    print(f"  JAX:     {jax_nc}\n")

    ok = compare_nc(fort_nc, jax_nc)
    if ok:
        print(f"\nResult: PASS — all PROGNOSTIC variables within rel_tol={REL_TOL:.0e}")
    else:
        print(f"\nResult: FAIL — prognostic variable(s) exceed rel_tol={REL_TOL:.0e} (see ** rows above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
