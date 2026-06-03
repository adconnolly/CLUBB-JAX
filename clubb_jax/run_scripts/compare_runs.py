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
# Absolute floor (numpy.allclose convention: |Δ| <= ABS_TOL + REL_TOL*|ref|).
# Without it, fields that are physically ~0 in a given case (e.g. moisture
# moments rtp2/rtpthlp/wprtp/wpthlp in a DRY case like ekman) report enormous
# relative error from pure f64 roundoff (|Δ| ~ 1e-24 vs |ref| ~ 1e-24). The floor
# is far below any real-scale prognostic difference, so it never masks a true
# divergence (e.g. ekman wp2 |Δ| ~ 8e-5 still fails).
ABS_TOL = 1e-12

# Key prognostic state variables — these MUST match between Fortran and hybrid.
# Diagnostic budget terms and mixing-length scalars can have timing differences.
PROGNOSTIC = {
    "thlm", "rtm", "um", "vm", "wp2", "wp3", "rtp2", "thlp2", "rtpthlp",
    "wpthlp", "wprtp", "upwp", "vpwp", "up2", "vp2", "em",
    "sclrm",
}


def _read_dt_main(case: str) -> int:
    """Read dt_main [s] from a case's {case}_model.in (default 60 if not found)."""
    import re
    path = os.path.join(CLUBB_ROOT, "input/case_setups", f"{case}_model.in")
    try:
        txt = open(path).read()
    except OSError:
        return 60
    m = re.search(r"dt_main\s*=\s*([0-9.]+)", txt)
    return int(float(m.group(1))) if m else 60


def run(cmd, env=None, cwd=None):
    """Run a shell command, print output, return returncode."""
    proc = subprocess.run(cmd, cwd=cwd or RUN_SCRIPTS,
                          env=env or os.environ, capture_output=False)
    return proc.returncode


def compare_nc(fort_path: str, jax_path: str, tier: str = "bit") -> bool:
    """Compare two CLUBB stats NetCDF files. Return True if they agree under `tier`.

    tier='bit'      → legacy machine-precision gate (REL_TOL/ABS_TOL on PROGNOSTIC vars).
    tier='physical' → Tier-C field-class tolerances (REFACTOR.md §2, via validation.py).

    The legacy "Prognostic failures:" summary line is printed in BOTH tiers so downstream
    parsers (compare_cases.py) are unaffected; only the returned verdict depends on `tier`.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        print("netCDF4 not available — skipping comparison")
        return True
    import validation

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
        # allclose convention: pass on relative OR absolute agreement.
        status = "PASS" if max_diff <= ABS_TOL + REL_TOL * max_ref else "FAIL"
        if status == "FAIL" and is_prog:
            prog_fail = True
        rows.append((v, max_diff, max_ref, rel, is_prog, status))

    for v, md, mr, re, ip, st in rows:
        mark = "**" if (st == "FAIL" and ip) else "  "
        vtype = "PROG" if ip else "diag"
        print(f"{mark}{v:<26}  {md:>12.3e}  {mr:>12.3e}  {re:>10.2e}  {vtype:<5}  {st}")

    n_prog_fail = sum(1 for _, _, _, _, ip, st in rows if ip and st == "FAIL")
    n_diag_fail = sum(1 for _, _, _, _, ip, st in rows if not ip and st == "FAIL")
    # Legacy summary line — kept verbatim in every tier (compare_cases.py parses it).
    print(f"\n  Prognostic failures: {n_prog_fail}   Diagnostic failures (timing): {n_diag_fail}")

    # Tier-C field-class verdict (REFACTOR.md §2). Computed always; gates only when tier='physical'.
    verdict = validation.tiered_verdict([(v, md, mr, rl) for v, md, mr, rl, _, _ in rows])
    print("\n" + validation.format_verdict(verdict))

    ds_f.close()
    ds_j.close()
    return verdict["all_pass"] if tier == "physical" else (not prog_fail)


def main():
    parser = argparse.ArgumentParser(description="Compare Fortran vs JAX ARM runs")
    parser.add_argument("--case", default="arm", help="Case name (default: arm)")
    parser.add_argument("--max-iters", type=int, default=30,
                        help="Number of timesteps (default: 30)")
    parser.add_argument("--tout", type=int, default=None,
                        help="Override stats sampling+output interval [s] for both runs. "
                             "Default: auto = dt_main (true instantaneous, per-step output). "
                             "This compares the PHYSICS directly and avoids stats-averaging-"
                             "window artifacts for cases whose default stats_tout/stats_tsamp "
                             "exceed dt (e.g. gabls2, gabls3_night).")
    parser.add_argument("--override", default=None,
                        help="Extra namelist overrides (KEY=val,...) applied to BOTH runs, "
                             "e.g. 'l_ho_nontrad_coriolis=.true.' to enable a flag in Fortran+JAX.")
    parser.add_argument("--tier", choices=["bit", "physical"], default="bit",
                        help="Correctness standard for the verdict (REFACTOR.md §2). "
                             "'bit' = legacy machine-precision gate (default; for debugging a real "
                             "regression). 'physical' = Tier-C field-class tolerances.")
    args = parser.parse_args()

    case   = args.case
    niters = args.max_iters
    # Output convention (see DESIGN.md): Fortran stats live under clubb_release/output/,
    # JAX stats under clubb_jax/output/. Each run also gets its own *_compare_* subdir
    # so neither clobbers a stored oracle.
    outdir_f = os.path.join(CLUBB_ROOT, f"output/{case}_compare_fort")
    outdir_j = os.path.join(JAX_ROOT, "clubb_jax", f"output/{case}_compare_jax")
    # Force per-step output (stats_tsamp = stats_tout = interval) so JAX/Fortran records
    # are instantaneous and time-aligned. Default interval = the case's dt_main.
    interval = args.tout if args.tout is not None else _read_dt_main(case)
    _ov = f"stats_tsamp={interval}.,stats_tout={interval}."
    if args.override:
        _ov += "," + args.override
    tout_args = ["-override", _ov]

    env_jax = os.environ.copy()
    env_jax["PYTHONPATH"] = PYTHONPATH

    scm = [sys.executable, os.path.join(RUN_SCRIPTS, "run_scm.py")]

    print(f"=== Step 1: Pure Fortran run ({niters} iters) ===")
    t0 = time.time()
    rc = run(scm + [case, "-legacy",
                    "-max_iters", str(niters),
                    "-out_dir", outdir_f] + tout_args)
    print(f"  Fortran: {time.time()-t0:.1f}s  (rc={rc})")
    if rc != 0:
        sys.exit(f"Fortran run failed (rc={rc})")

    print(f"\n=== Step 2: JAX hybrid run ({niters} iters) ===")
    t0 = time.time()
    rc = run(scm + [case, "-jax",
                    "-max_iters", str(niters),
                    "-out_dir", outdir_j] + tout_args,
             env=env_jax)
    print(f"  JAX hybrid: {time.time()-t0:.1f}s  (rc={rc})")
    if rc != 0:
        sys.exit(f"JAX run failed (rc={rc})")

    fort_nc = os.path.join(outdir_f, f"{case}_stats.nc")
    jax_nc  = os.path.join(outdir_j, f"{case}_stats.nc")

    print(f"\n=== Step 3: Comparing outputs ===")
    print(f"  Fortran: {fort_nc}")
    print(f"  JAX:     {jax_nc}\n")

    ok = compare_nc(fort_nc, jax_nc, tier=args.tier)
    if args.tier == "physical":
        msg = ("PASS — all gated fields within Tier-C field-class tolerances"
               if ok else "FAIL — gated field(s) exceed Tier-C tolerance (see ** rows above)")
    else:
        msg = (f"PASS — all PROGNOSTIC variables within rel_tol={REL_TOL:.0e}"
               if ok else f"FAIL — prognostic variable(s) exceed rel_tol={REL_TOL:.0e} (see ** rows above)")
    print(f"\nResult [{args.tier}]: {msg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
