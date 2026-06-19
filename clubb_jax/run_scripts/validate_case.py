#!/usr/bin/env python3
"""validate_case.py — one-command tiered correctness verdict for a case (REFACTOR.md §2, Phase 0 P0.5).

Runs a case and reports the combined verdict across the tiers:
  * Tier A (invariants)  — oracle-free, ALWAYS run (finiteness / positivity / Cauchy–Schwarz).
  * Tier B (golden)      — vs the stored golden JAX trajectory (rel 1e-9), if a golden exists.
  * Tier C (vs Fortran)  — field-class physical fidelity, run unless --no-fortran.

Reuses existing tooling rather than re-implementing it (adversarial-review R1): the Fortran+JAX runs
and the Tier-C diff come from `compare_runs.py`; Tier A from `invariants.py`; Tier B from `golden.py`.
When Fortran is unavailable, pass --no-fortran to get the oracle-free Tier A + B verdict.

Usage:
    python run_scripts/validate_case.py --case arm                 # uses golden n_records for --max-iters
    python run_scripts/validate_case.py --case rico --max-iters 30
    python run_scripts/validate_case.py --case arm --no-fortran    # Tier A+B only (no oracle needed)
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN_SCRIPTS)
import golden as G        # noqa: E402
import invariants as INV  # noqa: E402
from compare_runs import compare_output_dirs  # noqa: E402

JAX_ROOT = G.JAX_ROOT
CLUBB_ROOT = os.path.join(JAX_ROOT, "clubb_release")


def _run_jax_only(case: str, niters: int) -> str | None:
    """Run the JAX driver only; return the stats path (or None on failure)."""
    out = os.path.join(JAX_ROOT, "clubb_jax", "output", f"{case}_validate_jax")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([JAX_ROOT, CLUBB_ROOT,
                                         os.path.join(CLUBB_ROOT, "clubb_python_api")])
    cmd = [sys.executable, os.path.join(RUN_SCRIPTS, "run_scm.py"), case, "-jax",
           "-max_iters", str(niters), "-out_dir", out]
    rc = subprocess.run(cmd, env=env).returncode
    p = os.path.join(out, f"{case}_stats.nc")
    return p if (rc == 0 and os.path.isfile(p)) else None


def _run_compare(case: str, niters: int):
    """Run compare_runs.py --tier physical; return (tierc_ok, jax_stats_path)."""
    cmd = [sys.executable, os.path.join(RUN_SCRIPTS, "compare_runs.py"),
           "--case", case, "--max-iters", str(niters), "--tier", "physical"]
    rc = subprocess.run(cmd).returncode
    _, jax_dir = compare_output_dirs(case)
    jax_nc = os.path.join(jax_dir, f"{case}_stats.nc")
    return (rc == 0), (jax_nc if os.path.isfile(jax_nc) else None)


def main():
    ap = argparse.ArgumentParser(description="Tiered correctness verdict for a CLUBB case.")
    ap.add_argument("--case", default="arm")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="timesteps; default = the golden's n_records if a golden exists, else 30.")
    ap.add_argument("--no-fortran", action="store_true",
                    help="skip Tier C (no Fortran oracle needed) — Tier A + B only.")
    args = ap.parse_args()
    case = args.case

    manifest = G.load_manifest()
    niters = args.max_iters or (manifest.get(case, {}).get("n_records") or 30)

    import netCDF4 as nc

    # ── Run + collect the JAX stats and the Tier-C verdict ──
    tierc = None  # None = not run
    if args.no_fortran:
        print(f"=== validate_case {case} (Tier A+B only, {niters} steps) ===")
        jax_nc = _run_jax_only(case, niters)
    else:
        print(f"=== validate_case {case} (Tier A+B+C, {niters} steps) ===")
        tierc, jax_nc = _run_compare(case, niters)
    if jax_nc is None:
        sys.exit(f"FAIL: JAX run produced no stats for {case}")

    # ── Tier A (always) ──
    tier_a = INV.check_all(nc.Dataset(jax_nc))
    # ── Tier B (if golden exists) ──
    tier_b = G.compare_to_golden(case, jax_nc)

    print("\n" + "=" * 60)
    print(f"TIERED VERDICT — {case}")
    print("=" * 60)
    print(f"  Tier A (invariants):  {'PASS' if tier_a['ok'] else 'FAIL'}")
    if not tier_a["ok"]:
        for kind, vs in tier_a["violations"].items():
            if vs:
                print(f"      {kind}: {vs[:3]}")
    if tier_b["reason"]:
        print(f"  Tier B (golden):      SKIP ({tier_b['reason']})")
        tier_b_ok = True  # no golden → not a failure, just not checked
    else:
        print(f"  Tier B (golden):      {'PASS' if tier_b['ok'] else 'FAIL'}  "
              f"(checked {tier_b['n_checked']} vars)")
        if not tier_b["ok"]:
            print(f"      worst: {tier_b['fails'][:3]}")
        tier_b_ok = tier_b["ok"]
    if tierc is None:
        print("  Tier C (vs Fortran):  SKIP (--no-fortran)")
        tier_c_ok = True
    else:
        print(f"  Tier C (vs Fortran):  {'PASS' if tierc else 'FAIL'}  (see compare table above)")
        tier_c_ok = tierc

    ok = tier_a["ok"] and tier_b_ok and tier_c_ok
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
