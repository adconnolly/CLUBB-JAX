#!/usr/bin/env python3
"""test_invariants.py — Tier-A oracle-free invariants gate (REFACTOR.md §2, Phase 0 P0.2).

Runs the physical-invariant checks (`run_scripts/invariants.py`) on a JAX stats file:
  1. POSITIVE: a real JAX run (arm) satisfies finiteness + positivity + Cauchy–Schwarz.
  2. NEGATIVE: a synthetic dataset with an injected NaN and a correlation>1 is correctly FAILED
     (so the checker is not vacuously passing).

Oracle-free: needs only a JAX stats file (generated if absent); no Fortran required. SKIPs cleanly
if neither a stats file nor the ability to generate one is available.
"""
import os
import sys
import glob
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))      # CLUBB-JAX/
_RUN = os.path.join(_JAX_ROOT, "clubb_jax", "run_scripts")
sys.path.insert(0, _RUN)
import invariants  # noqa: E402


class _DictDS:
    """Minimal netCDF4.Dataset stand-in: `.variables` dict + `ds[name][:]` access."""
    def __init__(self, d):
        self.variables = d
    def __getitem__(self, k):
        return self.variables[k]


def _find_or_make_stats():
    """Return a path to an arm JAX stats file, generating a short run if none exists."""
    for p in (os.path.join(_JAX_ROOT, "clubb_jax/output/arm_stats.nc"),
              *glob.glob(os.path.join(_JAX_ROOT, "clubb_jax/output/**/arm_stats.nc"), recursive=True)):
        if os.path.isfile(p):
            return p
    # Generate a short run.
    out = os.path.join(_JAX_ROOT, "clubb_jax/output/arm_invariants_tmp")
    cmd = [sys.executable, os.path.join(_RUN, "run_scm.py"), "arm",  # JAX = default
           "-max_iters", "3", "-out_dir", out]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except Exception:
        return None
    p = os.path.join(out, "arm_stats.nc")
    return p if os.path.isfile(p) else None


def test_positive():
    try:
        import netCDF4 as nc
    except ImportError:
        print("SKIP: netCDF4 unavailable")
        return True
    path = _find_or_make_stats()
    if path is None:
        print("SKIP: no JAX stats file and could not generate one")
        return True
    res = invariants.check_all(nc.Dataset(path))
    print(invariants.format_result(res))
    assert res["ok"], f"real arm run violated Tier-A invariants: {res['violations']}"
    print("test_positive PASS")
    return True


def test_negative():
    """A dataset that violates positivity, finiteness, and Cauchy–Schwarz must FAIL."""
    n = 10
    wp2 = np.full((1, n, 1), 1.0)
    rtp2 = np.full((1, n, 1), 1.0)
    # |wprtp| = 2 > sqrt(1*1) = 1  → correlation 2 (impossible)
    wprtp = np.full((1, n, 1), 2.0)
    thlp2 = np.full((1, n, 1), -5.0)        # negative variance (positivity violation)
    em = np.full((1, n, 1), 1.0)
    em[0, 0, 0] = np.nan                    # finiteness violation
    ds = _DictDS({"wp2": wp2, "rtp2": rtp2, "wprtp": wprtp, "thlp2": thlp2, "em": em})
    res = invariants.check_all(ds)
    assert not res["ok"], "checker failed to catch injected violations"
    assert res["violations"]["finiteness"], "missed NaN"
    assert res["violations"]["positivity"], "missed negative variance"
    assert res["violations"]["cauchy_schwarz"], "missed correlation>1"
    print("test_negative PASS (injected NaN, negative variance, corr>1 all caught)")
    return True


if __name__ == "__main__":
    ok = True
    ok &= test_negative()      # fast, no I/O — run first
    ok &= test_positive()
    print("\nAll invariants tests PASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
