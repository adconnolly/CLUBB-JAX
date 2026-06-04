#!/usr/bin/env python3
"""test_simple_rad_lba.py — validate the LBA prescribed-radiation port against the real data + a literal loop."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.Radiation.simple_rad_lba import (
    load_lba_rad_table, lba_radhtz, simple_rad_lba, _LBA_DT, _LBA_NTIMES, _LBA_NZRAD)
from clubb_jax.src.Benchmark_cases.generic_forcings import _zlinterp

_FORCINGS = os.path.join(_ROOT, "clubb_release", "input", "case_setups", "lba_forcings")


def _ref_radhtz(time, krad):
    """Literal transcription of the simple_rad_lba time interpolation (Fortran)."""
    if time <= _LBA_DT:
        return krad[:, 0].copy()
    if time >= _LBA_NTIMES * _LBA_DT:
        return krad[:, _LBA_NTIMES - 1].copy()
    i1 = 1
    while i1 <= _LBA_NTIMES - 1:
        i2 = i1 + 1
        if _LBA_DT * i1 <= time < _LBA_DT * i2:
            a = (time - _LBA_DT * i1) / (_LBA_DT * i2 - _LBA_DT * i1)
            return a * (krad[:, i2 - 1] - krad[:, i1 - 1]) + krad[:, i1 - 1]   # 1-based -> 0-based
        i1 = i2
    return krad[:, -1].copy()


def test_table_load():
    zrad, krad = load_lba_rad_table(_FORCINGS)
    assert zrad.shape == (_LBA_NZRAD,) and krad.shape == (_LBA_NZRAD, _LBA_NTIMES)
    assert zrad[0] == 42.5 and abs(krad[0, 0] - (-0.000016042)) < 1e-15, "first values mismatch"
    assert np.all(np.diff(zrad) > 0), "heights not ascending"
    # Physically reasonable radiative tendencies (mostly cooling, |rate| < ~4 K/day).
    assert np.all(np.abs(krad) < 5e-5) and (krad <= 0).mean() > 0.5, "heating rates out of physical range"
    print(f"  load_lba_rad_table: zrad{zrad.shape} krad{krad.shape}, zrad[0]={zrad[0]}, krad[0,0]={krad[0,0]:.3e}  PASS")


def test_time_interp_vs_literal():
    _, krad = load_lba_rad_table(_FORCINGS)
    worst = 0.0
    for time in (0.0, 300.0, 600.0, 601.0, 900.0, 1500.0, 12345.0, 21000.0, 21600.0, 30000.0):
        got = np.asarray(lba_radhtz(time, krad))
        ref = _ref_radhtz(time, krad)
        worst = max(worst, np.max(np.abs(got - ref)))
    assert worst < 1e-15, f"lba_radhtz vs literal {worst:.2e}"
    print(f"  lba_radhtz vs literal time-interp (10 times incl. boundaries): max diff {worst:.1e}  PASS")


def test_full_radht_vs_literal():
    zrad, krad = load_lba_rad_table(_FORCINGS)
    rng = np.random.default_rng(1)
    zt = np.sort(rng.uniform(0.0, 18000.0, 60))   # thermodynamic grid spanning + beyond the rad table
    worst = 0.0
    for time in (300.0, 1500.0, 12345.0, 25000.0):
        got = simple_rad_lba(time, zrad, krad, zt)
        ref = _zlinterp(zt, zrad, _ref_radhtz(time, krad))
        worst = max(worst, np.max(np.abs(got - ref)))
    # zero-extrapolation above the table top
    rad = simple_rad_lba(1500.0, zrad, krad, zt)
    assert np.all(rad[zt > zrad[-1]] == 0.0), "should be 0 above the rad table top"
    assert worst < 1e-15, f"simple_rad_lba vs literal {worst:.2e}"
    print(f"  simple_rad_lba (time-interp + vertical zlinterp) vs literal: max diff {worst:.1e}; zero-extrap top  PASS")


def test_differentiable():
    _, krad = load_lba_rad_table(_FORCINGS)
    # differentiable w.r.t. the table values (linear interpolation)
    g = jax.grad(lambda k: jnp.sum(lba_radhtz(1500.0, k) ** 2))(jnp.asarray(krad))
    assert np.isfinite(np.asarray(g)).all() and np.any(np.asarray(g) != 0.0), "grad bad"
    print("  jax.grad(lba_radhtz) wrt the heating table: finite, nonzero  PASS")


def main():
    print("test_simple_rad_lba:")
    for t in (test_table_load, test_time_interp_vs_literal, test_full_radht_vs_literal, test_differentiable):
        t()
    print("All simple_rad_lba checks PASSED")


if __name__ == "__main__":
    main()
