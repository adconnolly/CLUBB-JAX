#!/usr/bin/env python3
"""test_remapping_module.py — validate the JAX mass-conserving remapping port (remapping_module.F90).

Ullrich-linear conservative vertical remapping. Oracles:
  1. f2py bit-shadow vs f2py_remap_vals_to_target_same_grid — source grid == target grid, so the remap is the
     identity and the output must equal source_values; this exercises the whole mass→pressure→matrix→matvec
     pipeline against the Fortran. SKIPs if clubb_f2py/clubb_python unbuilt.
  2. remapping_matrix identity (same levels) + row-conservation; calc_mass_over_grid_intervals vs the analytic
     integral of a known linear density; matrix_vector_mult vs einsum; and a finite jax.grad.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for p in (_ROOT + "/clubb_release", _ROOT + "/clubb_release/clubb_python_api"):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.remapping_module import (
    calc_mass_over_grid_intervals, remapping_matrix, matrix_vector_mult, remap_vals_to_target_same_grid)
from clubb_jax.src.derived_types.grid_class import setup_grid

_NG, _DZ, _ZTOP = 2, 50.0, 1500.0


def _rho_spline(ng, ztop):
    # Density spline covering [0, ztop+pad], decreasing with height; ascending levels.
    levels = np.linspace(0.0, ztop + 100.0, 24)
    vals = 1.2 * np.exp(-levels / 8000.0)
    return np.tile(vals, (ng, 1)), np.tile(levels, (ng, 1))


def test_f2py_oracle():
    try:
        import clubb_f2py
        from clubb_python import clubb_api
        from clubb_python.derived_types.err_info import ErrInfo
    except Exception as e:
        print(f"  f2py remap_vals_to_target_same_grid oracle: SKIP ({type(e).__name__})")
        return
    jgr = setup_grid(ngrdcol=_NG, deltaz=_DZ, zm_init=0.0, zm_top=_ZTOP, grid_type=1)
    ng, nzm = jgr.zm.shape
    nzt = nzm - 1
    clubb_api.init_err_info(ng)
    cf = clubb_api.get_default_config_flags(); clubb_api.init_config_flags(cf)
    clubb_api.setup_grid(nzmax=nzm, ngrdcol=ng, sfc_elevation=np.zeros(ng),
                         l_implemented=False, l_ascending_grid=True, grid_type=2,
                         deltaz=np.full(ng, _DZ), zm_init=np.zeros(ng), zm_top=np.full(ng, float(jgr.zm[0, -1])),
                         momentum_heights=np.asfortranarray(np.asarray(jgr.zm)),
                         thermodynamic_heights=np.asfortranarray(np.asarray(jgr.zt)),
                         err_info=ErrInfo(ngrdcol=ng))
    rho_vals, rho_levels = _rho_spline(ng, float(jgr.zm[0, -1]))
    p_sfc = np.full(ng, 1.0e5)
    rng = np.random.default_rng(7)

    worst = 0.0
    for l_zt in (True, False):
        ncell = nzt if l_zt else nzm
        src = rng.uniform(-2, 2, (ng, ncell))
        f = np.asarray(clubb_f2py.f2py_remap_vals_to_target_same_grid(
            nzm, nzt, src, ncell, rho_vals, rho_levels, 1, p_sfc, 1, l_zt))
        g = np.asarray(remap_vals_to_target_same_grid(
            src, np.asarray(jgr.zm), np.asarray(jgr.zt), rho_vals, rho_levels, p_sfc,
            grid_remap_method=1, l_zt_variable=l_zt))
        worst = max(worst, np.max(np.abs(g - f)))
    assert worst < 1e-12, f"remap_vals_to_target_same_grid f2py mismatch {worst:.2e}"
    print(f"  f2py remap_vals_to_target_same_grid: bit-match (zt+zm variables), worst {worst:.2e}  PASS")


def test_remapping_matrix_identity_and_conservation():
    rng = np.random.default_rng(2)
    # Strictly decreasing pressure levels (as built from hydrostatic integration).
    p = np.cumsum(np.concatenate([[1e5], -rng.uniform(500, 1500, 9)]))[None, :]
    R = np.asarray(remapping_matrix(p, p))
    assert np.allclose(R[0], np.eye(9)), "same-level remapping matrix is not identity"
    # Refined target (split each source cell in two): each target row sums to 1 (consistency).
    pt = np.sort(np.concatenate([p[0], 0.5 * (p[0, :-1] + p[0, 1:])]))[::-1][None, :]
    R2 = np.asarray(remapping_matrix(p, pt))
    assert np.allclose(R2.sum(axis=2), 1.0), "remapping matrix rows do not sum to 1"
    print("  remapping_matrix: identity (same grid) + row-sum consistency  PASS")


def test_calc_mass_analytic():
    # Linear density rho(z) = a + b z over [0, 1000]; analytic mass over [z0,z1] = a(z1-z0) + b/2(z1^2-z0^2).
    a, b = 1.2, -3.0e-4
    levels = np.linspace(0, 1000, 11)
    vals = a + b * levels
    grid = np.array([[0.0, 250.0, 600.0, 1000.0]])
    mass = np.asarray(calc_mass_over_grid_intervals(vals[None, :], levels[None, :], grid))[0]
    edges = grid[0]
    exact = a * (edges[1:] - edges[:-1]) + 0.5 * b * (edges[1:] ** 2 - edges[:-1] ** 2)
    assert np.allclose(mass, exact, atol=1e-9), f"mass integral mismatch: {mass} vs {exact}"
    print(f"  calc_mass_over_grid_intervals: matches analytic linear-density integral (max |Δ| "
          f"{np.max(np.abs(mass - exact)):.2e})  PASS")


def test_matvec_and_grad():
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, (2, 4)); A = rng.uniform(-1, 1, (2, 3, 4))
    y = np.asarray(matrix_vector_mult(x, A))
    assert np.allclose(y, np.einsum('ikj,ij->ik', A, x)), "matrix_vector_mult != einsum"
    a, b = 1.2, -3.0e-4
    levels = jnp.asarray(np.linspace(0, 1000, 11))
    grid = jnp.asarray([[0.0, 400.0, 1000.0]])
    g = np.asarray(jax.grad(lambda v: jnp.sum(
        calc_mass_over_grid_intervals(v[None, :], levels[None, :], grid) ** 2))(a + b * levels))
    assert np.isfinite(g).all(), "non-finite grad through calc_mass_over_grid_intervals"
    print(f"  matrix_vector_mult==einsum + finite jax.grad through calc_mass ({g.size} entries)  PASS")


def main():
    print("test_remapping_module:")
    for t in (test_f2py_oracle, test_remapping_matrix_identity_and_conservation,
              test_calc_mass_analytic, test_matvec_and_grad):
        t()
    print("All remapping_module checks PASSED")


if __name__ == "__main__":
    main()
