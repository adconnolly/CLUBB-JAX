"""Bit-level equivalence tests: JAX implementations vs. compiled Fortran oracle.

The Fortran oracle is a real gfortran-compiled executable built from the
actual CLUBB source files.  It accepts parameters on stdin and writes
results to stdout.  These tests compare JAX outputs against the Fortran
outputs for the same inputs.

Tested routines:
  - zm2zt  (grid_class.F90  linear_interpolated_azt_2D via zm2zt_api)
  - zt2zm  (grid_class.F90  linear_interpolated_azm_2D via zt2zm_api)
  - diffusion_zt_lhs  (diffusion.F90)
  - diffusion_zm_lhs  (diffusion.F90)

Tolerance: 1e-13 (well within double-precision round-off; both sides use
the same IEEE 754 double arithmetic).

Run:
    python clubb_jax/tests/test_vs_fortran.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import struct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zt2zm_jax
from clubb_jax.src.CLUBB_core.diffusion import diffusion_zt_lhs_jax, diffusion_zm_lhs_jax
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_lu_solve_jax

ORACLE_EXE = os.path.join(
    os.path.dirname(__file__), '..', 'fortran_oracle', 'build_exe', 'oracle_driver.exe'
)
HAS_ORACLE = os.path.isfile(ORACLE_EXE)


# ──────────────────────────────────────────────────────────────────────────────
# Oracle driver
# ──────────────────────────────────────────────────────────────────────────────

def call_oracle(nzm, ngrdcol, deltaz, zm_init, K_zm, K_zt, nu, rho_zm, rho_zt):
    """Run oracle_driver.exe and parse output into numpy arrays.

    Returns dict with keys: ZM2ZT, ZT2ZM, DIFF_ZT_LHS, DIFF_ZM_LHS.
    """
    nzt = nzm - 1
    stdin = f"{nzm}\n{ngrdcol}\n{deltaz:.17g}\n{zm_init:.17g}\n{K_zm:.17g}\n{K_zt:.17g}\n{nu:.17g}\n{rho_zm:.17g}\n{rho_zt:.17g}\n"
    result = subprocess.run(
        [ORACLE_EXE], input=stdin, capture_output=True, text=True, check=True
    )
    out = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        tag = parts[0]
        vals = np.array([float(v) for v in parts[1:]], dtype=np.float64)
        if tag == 'ZM2ZT':
            # shape (nzt, ngrdcol) stored k-major, i-minor
            out['ZM2ZT'] = vals.reshape(nzt, ngrdcol)
        elif tag == 'ZT2ZM':
            out['ZT2ZM'] = vals.reshape(nzm, ngrdcol)
        elif tag == 'DIFF_ZT_LHS':
            # 3 values per (ngrdcol * nzt) entry, k-major, i-minor
            out['DIFF_ZT_LHS'] = vals.reshape(nzt, ngrdcol, 3)
        elif tag == 'DIFF_ZM_LHS':
            out['DIFF_ZM_LHS'] = vals.reshape(nzm, ngrdcol, 3)
        elif tag == 'SOLVER_ZT_SOLN':
            out['SOLVER_ZT_SOLN'] = vals.reshape(nzt, ngrdcol)
        elif tag == 'SOLVER_ZM_SOLN':
            out['SOLVER_ZM_SOLN'] = vals.reshape(nzm, ngrdcol)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Grid helper
# ──────────────────────────────────────────────────────────────────────────────

def make_even_grid(nzm, deltaz, zm_init, ngrdcol):
    zm_1d = np.array([zm_init + k * deltaz for k in range(nzm)], dtype=np.float64)
    zt_1d = 0.5 * (zm_1d[:-1] + zm_1d[1:])
    nzt = nzm - 1

    zm = np.tile(zm_1d, (ngrdcol, 1))
    zt = np.tile(zt_1d, (ngrdcol, 1))
    invrs_dzt = 1.0 / (zm[:, 1:] - zm[:, :-1])

    dzm_int = zt[:, 1:] - zt[:, :-1]
    invrs_dzm_int = 1.0 / dzm_int
    invrs_dzm = np.concatenate([
        invrs_dzm_int[:, :1], invrs_dzm_int, invrs_dzm_int[:, -1:]
    ], axis=1)

    class Grid:
        pass
    gr = Grid()
    gr.nzm, gr.nzt, gr.ngrdcol = nzm, nzt, ngrdcol
    gr.zm, gr.zt = zm, zt
    gr.invrs_dzt = invrs_dzt
    gr.invrs_dzm = invrs_dzm
    return gr


def _to_jax(gr):
    class JG:
        pass
    jgr = JG()
    for a in ('nzm', 'nzt', 'ngrdcol'):
        setattr(jgr, a, getattr(gr, a))
    for a in ('zm', 'zt', 'invrs_dzt', 'invrs_dzm'):
        setattr(jgr, a, jnp.array(getattr(gr, a), dtype=jnp.float64))
    return jgr


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_zm2zt_vs_fortran():
    """JAX zm2zt matches Fortran linear_interpolated_azt_2D (linear profile)."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    nzm, ngrdcol, deltaz, zm_init = 7, 2, 150.0, 75.0
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init, 5.0, 1.0, 0.0, 1.0, 1.0)

    gr = make_even_grid(nzm, deltaz, zm_init, ngrdcol)
    # Linear profile f = 2 + 0.5*z  (matches oracle_driver)
    azm = (2.0 + 0.5 * gr.zm).astype(np.float64)
    jgr = _to_jax(gr)
    azt_jax = np.asarray(zm2zt_jax(jnp.array(azm), jgr))

    # Oracle: ZM2ZT shape (nzt, ngrdcol); JAX: (ngrdcol, nzt) — transpose oracle
    azt_fortran = oracle['ZM2ZT'].T           # → (ngrdcol, nzt)
    err = np.max(np.abs(azt_jax - azt_fortran))
    print(f"  zm2zt max_err = {err:.3e}", "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"zm2zt mismatch: {err}"


def test_zt2zm_vs_fortran():
    """JAX zt2zm matches Fortran linear_interpolated_azm_2D (linear profile)."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    nzm, ngrdcol, deltaz, zm_init = 7, 2, 150.0, 75.0
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init, 5.0, 1.0, 0.0, 1.0, 1.0)

    gr = make_even_grid(nzm, deltaz, zm_init, ngrdcol)
    azt = (2.0 + 0.5 * gr.zt).astype(np.float64)
    jgr = _to_jax(gr)
    azm_jax = np.asarray(zt2zm_jax(jnp.array(azt), jgr))

    azm_fortran = oracle['ZT2ZM'].T           # → (ngrdcol, nzm)
    err = np.max(np.abs(azm_jax - azm_fortran))
    print(f"  zt2zm max_err = {err:.3e}", "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"zt2zm mismatch: {err}"


def _run_diffusion_zt_comparison(nzm, ngrdcol, deltaz, zm_init, K_zm_val, nu_val, rho_val):
    """Helper: run oracle + JAX for diffusion_zt_lhs and return (err, fortran, jax)."""
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init,
                         K_zm_val, 1.0, nu_val, rho_val, 1.0)
    nzt = nzm - 1
    gr = make_even_grid(nzm, deltaz, zm_init, ngrdcol)
    jgr = _to_jax(gr)
    K_zm = K_zm_val * jnp.ones((ngrdcol, nzm))
    nu = nu_val * jnp.ones(ngrdcol)
    invrs_rho_ds_zt = (1.0 / rho_val) * jnp.ones((ngrdcol, nzt))
    rho_ds_zm = rho_val * jnp.ones((ngrdcol, nzm))
    lhs_jax = np.asarray(diffusion_zt_lhs_jax(K_zm, nu, invrs_rho_ds_zt, rho_ds_zm, jgr))

    # Oracle DIFF_ZT_LHS: shape (nzt, ngrdcol, 3) — k, col, [super,main,sub]
    # JAX lhs: shape (3, ngrdcol, nzt) — [super,main,sub], col, k
    lhs_fort = oracle['DIFF_ZT_LHS']         # (nzt, ngrdcol, 3)
    lhs_fort_t = lhs_fort.transpose(2, 1, 0) # → (3, ngrdcol, nzt)
    err = np.max(np.abs(lhs_jax - lhs_fort_t))
    return err, lhs_fort_t, lhs_jax


def test_diffusion_zt_lhs_vs_fortran_uniform():
    """diffusion_zt_lhs: JAX == Fortran for uniform grid, constant K/rho."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    err, _, _ = _run_diffusion_zt_comparison(5, 1, 100.0, 50.0, 10.0, 0.0, 1.0)
    print(f"  diffusion_zt_lhs (nzm=5, K=10) max_err = {err:.3e}",
          "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"diffusion_zt_lhs mismatch: {err}"


def test_diffusion_zt_lhs_vs_fortran_varied():
    """diffusion_zt_lhs: JAX == Fortran for varied K, nu, rho, grid sizes."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    cases = [
        (8, 3, 200.0, 100.0, 5.0,  0.5, 1.2),
        (10, 1, 50.0,  25.0, 0.1,  0.0, 0.8),
        (6,  2, 500.0, 250.0, 20.0, 2.0, 1.5),
    ]
    for nzm, ngrdcol, dz, z0, K, nu, rho in cases:
        err, _, _ = _run_diffusion_zt_comparison(nzm, ngrdcol, dz, z0, K, nu, rho)
        print(f"  nzm={nzm} ngrdcol={ngrdcol} K={K} nu={nu} rho={rho}: err={err:.3e}",
              "PASS" if err < 1e-13 else "FAIL")
        assert err < 1e-13, f"diffusion_zt_lhs mismatch (nzm={nzm}): {err}"


def _run_diffusion_zm_comparison(nzm, ngrdcol, deltaz, zm_init, K_zt_val, nu_val, rho_val):
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init,
                         1.0, K_zt_val, nu_val, 1.0, rho_val)
    nzt = nzm - 1
    gr = make_even_grid(nzm, deltaz, zm_init, ngrdcol)
    jgr = _to_jax(gr)
    K_zt = K_zt_val * jnp.ones((ngrdcol, nzt))
    nu = nu_val * jnp.ones(ngrdcol)
    invrs_rho_ds_zm = (1.0 / rho_val) * jnp.ones((ngrdcol, nzm))
    rho_ds_zt = rho_val * jnp.ones((ngrdcol, nzt))
    lhs_jax = np.asarray(diffusion_zm_lhs_jax(K_zt, nu, invrs_rho_ds_zm, rho_ds_zt, jgr))

    lhs_fort = oracle['DIFF_ZM_LHS']          # (nzm, ngrdcol, 3)
    lhs_fort_t = lhs_fort.transpose(2, 1, 0)  # → (3, ngrdcol, nzm)
    err = np.max(np.abs(lhs_jax - lhs_fort_t))
    return err, lhs_fort_t, lhs_jax


def test_diffusion_zm_lhs_vs_fortran_uniform():
    """diffusion_zm_lhs: JAX == Fortran for uniform grid, constant K/rho."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    err, _, _ = _run_diffusion_zm_comparison(5, 1, 100.0, 50.0, 1.0, 0.0, 1.0)
    print(f"  diffusion_zm_lhs (nzm=5, K=1) max_err = {err:.3e}",
          "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"diffusion_zm_lhs mismatch: {err}"


def test_diffusion_zm_lhs_vs_fortran_varied():
    """diffusion_zm_lhs: JAX == Fortran for varied K, nu, rho, grid sizes."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    cases = [
        (8, 3, 200.0, 100.0, 4.0,  0.3, 1.1),
        (10, 1, 50.0,  25.0, 0.2,  0.0, 0.9),
        (6,  2, 300.0, 150.0, 15.0, 1.5, 1.3),
    ]
    for nzm, ngrdcol, dz, z0, K, nu, rho in cases:
        err, _, _ = _run_diffusion_zm_comparison(nzm, ngrdcol, dz, z0, K, nu, rho)
        print(f"  nzm={nzm} ngrdcol={ngrdcol} K={K} nu={nu} rho={rho}: err={err:.3e}",
              "PASS" if err < 1e-13 else "FAIL")
        assert err < 1e-13, f"diffusion_zm_lhs mismatch (nzm={nzm}): {err}"


def _make_artificial_lhs(ndim, ngrdcol):
    """Build the same well-conditioned test system used by the oracle.

    main=4, super=-1 (0 at top), sub=-1 (0 at bottom), rhs=k (1-based).
    Returns (lhs_jax, rhs_jax) with lhs shape (3, ngrdcol, ndim).
    """
    sup_np = -np.ones((ngrdcol, ndim))
    mid_np =  4.0 * np.ones((ngrdcol, ndim))
    sub_np = -np.ones((ngrdcol, ndim))
    sup_np[:, -1] = 0.0   # top boundary
    sub_np[:,  0] = 0.0   # bottom boundary
    rhs_np = np.tile(np.arange(1, ndim + 1, dtype=np.float64), (ngrdcol, 1))

    lhs_jax = jnp.stack([
        jnp.array(sup_np, dtype=jnp.float64),
        jnp.array(mid_np, dtype=jnp.float64),
        jnp.array(sub_np, dtype=jnp.float64),
    ], axis=0)   # (3, ngrdcol, ndim)
    rhs_jax = jnp.array(rhs_np, dtype=jnp.float64)
    return lhs_jax, rhs_jax


def test_solver_zt_vs_fortran():
    """JAX solver matches Fortran tridiag_lu_solve on zt-grid artificial system."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    nzm, ngrdcol, deltaz, zm_init = 7, 2, 150.0, 75.0
    nzt = nzm - 1
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init, 5.0, 1.0, 0.0, 1.0, 1.0)

    lhs_jax, rhs_jax = _make_artificial_lhs(nzt, ngrdcol)
    soln_jax = np.asarray(tridiag_lu_solve_jax(lhs_jax, rhs_jax))  # (ngrdcol, nzt)

    # Oracle SOLVER_ZT_SOLN: shape (nzt, ngrdcol) → transpose to (ngrdcol, nzt)
    soln_fort = oracle['SOLVER_ZT_SOLN'].T
    err = np.max(np.abs(soln_jax - soln_fort))
    print(f"  solver_zt max_err = {err:.3e}", "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"solver_zt mismatch: {err}"


def test_solver_zm_vs_fortran():
    """JAX solver matches Fortran tridiag_lu_solve on zm-grid artificial system."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    nzm, ngrdcol, deltaz, zm_init = 7, 2, 150.0, 75.0
    oracle = call_oracle(nzm, ngrdcol, deltaz, zm_init, 5.0, 1.0, 0.0, 1.0, 1.0)

    lhs_jax, rhs_jax = _make_artificial_lhs(nzm, ngrdcol)
    soln_jax = np.asarray(tridiag_lu_solve_jax(lhs_jax, rhs_jax))  # (ngrdcol, nzm)

    soln_fort = oracle['SOLVER_ZM_SOLN'].T   # (ngrdcol, nzm)
    err = np.max(np.abs(soln_jax - soln_fort))
    print(f"  solver_zm max_err = {err:.3e}", "  PASS" if err < 1e-13 else "  FAIL")
    assert err < 1e-13, f"solver_zm mismatch: {err}"


def test_solver_vs_fortran_varied():
    """JAX solver matches Fortran for varied grid sizes and column counts."""
    if not HAS_ORACLE or not HAS_JAX:
        print("  SKIP"); return
    cases = [
        (5, 1, 100.0, 50.0),
        (10, 3, 200.0, 100.0),
        (15, 4, 50.0, 25.0),
    ]
    for nzm, ngrdcol, dz, z0 in cases:
        nzt = nzm - 1
        oracle = call_oracle(nzm, ngrdcol, dz, z0, 5.0, 1.0, 0.0, 1.0, 1.0)

        lhs_jax, rhs_jax = _make_artificial_lhs(nzt, ngrdcol)
        soln_jax = np.asarray(tridiag_lu_solve_jax(lhs_jax, rhs_jax))
        soln_fort = oracle['SOLVER_ZT_SOLN'].T
        err = np.max(np.abs(soln_jax - soln_fort))
        print(f"  nzm={nzm} ngrdcol={ngrdcol}: err={err:.3e}",
              "PASS" if err < 1e-13 else "FAIL")
        assert err < 1e-13, f"solver mismatch (nzm={nzm}): {err}"


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("CLUBB JAX vs. Fortran oracle equivalence tests")
    print("=" * 60)

    if not HAS_JAX:
        print("ERROR: JAX not available.")
        sys.exit(1)
    if not HAS_ORACLE:
        # The standalone `fortran_oracle` exe is SUPERSEDED by compare_runs.py (compiled clubb_standalone),
        # so it is normally absent. SKIP cleanly (exit 0) so the suite stays green/portable — these grid-op
        # vs-Fortran checks run only if someone rebuilds that legacy oracle. (Diffusion/solver are also
        # covered by test_diffusion / test_solver / test_penta_solver.)
        print(f"  SKIP: legacy fortran_oracle exe not present ({ORACLE_EXE}); "
              f"use compare_runs.py for Fortran comparison")
        sys.exit(0)

    tests = [
        test_zm2zt_vs_fortran,
        test_zt2zm_vs_fortran,
        test_diffusion_zt_lhs_vs_fortran_uniform,
        test_diffusion_zt_lhs_vs_fortran_varied,
        test_diffusion_zm_lhs_vs_fortran_uniform,
        test_diffusion_zm_lhs_vs_fortran_varied,
        test_solver_zt_vs_fortran,
        test_solver_zm_vs_fortran,
        test_solver_vs_fortran_varied,
    ]

    passed = failed = 0
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
