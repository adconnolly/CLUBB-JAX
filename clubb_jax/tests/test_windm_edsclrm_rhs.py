#!/usr/bin/env python3
"""test_windm_edsclrm_rhs.py — pin the windm/edsclrm Crank-Nicholson RHS.

`windm_edsclrm_rhs` (advance_windm_edsclrm_module.py ↔ ...F90:windm_edsclrm_rhs, F90:2352) builds the RHS of the
implicit wind/edsclr eddy-diffusion solve: the Crank-Nicholson EXPLICIT half. It is live-path (every standalone case)
but validated only end-to-end. The CN scheme is

    (I/dt + 0.5·L_diff + …) xⁿ⁺¹ = (I/dt − 0.5·L_diff) xⁿ + tndcy

so the RHS must apply the SAME banded diffusion operator `lhs_diff` that `windm_edsclrm_lhs` uses at +0.5 (tested
iter 544) — here at −0.5 — to xⁿ, plus xⁿ/dt and the explicit tendency, with the tridiagonal operator boundary-
truncated (no sub-diagonal at the lower boundary, no super-diagonal at the upper). This pins exactly that: an
INDEPENDENT banded matrix-vector product of `lhs_diff` applied to `xm` (band layout [super, diag, sub] = [0,1,2]),
then `rhs == −0.5·(lhs_diff @ xm) + xm_tndcy + xm/dt`. Plus the no-diffusion property (lhs_diff=0 → rhs = xm/dt +
tndcy) and a finite jax.grad. Oracle-independent; never SKIPs. (iter 545)
"""
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

from clubb_jax.src.CLUBB_core.advance_windm_edsclrm_module import (
    windm_edsclrm_rhs, windm_edsclrm_um,
)
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats
from clubb_jax.src.derived_types.grid_class import setup_grid

_NG, _NZT = 2, 8
_DZ = 40.0


def _make_gr():
    # nzm = _NZT + 1 (momentum levels); nzt = _NZT (thermodynamic levels)
    return setup_grid(_NG, _DZ, _DZ, _DZ * (_NZT + 1))


def _empty_stats(gr):
    return JaxStats.empty(
        l_sample=False, names=(), ncol=gr.ngrdcol, max_nlev=max(gr.nzm, gr.nzt),
    )


def _rhs_call(gr, lhs_diff, xm, xm_tndcy, dt):
    """Wrapper that supplies all new leading/trailing args, with l_imp_sfc_momentum_flux=True
    to skip the surface-flux branch (so only the CN tridiagonal part is exercised)."""
    nzm = gr.nzm; nzt = gr.nzt; ng = gr.ngrdcol
    rho_ds_zm = jnp.ones((ng, nzm))
    invrs_rho_ds_zt = jnp.ones((ng, nzt))
    xpwp_sfc = jnp.zeros(ng)
    rhs, _stats = windm_edsclrm_rhs(
        nzm, nzt, ng, gr, windm_edsclrm_um, dt,
        lhs_diff, xm, xm_tndcy,
        rho_ds_zm, invrs_rho_ds_zt,
        True, xpwp_sfc,
        _empty_stats(gr),
    )
    return rhs


def _banded_matvec(lhs_diff, xm):
    """Independent tridiagonal matvec: out[k] = sub[k]·x[k−1] + diag[k]·x[k] + super[k]·x[k+1], boundary-truncated.
    Band layout [super, diag, sub] = [0, 1, 2] (the JAX/Fortran windm convention)."""
    sup, diag, sub = lhs_diff[0], lhs_diff[1], lhs_diff[2]
    out = np.empty_like(xm)
    nzt = xm.shape[1]
    for k in range(nzt):
        v = diag[:, k] * xm[:, k]
        if k - 1 >= 0:
            v = v + sub[:, k] * xm[:, k - 1]
        if k + 1 < nzt:
            v = v + sup[:, k] * xm[:, k + 1]
        out[:, k] = v
    return out


def _inputs(rng):
    lhs_diff = rng.uniform(-2.0, 2.0, (3, _NG, _NZT))
    xm = rng.uniform(-12.0, 12.0, (_NG, _NZT))
    xm_tndcy = rng.uniform(-1e-3, 1e-3, (_NG, _NZT))
    return lhs_diff, xm, xm_tndcy


def test_rhs_is_cn_explicit_half():
    gr = _make_gr()
    rng = np.random.default_rng(545)
    dt = 60.0
    lhs_diff, xm, xm_tndcy = _inputs(rng)
    got = np.asarray(_rhs_call(gr, jnp.asarray(lhs_diff), jnp.asarray(xm), jnp.asarray(xm_tndcy), dt))
    ref = -0.5 * _banded_matvec(lhs_diff, xm) + xm_tndcy + xm / dt
    worst = float(np.max(np.abs(got - ref)))
    assert worst < 1e-12, f"windm_edsclrm_rhs mismatch vs CN explicit half {worst:.2e}"
    print(f"  windm_edsclrm_rhs == −0.5·(lhs_diff@xm) + tndcy + xm/dt, boundary-truncated (worst {worst:.1e})  PASS")


def test_no_diffusion_is_pure_time_plus_tendency():
    gr = _make_gr()
    rng = np.random.default_rng(3)
    dt = 30.0
    _, xm, xm_tndcy = _inputs(rng)
    zero = np.zeros((3, _NG, _NZT))
    got = np.asarray(_rhs_call(gr, jnp.asarray(zero), jnp.asarray(xm), jnp.asarray(xm_tndcy), dt))
    ref = xm_tndcy + xm / dt
    assert np.max(np.abs(got - ref)) < 1e-13, "lhs_diff=0 must give rhs = xm/dt + tndcy"
    print("  no-diffusion: rhs = xm/dt + tndcy (pure time + tendency)  PASS")


def test_grad_finite():
    gr = _make_gr()
    rng = np.random.default_rng(99)
    lhs_diff, xm, xm_tndcy = _inputs(rng)
    g = jax.grad(lambda x: jnp.sum(
        _rhs_call(gr, jnp.asarray(lhs_diff), x, jnp.asarray(xm_tndcy), 60.0) ** 2))(jnp.asarray(xm))
    assert np.all(np.isfinite(np.asarray(g))), "non-finite grad wrt xm"
    print("  jax.grad(windm_edsclrm_rhs) wrt xm finite  PASS")


def main():
    print("test_windm_edsclrm_rhs:")
    test_rhs_is_cn_explicit_half()
    test_no_diffusion_is_pure_time_plus_tendency()
    test_grad_finite()
    print("All windm_edsclrm_rhs checks PASSED")


if __name__ == "__main__":
    main()
