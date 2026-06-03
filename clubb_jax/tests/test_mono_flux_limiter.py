#!/usr/bin/env python3
"""test_mono_flux_limiter.py — REFACTOR B2 (iter11): the JAX (lax.scan) monotonic flux limiter is
BIT-EXACT to the NumPy reference, and is differentiable.

The limiter fires only for strong-shear/stable BLs (atex, gabls3_night). To validate the conversion
without a slow full-case run, this compares `monotonic_turbulent_flux_limit_jax` against the original
NumPy `monotonic_turbulent_flux_limit` on synthetic inputs constructed to TRIGGER the clip + re-solve
(tight allowable bounds + large wpxp), across the solve types (rtm = spikefix, thlm, um = is_uv branch).
Then a `jax.grad` smoke check (the limiter is now differentiable w.r.t. the fields).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "../..")))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from clubb_jax.src.derived_types.grid_class import setup_grid
from clubb_jax.src.CLUBB_core.mono_flux_limiter import (
    monotonic_turbulent_flux_limit, monotonic_turbulent_flux_limit_jax,
    calc_turb_adv_range, MFL_RTM, MFL_THLM, MFL_UM,
)


def _make_inputs(rng, gr, ngrdcol, nzt, solve_type):
    nzm = nzt + 1
    # w-PDF fields (zm) → realistic integer turbulent-advection ranges via calc_turb_adv_range.
    w1 = -0.5 + rng.random((ngrdcol, nzm))
    w2 = -0.5 + rng.random((ngrdcol, nzm))
    vw1 = 0.05 + 0.5 * rng.random((ngrdcol, nzm))
    vw2 = 0.05 + 0.5 * rng.random((ngrdcol, nzm))
    mf = 0.3 + 0.4 * rng.random((ngrdcol, nzm))
    dt = 60.0
    lle, hle = calc_turb_adv_range(w1, w2, vw1, vw2, mf, gr, dt)

    if solve_type in (MFL_UM,):
        scale, xp2_thr = 5.0, 1e-3
    elif solve_type == MFL_RTM:
        scale, xp2_thr = 1e-2, 1e-9
    else:  # thlm
        scale, xp2_thr = 300.0, 1e-4
    xm_old = scale * (0.5 + rng.random((ngrdcol, nzt)))
    xm = xm_old + 0.01 * scale * (rng.random((ngrdcol, nzt)) - 0.5)
    xm_forcing = 1e-4 * scale * (rng.random((ngrdcol, nzt)) - 0.5)
    # large wpxp + tight xp2 → force the limiter to clip
    wpxp = 2.0 * scale * (rng.random((ngrdcol, nzm)) - 0.5)
    xp2 = (0.01 * scale) ** 2 * rng.random((ngrdcol, nzm))
    rho_ds_zm = 1.0 + 0.1 * rng.random((ngrdcol, nzm))
    rho_ds_zt = 1.0 + 0.1 * rng.random((ngrdcol, nzt))
    wm_zt = 0.01 * (rng.random((ngrdcol, nzt)) - 0.5)
    args = dict(solve_type=solve_type, xm=xm, wpxp=wpxp, xm_old=xm_old, xp2=xp2, wm_zt=wm_zt,
                xm_forcing=xm_forcing, rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
                invrs_rho_ds_zm=1.0 / rho_ds_zm, invrs_rho_ds_zt=1.0 / rho_ds_zt,
                xp2_threshold=xp2_thr, xm_tol=1e-4 * scale, low_lev_effect=lle,
                high_lev_effect=hle, gr=gr, dt=dt)
    return args


def test_bit_exact():
    gr = setup_grid(ngrdcol=2, deltaz=50.0, zm_init=0.0, zm_top=2000.0, grid_type=1)
    nzt = gr.zt.shape[1]
    worst = 0.0
    n_clipped = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        for st in (MFL_RTM, MFL_THLM, MFL_UM):
            a = _make_inputs(rng, gr, 2, nzt, st)
            xm_np, wpxp_np = monotonic_turbulent_flux_limit(**a)
            xm_jx, wpxp_jx = monotonic_turbulent_flux_limit_jax(**a)
            xm_jx = np.asarray(xm_jx); wpxp_jx = np.asarray(wpxp_jx)
            # confirm the limiter actually did something (else the test is vacuous)
            if np.max(np.abs(wpxp_np - a["wpxp"])) > 0:
                n_clipped += 1
            for ref, got in ((xm_np, xm_jx), (wpxp_np, wpxp_jx)):
                den = np.max(np.abs(ref)) + 1e-300
                worst = max(worst, np.max(np.abs(ref - got)) / den)
    assert n_clipped > 0, "test vacuous — the limiter never clipped; strengthen the inputs"
    assert worst < 1e-12, f"JAX flux limiter not bit-exact to NumPy: worst rel {worst:.2e}"
    print(f"  mono flux limiter: JAX (lax.scan) vs NumPy BIT-EXACT (worst rel {worst:.1e}; "
          f"{n_clipped}/18 cases clipped)  PASS")
    return True


def test_differentiable():
    gr = setup_grid(ngrdcol=1, deltaz=50.0, zm_init=0.0, zm_top=2000.0, grid_type=1)
    nzt = gr.zt.shape[1]
    rng = np.random.default_rng(0)
    a = _make_inputs(rng, gr, 1, nzt, MFL_RTM)

    def loss(wpxp):
        b = dict(a); b["wpxp"] = wpxp
        xm_new, wpxp_new = monotonic_turbulent_flux_limit_jax(**b)
        return jnp.sum(xm_new ** 2) + jnp.sum(wpxp_new ** 2)

    g = jax.grad(loss)(jnp.asarray(a["wpxp"]))
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.sum(jnp.abs(g))) > 0, "grad not finite/nonzero"
    print(f"  mono flux limiter: jax.grad w.r.t. wpxp finite+nonzero (|g|sum={float(jnp.sum(jnp.abs(g))):.2e})  PASS")
    return True


if __name__ == "__main__":
    ok = True
    ok &= test_bit_exact()
    ok &= test_differentiable()
    print("\nAll mono_flux_limiter tests PASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
