#!/usr/bin/env python3
"""test_clip_covar.py — validate the JAX clip_covar port (clip_explicit.F90:clip_covar).

Clips a covariance x'y' to ±max_mag_corr·sqrt(x'^2·y'^2) on interior momentum levels (boundaries untouched), so
|corr(x,y)| <= max_mag_corr. Ported (advance_xm_wpxp_module.py) but lacking a dedicated f2py test. Oracles:
  1. f2py bit-shadow vs f2py_clip_covar — both the clipped covariance and the net change, for a non-flux
     (clip_rtpthlp) and a flux (clip_wprtp) solve_type. SKIPs if clubb_f2py is unbuilt.
  2. Realizability: |corr| <= max_mag_corr after clipping on interior levels.
  3. A finite jax.grad.
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

from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import clip_covar
from clubb_jax.src.CLUBB_core.clip_explicit import clip_covars_denom
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats

NG, NZM = 2, 9
CLIP_RTPTHLP, CLIP_WPRTP = 3, 8
MAX_MAG = 0.99
DT = 60.0


def _inputs(seed):
    rng = np.random.default_rng(seed)
    xp2 = rng.uniform(0.1, 2.0, (NG, NZM))
    yp2 = rng.uniform(0.1, 2.0, (NG, NZM))
    # xpyp deliberately exceeds the bound at many levels to exercise both clip branches.
    xpyp = rng.uniform(-3.0, 3.0, (NG, NZM)) * np.sqrt(xp2 * yp2)
    return xp2, yp2, xpyp


def _empty_stats(nzm=NZM, ngrdcol=NG):
    return JaxStats.empty(
        l_sample=False,
        names=(),
        ncol=ngrdcol,
        max_nlev=nzm,
    )


def test_f2py_oracle():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py clip_covar oracle: SKIP ({type(e).__name__})")
        return
    worst = 0.0
    for solve_type in (CLIP_RTPTHLP, CLIP_WPRTP):
        xp2, yp2, xpyp = _inputs(11 + solve_type)
        f_xpyp, f_chnge = clubb_f2py.f2py_clip_covar(
            solve_type, np.asfortranarray(xp2), np.asfortranarray(yp2), np.asfortranarray(xpyp.copy()))
        g_xpyp, g_chnge = clip_covar(NZM, NG, solve_type, xp2, yp2, xpyp)
        g_xpyp = np.asarray(g_xpyp)
        g_chnge = np.array(g_chnge)  # writable copy; boundaries already 0 from JAX interior_mask
        worst = max(worst, np.max(np.abs(g_xpyp - np.asarray(f_xpyp))),
                    np.max(np.abs(g_chnge - np.asarray(f_chnge))))
    assert worst < 1e-12, f"clip_covar f2py mismatch {worst:.2e}"
    print(f"  f2py clip_covar: bit-match (xpyp + change, non-flux/flux), worst {worst:.2e}  PASS")


def test_clip_covars_denom_f2py():
    """clip_covars_denom (clip_explicit.F90) — the post-wp2/wp3-solve Cauchy-Schwarz clip driver that calls
    clip_covar on wprtp(/wp2,rtp2), wpthlp(/wp2,thlp2) and (l_tke_aniso) upwp(/wp2,up2)+vpwp(/wp2,vp2), else
    upwp/vpwp against wp2. Validate the 4 clipped covars vs f2py_clip_covars_denom for BOTH l_tke_aniso branches.
    The JAX hardcodes sclr_dim=0 + l_linearize_pbl_winds=False, so drive the f2py with sclr_dim=1 (dummy zero
    sclrp2/wpsclrp — its scalar clip cannot perturb the 4 momentum/thermo covars) and zero pert winds, comparing
    only the 4 returned fields. SKIPs if clubb_f2py is unbuilt. (iter 434)"""
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py clip_covars_denom oracle: SKIP ({type(e).__name__})")
        return
    rng = np.random.default_rng(3)
    worst = 0.0
    sclr_dim = 0
    sclrp2 = np.zeros((NG, NZM, max(sclr_dim, 1)))
    wpsclrp = np.zeros((NG, NZM, max(sclr_dim, 1)))
    upwp_pert = np.zeros((NG, NZM))
    vpwp_pert = np.zeros((NG, NZM))
    for aniso in (True, False):
        for _ in range(15):
            wp2 = rng.uniform(0.0, 1.0, (NG, NZM)); rtp2 = rng.uniform(0.0, 1e-6, (NG, NZM))
            thlp2 = rng.uniform(0.0, 1.0, (NG, NZM)); up2 = rng.uniform(0.0, 1.0, (NG, NZM))
            vp2 = rng.uniform(0.0, 1.0, (NG, NZM))
            # large covars to force clipping past the ±0.99 correlation bound
            wprtp = rng.uniform(-1, 1, (NG, NZM)) * 1e-3; wpthlp = rng.uniform(-2, 2, (NG, NZM))
            upwp = rng.uniform(-2, 2, (NG, NZM)); vpwp = rng.uniform(-2, 2, (NG, NZM))
            stats = _empty_stats()
            j_out = clip_covars_denom(
                NZM, NG, sclr_dim, DT,
                jnp.asarray(rtp2), jnp.asarray(thlp2), jnp.asarray(up2), jnp.asarray(vp2),
                jnp.asarray(wp2), jnp.asarray(sclrp2),
                bool(aniso), False, False,
                stats,
                0, 0, 0, 0,
                jnp.asarray(wprtp), jnp.asarray(wpthlp), jnp.asarray(upwp), jnp.asarray(vpwp),
                jnp.asarray(wpsclrp), jnp.asarray(upwp_pert), jnp.asarray(vpwp_pert),
            )
            # j_out = (wprtp, wpthlp, upwp, vpwp, wpsclrp, upwp_pert, vpwp_pert,
            #          wprtp_cl_num, wpthlp_cl_num, upwp_cl_num, vpwp_cl_num, stats)
            sclrp2_f = np.zeros((NG, NZM, 1)); wpsclrp_f = np.zeros((NG, NZM, 1))
            z = np.zeros((NG, NZM))
            # f2py signature: f2py_clip_covars_denom(sclr_dim, dt, rtp2, ...) — nzm/ngrdcol inferred from arrays.
            # Returns (wprtp_cl_num, wpthlp_cl_num, upwp_cl_num, vpwp_cl_num, wprtp, wpthlp, upwp, vpwp, ...)
            f = clubb_f2py.f2py_clip_covars_denom(
                1, DT,
                rtp2, thlp2, up2, vp2, wp2, sclrp2_f,
                int(aniso), 0, 0,
                0, 0, 0, 0,
                wprtp.copy(), wpthlp.copy(), upwp.copy(), vpwp.copy(),
                wpsclrp_f, z.copy(), z.copy())
            for k in range(4):   # j_out: indices 4..7 are (wprtp, wpthlp, upwp, vpwp); f2py: same at f[4..7]
                worst = max(worst, float(np.max(np.abs(np.asarray(j_out[4 + k]) - np.asarray(f[4 + k])))))
    assert worst < 1e-12, f"clip_covars_denom f2py mismatch {worst:.2e}"
    print(f"  f2py clip_covars_denom (4 clipped covars, both l_tke_aniso branches, 30 cases): "
          f"bit-match, worst {worst:.2e}  PASS")


def test_realizability():
    xp2, yp2, xpyp = _inputs(5)
    g, _ = clip_covar(NZM, NG, CLIP_RTPTHLP, xp2, yp2, xpyp)
    g = np.asarray(g)
    corr = g / np.sqrt(xp2 * yp2)
    # Interior levels must satisfy |corr| <= max_mag_corr (boundaries are left as-is).
    assert np.all(np.abs(corr[:, 1:-1]) <= MAX_MAG + 1e-12), "interior correlation exceeds max_mag_corr"
    # Boundaries untouched.
    assert np.allclose(g[:, 0], xpyp[:, 0]) and np.allclose(g[:, -1], xpyp[:, -1]), "boundaries changed"
    print("  realizability: |corr|<=0.99 on interior, boundaries untouched  PASS")


def test_differentiable():
    xp2, yp2, xpyp = _inputs(7)
    def loss(v):
        out, _ = clip_covar(NZM, NG, CLIP_RTPTHLP, xp2, yp2, v)
        return jnp.sum(out ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(xpyp)))
    assert np.isfinite(g).all(), "non-finite grad through clip_covar"
    print(f"  jax.grad through clip_covar: finite ({g.size} entries)  PASS")


def main():
    print("test_clip_covar:")
    for t in (test_f2py_oracle, test_clip_covars_denom_f2py, test_realizability, test_differentiable):
        t()
    print("All clip_covar checks PASSED")


if __name__ == "__main__":
    main()
