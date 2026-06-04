#!/usr/bin/env python3
"""test_clip_covar.py — validate the JAX clip_covar_jax port (clip_explicit.F90:clip_covar).

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

from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import clip_covar_jax

NG, NZM = 2, 9
CLIP_RTPTHLP, CLIP_WPRTP = 3, 8
MAX_MAG = 0.99


def _inputs(seed):
    rng = np.random.default_rng(seed)
    xp2 = rng.uniform(0.1, 2.0, (NG, NZM))
    yp2 = rng.uniform(0.1, 2.0, (NG, NZM))
    # xpyp deliberately exceeds the bound at many levels to exercise both clip branches.
    xpyp = rng.uniform(-3.0, 3.0, (NG, NZM)) * np.sqrt(xp2 * yp2)
    return xp2, yp2, xpyp


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
        g_xpyp = np.asarray(clip_covar_jax(xpyp, xp2, yp2, MAX_MAG))
        # JAX returns the clipped covariance; reconstruct the change with zero boundaries (Fortran convention).
        g_chnge = g_xpyp - xpyp
        g_chnge[:, 0] = 0.0; g_chnge[:, -1] = 0.0
        worst = max(worst, np.max(np.abs(g_xpyp - np.asarray(f_xpyp))),
                    np.max(np.abs(g_chnge - np.asarray(f_chnge))))
    assert worst < 1e-12, f"clip_covar f2py mismatch {worst:.2e}"
    print(f"  f2py clip_covar: bit-match (xpyp + change, non-flux/flux), worst {worst:.2e}  PASS")


def test_realizability():
    xp2, yp2, xpyp = _inputs(5)
    g = np.asarray(clip_covar_jax(xpyp, xp2, yp2, MAX_MAG))
    corr = g / np.sqrt(xp2 * yp2)
    # Interior levels must satisfy |corr| <= max_mag_corr (boundaries are left as-is).
    assert np.all(np.abs(corr[:, 1:-1]) <= MAX_MAG + 1e-12), "interior correlation exceeds max_mag_corr"
    # Boundaries untouched.
    assert np.allclose(g[:, 0], xpyp[:, 0]) and np.allclose(g[:, -1], xpyp[:, -1]), "boundaries changed"
    print("  realizability: |corr|<=0.99 on interior, boundaries untouched  PASS")


def test_differentiable():
    xp2, yp2, xpyp = _inputs(7)
    def loss(v):
        return jnp.sum(clip_covar_jax(v, xp2, yp2, MAX_MAG) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(xpyp)))
    assert np.isfinite(g).all(), "non-finite grad through clip_covar_jax"
    print(f"  jax.grad through clip_covar_jax: finite ({g.size} entries)  PASS")


def main():
    print("test_clip_covar:")
    for t in (test_f2py_oracle, test_realizability, test_differentiable):
        t()
    print("All clip_covar checks PASSED")


if __name__ == "__main__":
    main()
