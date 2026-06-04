#!/usr/bin/env python3
"""test_close_luhar_pdf.py — validate the JAX close_Luhar_pdf port (adg1_adg2_3d_luhar_pdf.F90).

PDF component widths/means/variances for the Luhar closure. Oracles:
  1. f2py bit-shadow vs f2py_close_luhar_pdf (8 outputs), in the well-defined xp2 > x_tol_sqd regime (the
     Fortran's degenerate branch reads an uninitialized sgn, so it is not compared). SKIPs if unbuilt.
  2. Moment reconstruction: the binormal reproduces the overall mean xm and variance xp2.
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

from clubb_jax.src.CLUBB_core.adg1_adg2_3d_luhar_pdf import close_Luhar_pdf

NG, NZ = 2, 6
_X_TOL_SQD = 1.0e-8


def _inputs(seed):
    rng = np.random.default_rng(seed)
    xm = rng.uniform(-2, 2, (NG, NZ))
    xp2 = rng.uniform(0.05, 2.0, (NG, NZ))      # all > x_tol_sqd
    mixt_frac = rng.uniform(0.2, 0.8, (NG, NZ))
    small_m = rng.uniform(0.05, 1.0, (NG, NZ))
    wpxp = rng.uniform(-1, 1, (NG, NZ))
    return xm, xp2, mixt_frac, small_m, wpxp


def test_f2py_oracle():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py close_luhar_pdf oracle: SKIP ({type(e).__name__})")
        return
    worst = 0.0
    for seed in (11, 22, 33):
        xm, xp2, mf, m, wpxp = _inputs(seed)
        f = clubb_f2py.f2py_close_luhar_pdf(xm, xp2, mf, m, wpxp, _X_TOL_SQD)
        g = close_Luhar_pdf(xm, xp2, mf, m, wpxp, _X_TOL_SQD)
        for fi, gi in zip(f, g):
            worst = max(worst, np.max(np.abs(np.asarray(gi) - np.asarray(fi))))
    assert worst < 1e-11, f"close_luhar_pdf f2py mismatch {worst:.2e}"
    print(f"  f2py close_luhar_pdf: bit-match (8 outputs, xp2>tol), worst {worst:.2e}  PASS")


def test_moment_reconstruction():
    xm, xp2, mf, m, wpxp = _inputs(5)
    ss1, ss2, v1, v2, x1n, x2n, x1, x2 = (np.asarray(x) for x in close_Luhar_pdf(xm, xp2, mf, m, wpxp, _X_TOL_SQD))
    xm_rec = mf * x1 + (1 - mf) * x2
    assert np.max(np.abs(xm_rec - xm)) < 1e-12, "overall mean not reproduced"
    xp2_rec = mf * ((x1 - xm) ** 2 + v1) + (1 - mf) * ((x2 - xm) ** 2 + v2)
    assert np.max(np.abs(xp2_rec - xp2)) < 1e-10, "overall variance not reproduced"
    print("  moment reconstruction: binormal reproduces overall mean & variance  PASS")


def test_differentiable():
    xm, xp2, mf, m, wpxp = _inputs(7)
    def loss(v):
        outs = close_Luhar_pdf(xm, v, mf, m, wpxp, _X_TOL_SQD)
        return sum(jnp.sum(o ** 2) for o in outs)
    g = np.asarray(jax.grad(loss)(jnp.asarray(xp2)))
    assert np.isfinite(g).all(), "non-finite grad through close_Luhar_pdf"
    print(f"  jax.grad through close_Luhar_pdf: finite ({g.size} entries)  PASS")


def main():
    print("test_close_luhar_pdf:")
    for t in (test_f2py_oracle, test_moment_reconstruction, test_differentiable):
        t()
    print("All close_Luhar_pdf checks PASSED")


if __name__ == "__main__":
    main()
