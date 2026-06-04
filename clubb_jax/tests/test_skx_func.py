#!/usr/bin/env python3
"""test_skx_func.py — validate the JAX skx_func_jax port (Skx_module.F90:skx_func).

Skx = xp3 · (xp2 + iSkw_denom_coef·x_tol²)^(-3/2) — the sensitivity-reduced skewness. Ported and used throughout
the PDF closure, but had no dedicated f2py test. Oracles:
  1. f2py bit-shadow vs f2py_skx_func, passing the SAME clubb_params array to both (only iSkw_denom_coef=73 is
     read), so the comparison isolates the JAX transcription of the (xp2 + denom)^(-3/2) algebra. SKIPs if
     clubb_f2py is unbuilt.
  2. Closed-form identity + the symmetric (xp3=0 -> Skx=0) limit.
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

from clubb_jax.src.CLUBB_core.advance_xp3_module import skx_func_jax

NG, NZ = 2, 6
NPARAMS = 102
I_SKW_DENOM = 73   # 1-based iSkw_denom_coef
_DENOM_COEF = 0.15
_X_TOL = 1.0e-2


def _params():
    p = np.zeros((NG, NPARAMS))
    p[:, I_SKW_DENOM - 1] = _DENOM_COEF
    return p


def test_f2py_oracle():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py skx_func oracle: SKIP ({type(e).__name__})")
        return
    rng = np.random.default_rng(4)
    params = _params()
    worst = 0.0
    for _ in range(10):
        xp2 = rng.uniform(0.0, 2.0, (NG, NZ))
        xp3 = rng.uniform(-1.0, 1.0, (NG, NZ))
        ref = np.asarray(clubb_f2py.f2py_skx_func(xp2, xp3, _X_TOL, params))
        got = np.asarray(skx_func_jax(xp2, xp3, _X_TOL, params))
        worst = max(worst, np.max(np.abs(got - ref)))
    assert worst < 1e-11, f"skx_func f2py mismatch {worst:.2e}"
    print(f"  f2py skx_func: bit-match over 10 configs, worst {worst:.2e}  PASS")


def test_closed_form():
    params = _params()
    xp2 = np.array([[0.5, 1.0]]); xp3 = np.array([[0.2, -0.3]])
    got = np.asarray(skx_func_jax(xp2, xp3, _X_TOL, params))
    denom = _DENOM_COEF * _X_TOL ** 2
    ref = xp3 * (xp2 + denom) ** (-1.5)
    assert np.max(np.abs(got - ref)) < 1e-12, "closed-form mismatch"
    # xp3 = 0 -> Skx = 0.
    z = np.asarray(skx_func_jax(xp2, np.zeros_like(xp3), _X_TOL, params))
    assert np.all(z == 0.0), "xp3=0 should give Skx=0"
    print("  closed-form identity + xp3=0 limit  PASS")


def test_differentiable():
    params = _params()
    xp2 = jnp.asarray(np.random.default_rng(1).uniform(0.1, 2.0, (NG, NZ)))
    def loss(xp3):
        return jnp.sum(skx_func_jax(xp2, xp3, _X_TOL, params) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(np.random.default_rng(2).uniform(-1, 1, (NG, NZ)))))
    assert np.isfinite(g).all(), "non-finite grad through skx_func_jax"
    print(f"  jax.grad through skx_func_jax: finite ({g.size} entries)  PASS")


def main():
    print("test_skx_func:")
    for t in (test_f2py_oracle, test_closed_form, test_differentiable):
        t()
    print("All skx_func checks PASSED")


if __name__ == "__main__":
    main()
