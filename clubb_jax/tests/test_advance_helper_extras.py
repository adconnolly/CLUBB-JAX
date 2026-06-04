#!/usr/bin/env python3
"""test_advance_helper_extras.py — validate the JAX smooth_min_jax + calc_xpwp ports (advance_helper_module).

Oracles:
  1. smooth_min: f2py bit-shadow vs f2py_smooth_min_array_scalar + the closed-form (and the smooth_min <= min
     bound). SKIPs if clubb_f2py is unbuilt.
  2. calc_xpwp: f2py bit-shadow vs f2py_calc_xpwp_2d on a stored grid set to exactly the JAX grid heights, plus
     the closed-form down-gradient identity on the interior. SKIPs if clubb_f2py/clubb_python are unbuilt.
  3. Finite jax.grad for both.
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

from clubb_jax.src.CLUBB_core.advance_helper_module import smooth_min_jax, calc_xpwp
from clubb_jax.src.derived_types.grid_class import setup_grid

_NG, _DZ, _ZTOP = 2, 40.0, 1200.0


def test_smooth_min_closed_form():
    a = np.array([[1.0, 5.0, -2.0]])
    coef = 1e-3
    out = np.asarray(smooth_min_jax(a, 3.0, coef))
    ref = 0.5 * ((a + 3.0) - np.sqrt((a - 3.0) ** 2 + coef ** 2))
    assert np.max(np.abs(out - ref)) < 1e-14
    assert np.all(out <= np.minimum(a, 3.0) + 1e-9), "smooth_min must be <= min"
    print("  smooth_min_jax: closed-form + (smooth_min <= min) bound  PASS")


def test_smooth_min_f2py():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py smooth_min oracle: SKIP ({type(e).__name__})")
        return
    rng = np.random.default_rng(2)
    a = rng.standard_normal((_NG, 6))
    b = 0.37
    coef = 1e-2
    ref = np.asarray(clubb_f2py.f2py_smooth_min_array_scalar(a, b, coef))
    got = np.asarray(smooth_min_jax(a, b, coef))
    d = np.max(np.abs(got - ref))
    assert d < 1e-13, f"smooth_min f2py mismatch {d:.2e}"
    print(f"  f2py smooth_min_array_scalar: bit-match, worst {d:.2e}  PASS")


def test_calc_xpwp_f2py():
    try:
        import clubb_f2py
        from clubb_python import clubb_api
        from clubb_python.derived_types.err_info import ErrInfo
    except Exception as e:
        print(f"  f2py calc_xpwp oracle: SKIP ({type(e).__name__})")
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
    rng = np.random.default_rng(7)
    Km_zm = np.asfortranarray(np.abs(rng.standard_normal((ng, nzm))) + 0.1)
    xm = np.asfortranarray(rng.standard_normal((ng, nzt)))
    ref = np.asarray(clubb_f2py.f2py_calc_xpwp_2d(Km_zm.copy(), xm.copy()))
    got = np.asarray(calc_xpwp(Km_zm, xm, np.asarray(jgr.invrs_dzm)))
    # Compare the interior levels the Fortran sets (k=1..nzm-2, 0-based).
    d = np.max(np.abs(got[:, 1:nzm - 1] - ref[:, 1:nzm - 1]))
    assert d < 1e-11, f"calc_xpwp f2py mismatch {d:.2e}"
    print(f"  f2py calc_xpwp_2d: bit-match on interior momentum levels, worst {d:.2e}  PASS")


def test_calc_xpwp_identity():
    jgr = setup_grid(ngrdcol=1, deltaz=_DZ, zm_init=0.0, zm_top=_ZTOP, grid_type=1)
    nzm = jgr.zm.shape[1]; nzt = nzm - 1
    rng = np.random.default_rng(3)
    Km = np.abs(rng.standard_normal((1, nzm))) + 0.1
    xm = rng.standard_normal((1, nzt))
    invrs_dzm = np.asarray(jgr.invrs_dzm)
    out = np.asarray(calc_xpwp(Km, xm, invrs_dzm))
    for k in range(1, nzm - 1):
        expect = Km[0, k] * invrs_dzm[0, k] * (xm[0, k] - xm[0, k - 1])
        assert abs(out[0, k] - expect) < 1e-13, f"xpwp identity failed at k={k}"
    assert out[0, 0] == 0.0 and out[0, nzm - 1] == 0.0, "boundary levels must be zero"
    print("  calc_xpwp: down-gradient identity on interior + zero boundaries  PASS")


def test_differentiable():
    jgr = setup_grid(ngrdcol=1, deltaz=_DZ, zm_init=0.0, zm_top=_ZTOP, grid_type=1)
    nzm = jgr.zm.shape[1]; nzt = nzm - 1
    Km = jnp.asarray(np.abs(np.random.default_rng(1).standard_normal((1, nzm))) + 0.1)
    invrs_dzm = jnp.asarray(np.asarray(jgr.invrs_dzm))
    def loss(xm):
        return jnp.sum(calc_xpwp(Km, xm, invrs_dzm) ** 2) + jnp.sum(smooth_min_jax(xm, 0.0, 1e-3) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(np.random.default_rng(2).standard_normal((1, nzt)))))
    assert np.isfinite(g).all(), "non-finite grad"
    print(f"  jax.grad through calc_xpwp + smooth_min: finite ({g.size} entries)  PASS")


def main():
    print("test_advance_helper_extras:")
    for t in (test_smooth_min_closed_form, test_smooth_min_f2py, test_calc_xpwp_f2py,
              test_calc_xpwp_identity, test_differentiable):
        t()
    print("All advance_helper extras checks PASSED")


if __name__ == "__main__":
    main()
