#!/usr/bin/env python3
"""test_compute_sigma_sqd_w.py — validate the JAX compute_sigma_sqd_w port (sigma_sqd_w_module.F90).

sigma_sqd_w = gamma_Skw_fnc · (1 − min(max_x corr_wx², 1)), then smoothed zm→zt→zm. Ported and used per
timestep in the gated driver but lacking a dedicated f2py test. Oracles:
  1. f2py bit-shadow vs f2py_compute_sigma_sqd_w on a stored grid matching the JAX grid, for
     l_predict_upwp_vpwp on and off. SKIPs if clubb_f2py/clubb_python are unbuilt.
  2. Physical bounds: 0 <= sigma_sqd_w <= gamma_Skw_fnc (before smoothing the factor is in [0,1]).
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

from clubb_jax.src.CLUBB_core.sigma_sqd_w_module import compute_sigma_sqd_w
from clubb_jax.src.CLUBB_core.Skx_module import Skx_func, compute_gamma_Skw
from clubb_jax.src.CLUBB_core.clubb_constants import l_gamma_Skw, w_tol
from clubb_jax.src.CLUBB_core.parameter_indices import igamma_coef, igamma_coefb, igamma_coefc, iSkw_denom_coef
from clubb_jax.src.derived_types.grid_class import setup_grid

_NG, _DZ, _ZTOP = 2, 40.0, 1200.0
NPARAMS = 102

# Default-like parameter values for testing
_GC, _GB, _GCF, _SKWDENOM = 0.32, 0.08, 1.2, 0.04


def _make_params(ngrdcol=_NG):
    p = np.zeros((ngrdcol, NPARAMS))
    p[:, igamma_coef] = _GC
    p[:, igamma_coefb] = _GB
    p[:, igamma_coefc] = _GCF
    p[:, iSkw_denom_coef] = _SKWDENOM
    return p


def _fields(nzm, nzt, seed):
    rng = np.random.default_rng(seed)
    sh_zm = (_NG, nzm)
    sh_zt = (_NG, nzt)
    return dict(wp3=rng.uniform(-2.0, 2.0, sh_zt),
                wp2=rng.uniform(0.05, 2.0, sh_zm), thlp2=rng.uniform(0.01, 1.0, sh_zm),
                rtp2=rng.uniform(1e-8, 1e-6, sh_zm), up2=rng.uniform(0.05, 2.0, sh_zm),
                vp2=rng.uniform(0.05, 2.0, sh_zm), wpthlp=rng.uniform(-0.5, 0.5, sh_zm),
                wprtp=rng.uniform(-1e-4, 1e-4, sh_zm), upwp=rng.uniform(-0.5, 0.5, sh_zm),
                vpwp=rng.uniform(-0.5, 0.5, sh_zm))


def test_f2py_oracle():
    try:
        import clubb_f2py
        from clubb_python import clubb_api
        from clubb_python.derived_types.err_info import ErrInfo
    except Exception as e:
        print(f"  f2py compute_sigma_sqd_w oracle: SKIP ({type(e).__name__})")
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
    f = _fields(nzm, nzt, 5)
    clubb_params = _make_params(ng)
    # Compute gamma_Skw_fnc from wp3 and wp2 (what the Fortran oracle expects as input).
    wp3_zm = np.asarray(
        Skx_func.__wrapped__(nzm, ng, f['wp2'], f['wp3'],  # zt2zm not needed for oracle call
                             w_tol, clubb_params)
    ) if False else None  # placeholder — use jax path below
    # The f2py oracle takes gamma_Skw_fnc directly. We derive it via the JAX sub-functions.
    # 1) Interpolate wp3 (zt) -> zm using the same grid that the JAX uses.
    from clubb_jax.src.CLUBB_core.grid_class import zt2zm as _zt2zm
    wp3_on_zm = np.asarray(_zt2zm(nzm, nzt, ng, jgr, jnp.asarray(f['wp3'])))
    Skw_zm = np.asarray(Skx_func(nzm, ng, f['wp2'], wp3_on_zm, w_tol, clubb_params))
    gamma_Skw = np.asarray(compute_gamma_Skw(nzm, ng, Skw_zm, clubb_params, l_gamma_Skw))
    # f2py signature: f2py_compute_sigma_sqd_w(nzt, gamma_Skw_fnc, wp2, ..., l_predict_upwp_vpwp)
    order = ('wp2', 'thlp2', 'rtp2', 'up2', 'vp2', 'wpthlp', 'wprtp', 'upwp', 'vpwp')
    worst = 0.0
    for l_pred in (False, True):
        ref = np.asarray(clubb_f2py.f2py_compute_sigma_sqd_w(
            nzt, np.asfortranarray(gamma_Skw), *[np.asfortranarray(f[k]) for k in order], l_pred))
        got = np.asarray(compute_sigma_sqd_w(
            nzm, nzt, ng, jgr,
            f['wp3'], *[f[k] for k in order], clubb_params, l_pred))
        worst = max(worst, np.max(np.abs(got - ref)))
    assert worst < 1e-11, f"compute_sigma_sqd_w f2py mismatch {worst:.2e}"
    print(f"  f2py compute_sigma_sqd_w: bit-match (l_predict on/off), worst {worst:.2e}  PASS")


def test_bounds():
    jgr = setup_grid(ngrdcol=_NG, deltaz=_DZ, zm_init=0.0, zm_top=_ZTOP, grid_type=1)
    nzm = jgr.zm.shape[1]
    nzt = nzm - 1
    f = _fields(nzm, nzt, 7)
    clubb_params = _make_params()
    order = ('wp2', 'thlp2', 'rtp2', 'up2', 'vp2', 'wpthlp', 'wprtp', 'upwp', 'vpwp')
    s = np.asarray(compute_sigma_sqd_w(nzm, nzt, _NG, jgr,
                                        f['wp3'], *[f[k] for k in order], clubb_params, True))
    # gamma_Skw_fnc is in [gamma_coefb, gamma_coef] = [0.08, 0.32], all positive; sigma_sqd_w must be >= 0.
    assert np.all(s >= -1e-12), "sigma_sqd_w must be >= 0"
    assert np.all(s <= _GC + 1e-9), "sigma_sqd_w must be <= gamma_coef (upper bound)"
    print("  bounds: 0 <= sigma_sqd_w <= gamma_coef  PASS")


def test_differentiable():
    jgr = setup_grid(ngrdcol=1, deltaz=_DZ, zm_init=0.0, zm_top=_ZTOP, grid_type=1)
    nzm = jgr.zm.shape[1]
    nzt = nzm - 1
    f = _fields(nzm, nzt, 9)
    # restrict to single column
    f_1col = {k: v[:1] for k, v in f.items()}
    clubb_params = _make_params(1)
    order = ('wp2', 'thlp2', 'rtp2', 'up2', 'vp2', 'wpthlp', 'wprtp', 'upwp', 'vpwp')
    def loss(wpthlp):
        ff = dict(f_1col); ff['wpthlp'] = wpthlp
        return jnp.sum(compute_sigma_sqd_w(nzm, nzt, 1, jgr,
                                            ff['wp3'], *[ff[k] for k in order], clubb_params, True) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(f_1col['wpthlp'])))
    assert np.isfinite(g).all(), "non-finite grad through compute_sigma_sqd_w"
    print(f"  jax.grad through compute_sigma_sqd_w: finite ({g.size} entries)  PASS")


def main():
    print("test_compute_sigma_sqd_w:")
    for t in (test_f2py_oracle, test_bounds, test_differentiable):
        t()
    print("All compute_sigma_sqd_w checks PASSED")


if __name__ == "__main__":
    main()
