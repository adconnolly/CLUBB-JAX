"""JAX port of advance_xp2_xpyp_module.F90 — LHS/RHS assembly for the xp2/xpyp equations.

Mirrors clubb_release/src/CLUBB_core/advance_xp2_xpyp_module.F90. The full
rtp2/thlp2/rtpthlp/up2/vp2 solve is the module-level driver `advance_xp2_xpyp` (extracted out of
advance_clubb_core_module.py, mirror-refactor iter 139/140 — advance_clubb_core now *calls* it),
which drives the standalone tridiagonal assembly helpers also held here (the ones that genuinely
belong to advance_xp2_xpyp_module.F90, not diffusion.F90):
  term_dp1_lhs   — dissipation-term-1 main-diagonal coefficient
  term_dp1_rhs   — dissipation-term-1 explicit (threshold) RHS
  xp2_xpyp_lhs   — full tridiagonal LHS assembly (diff + ma + ta + dp1 + 1/dt)
  xp2_xpyp_rhs   — full explicit RHS (TA over-implicit + TP + DP1 + forcing + tendency)
  update_xp2_mc  — rain-evaporation tendencies of the second moments (l_morr_xp2_mc)

Output layout lhs[3, ngrdcol, nzm]: [0]=superdiagonal, [1]=main, [2]=subdiagonal.
Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level).
Pure-jnp → differentiable.

References:
  src/CLUBB_core/advance_xp2_xpyp_module.F90, term_dp1_lhs/rhs, xp2_xpyp_lhs/rhs, update_xp2_mc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset
from clubb_jax.src.CLUBB_core.grid_class import zm2zt, zm2zt_jax
from clubb_jax.src.CLUBB_core.turbulent_adv_pdf import (
    xpyp_term_ta_pdf_lhs_jax,
    xpyp_term_ta_pdf_rhs_jax,
)
from clubb_jax.src.CLUBB_core.tridiag_lu_solver import tridiag_lu_solve_jax
from clubb_jax.src.CLUBB_core.diffusion import diffusion_zm_lhs_jax
from clubb_jax.src.CLUBB_core.fill_holes import fill_holes_vertical
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zm_lhs
from clubb_jax.src.CLUBB_core.clip_explicit import clip_covar, clip_variance
# precip_frac_double_delta_jax lives in its precip-fraction Fortran home (mirror-refactor iter 240);
# update_xp2_mc calls it here, mirroring `use precipitation_fraction`.
from clubb_jax.src.CLUBB_core.precipitation_fraction import precip_frac_double_delta_jax

from clubb_jax.src.CLUBB_core.constants_clubb import (
    cloud_frac_min, Cp, Lv, grav, zero_threshold,
    rt_tol, thl_tol, w_tol_sqd, max_mag_correlation, max_mag_correlation_flux,
    gamma_over_implicit_ts,
    ibeta, iC2rt, ic_K2, iC4, iC14, ic_K9, iC_uu_shr, iC_uu_buoy,
)
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax

jax.config.update("jax_enable_x64", True)   # update_xp2_mc's conservation arithmetic needs float64


def _clip_variance(field, threshold):
    """clip_variance (clip_explicit.F90) with the JAX tracer-convention wrapping —
    floor a variance field to `threshold` over levels 0..nzm-2 (the post-solve variance clip used by
    advance_xp2_xpyp). Mirrors the helper formerly in advance_clubb_core_module.py."""
    return _asarray(clip_variance(jnp.asarray(field), threshold))


def apply_lhs_band3_interior_jax(lhs3, x):
    """Apply a 3-band (tridiagonal) zm LHS operator to a field at interior levels:
    ``lhs3[0]*x[k+1] + lhs3[1]*x[k] + lhs3[2]*x[k-1]`` (zero at the boundary levels).

    This is the implicit-budget-finalize kernel used by the xp2_xpyp budget diagnostics —
    the JAX analog of the ``lhs(t_kp1)*x(kp1) + lhs(t_k)*x(k) + lhs(t_km1)*x(km1)`` matrix-vector
    product that advance_xp2_xpyp_module.F90:stats_finalize_xp2_xpyp_terms forms for the dp2/ta
    implicit contributions. Tracer-transparent (``_xp``/``_iset``) so it stays on the autodiff graph.
    """
    r = _xp.zeros_like(x)
    r = _iset(r, np.s_[:, 1:-1], (lhs3[0, :, 1:-1] * x[:, 2:]
                + lhs3[1, :, 1:-1] * x[:, 1:-1]
                + lhs3[2, :, 1:-1] * x[:, :-2]))
    return r


def apply_lhs_band2_zt2zm_interior_jax(lhs2, x):
    """Apply a 2-band zt-level LHS to a zt field, producing a zm-level result at interior levels:
    ``r[k] = lhs2[0,k]*x[k] + lhs2[1,k]*x[k-1]`` (zero at the boundary zm levels), where ``x`` is on the
    zt grid (nzt = nzm-1) and ``lhs2`` is (2, ngrdcol, nzm) so the result shape is taken from ``lhs2[0]``.

    The JAX form of the 2-band zt→zm turbulent-coupling apply shared by the wp2 turbulent-advection budget
    term (``wp2_ta``) and the wpxp turbulent-production budget term (``wprtp_tp``/``wpthlp_tp``). Tracer-
    transparent (``_xp``/``_iset``) so it stays on the autodiff graph.
    """
    r = _xp.zeros_like(lhs2[0])
    r = _iset(r, np.s_[:, 1:-1], (lhs2[0, :, 1:-1] * x[:, 1:]
                + lhs2[1, :, 1:-1] * x[:, :-1]))
    return r


def finalize_implicit_budget_interior_jax(rhs, lhs, field):
    """Finalize a partly-implicit budget term at interior levels: ``rhs[k] - lhs[k]*field[k]``
    (zero at the boundary levels). The JAX form of the begin(rhs) + finalize(-lhs_diagonal*field) sequence
    the Fortran budget code uses for the diagonal implicit pr1/pr2/dp1 contributions (wp2/wp3 budgets).
    Tracer-transparent (``_xp``/``_iset``) so it stays on the autodiff graph.
    """
    r = _xp.zeros_like(rhs)
    r = _iset(r, np.s_[:, 1:-1], rhs[:, 1:-1] - lhs[:, 1:-1] * field[:, 1:-1])
    return r


def term_dp1_lhs(
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
) -> jnp.ndarray:
    """Main-diagonal coefficient for dissipation term 1 of x_a'x_b' equations.

    Faithful port of Fortran term_dp1_lhs (advance_xp2_xpyp_module.F90).

    The d(x_a'x_b')/dt equation contains the implicit dissipation:
      + (C_n / tau_zm) * x_a'x_b'(t+1)
    which contributes only to the main diagonal of the tridiagonal system.

    Fortran index conventions (1-based → 0-based Python):
      k=1     → k=0        (lower boundary, zero)
      k=2..nzm-1 → k=1..nzm-2 (interior)
      k=nzm   → k=nzm-1   (upper boundary, zero)

    Args:
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm levels, shape (ngrdcol, nzm).

    Returns:
        lhs: shape (ngrdcol, nzm). Main diagonal only — no super/sub-diagonal.
    """
    interior = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]
    ngrdcol = Cn.shape[0]
    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=Cn.dtype)
    return jnp.concatenate([zeros_bnd, interior, zeros_bnd], axis=1)


# JIT-compiled production version
term_dp1_lhs = jit(term_dp1_lhs)


_GAMMA_OVER_IMPLICIT_TS = gamma_over_implicit_ts   # constants_clubb.F90 (imported above)


def xp2_xpyp_lhs(
    lhs_ta: jnp.ndarray,
    lhs_ma: jnp.ndarray,
    lhs_diff: jnp.ndarray,
    lhs_dp1: jnp.ndarray,
    dt: float,
    gamma: float = _GAMMA_OVER_IMPLICIT_TS,
) -> jnp.ndarray:
    """Assemble full tridiagonal LHS for xp2/xpyp equations.

    Faithful port of Fortran xp2_xpyp_lhs (advance_xp2_xpyp_module.F90).

    For interior k=1..nzm-2 (Python):
      lhs[0,k] = lhs_diff[0,k] + lhs_ma[0,k] + lhs_ta[0,k] * gamma
      lhs[1,k] = lhs_diff[1,k] + lhs_ma[1,k] + lhs_ta[1,k] * gamma
                 + lhs_dp1[k]  (pre-scaled by caller)  + 1/dt
      lhs[2,k] = lhs_diff[2,k] + lhs_ma[2,k] + lhs_ta[2,k] * gamma

    Boundaries k=0 and k=nzm-1: lhs = [0, 1, 0] (fixed-value BCs).

    Args:
        lhs_ta:   Turbulent-advection LHS,   shape (3, ngrdcol, nzm).
        lhs_ma:   Mean-advection LHS,         shape (3, ngrdcol, nzm).
        lhs_diff: Eddy-diffusion LHS,         shape (3, ngrdcol, nzm).
        lhs_dp1:  Dissipation-term-1 diagonal, shape (ngrdcol, nzm),
                  already multiplied by gamma_over_implicit_ts by the caller.
        dt:       Timestep [s].
        gamma:    Over-implicit weight for turbulent-advection terms (default 1.5).

    Returns:
        lhs: shape (3, ngrdcol, nzm).
    """
    ngrdcol = lhs_ta.shape[1]

    # Interior k=1..nzm-2
    super_int = lhs_diff[0, :, 1:-1] + lhs_ma[0, :, 1:-1] + lhs_ta[0, :, 1:-1] * gamma
    main_int  = (lhs_diff[1, :, 1:-1] + lhs_ma[1, :, 1:-1] + lhs_ta[1, :, 1:-1] * gamma
                 + lhs_dp1[:, 1:-1] + 1.0 / dt)
    sub_int   = lhs_diff[2, :, 1:-1] + lhs_ma[2, :, 1:-1] + lhs_ta[2, :, 1:-1] * gamma

    # Boundaries: fixed-value BC → [super=0, main=1, sub=0]
    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=lhs_ta.dtype)
    ones_bnd  = jnp.ones((ngrdcol, 1),  dtype=lhs_ta.dtype)

    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag  = jnp.concatenate([ones_bnd,  main_int,  ones_bnd],  axis=1)
    subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)  # (3, ngrdcol, nzm)


# JIT-compiled production version
xp2_xpyp_lhs = jit(xp2_xpyp_lhs)


def term_dp1_rhs(
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    """Explicit portion of dissipation term 1 for x'y' equations (all levels).

    Faithful port of Fortran term_dp1_rhs (advance_xp2_xpyp_module.F90:5727).

    The explicit part of -(C_n/tau_zm) * (x'y' - threshold) is
    +(C_n/tau_zm) * threshold, applied at every level including boundaries.

    Args:
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm, shape (ngrdcol, nzm).
        threshold:    Scalar minimum value for x'y'.

    Returns:
        rhs: shape (ngrdcol, nzm).  All levels, no boundary zeroing.
    """
    return Cn * invrs_tau_zm * threshold


# JIT-compiled production version
term_dp1_rhs = jit(term_dp1_rhs)


def xp2_xpyp_rhs(
    lhs_ta: jnp.ndarray,
    rhs_ta: jnp.ndarray,
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
    threshold: float,
    xapxbp: jnp.ndarray,
    xam: jnp.ndarray,
    xbm: jnp.ndarray,
    wpxap: jnp.ndarray,
    wpxbp: jnp.ndarray,
    invrs_dzm: jnp.ndarray,
    xpyp_forcing: jnp.ndarray,
    dt: float,
    gamma: float = _GAMMA_OVER_IMPLICIT_TS,
) -> jnp.ndarray:
    """Explicit RHS of x'^2 and x'y' equations.

    Faithful port of Fortran xp2_xpyp_rhs (advance_xp2_xpyp_module.F90:3453).

    For interior levels k=1..nzm-2 (Python 0-based, ascending grid):
      rhs[k] = rhs_ta[k]
             + (1-gamma)*(-lhs_ta[0,k]*xapxbp[k+1]
                          -lhs_ta[1,k]*xapxbp[k]
                          -lhs_ta[2,k]*xapxbp[k-1])     [TA over-implicit]
             - wpxbp[k]*invrs_dzm[k]*(xam[k]-xam[k-1])  [TP term, part 1]
             - wpxap[k]*invrs_dzm[k]*(xbm[k]-xbm[k-1])  [TP term, part 2]
             + Cn[k]*invrs_tau_zm[k]*threshold            [DP1 explicit]
             + (1-gamma)*(-Cn[k]*invrs_tau_zm[k]*xapxbp[k]) [DP1 over-implicit]
             + xpyp_forcing[k]                            [forcing]
             + (1/dt)*xapxbp[k]                          [time tendency]

    Boundary conditions (ascending grid, fixed-value BCs):
      rhs[:, 0]  = xapxbp[:, 0]   (lower boundary: carry current value)
      rhs[:, -1] = threshold       (upper boundary: set to threshold)

    Band ordering: lhs_ta[0=super, 1=main, 2=sub], same as LHS convention.
    For ascending grid: super couples k to k+1 (above), sub couples k to k-1 (below).

    xam/xbm have shape (ngrdcol, nzt) where nzt=nzm-1.
    At interior k (Python 1..nzm-2): uses xam[:,1:]-xam[:,:-1] differences.

    Args:
        lhs_ta:       TA LHS, shape (3, ngrdcol, nzm). [0=super,1=main,2=sub].
        rhs_ta:       TA RHS, shape (ngrdcol, nzm).
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm, shape (ngrdcol, nzm).
        threshold:    Scalar minimum allowable value for x'y'.
        xapxbp:       x'y' on zm, shape (ngrdcol, nzm).
        xam:          x_am on zt, shape (ngrdcol, nzt).
        xbm:          x_bm on zt, shape (ngrdcol, nzt).
        wpxap:        w'x_a' on zm, shape (ngrdcol, nzm).
        wpxbp:        w'x_b' on zm, shape (ngrdcol, nzm).
        invrs_dzm:    1/dz on zm levels, shape (ngrdcol, nzm).
        xpyp_forcing: Forcing on zm, shape (ngrdcol, nzm).
        dt:           Timestep [s].
        gamma:        Over-implicit weight (default 1.5).

    Returns:
        rhs: shape (ngrdcol, nzm).
    """
    one_minus_gamma = 1.0 - gamma

    # term_tp_rhs: turbulent production at interior k=1..nzm-2 (the module routine)
    rhs_tp_int = term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm)

    # term_dp1_rhs: Cn * invrs_tau_zm * threshold (all levels, used at interior)
    rhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1] * threshold

    # term_dp1_lhs (interior, unscaled): Cn * invrs_tau_zm
    lhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]

    # Assemble interior RHS
    rhs_int = (
        rhs_ta[:, 1:-1]
        + one_minus_gamma * (
            - lhs_ta[0, :, 1:-1] * xapxbp[:, 2:]    # super * xapxbp[k+1]
            - lhs_ta[1, :, 1:-1] * xapxbp[:, 1:-1]  # main  * xapxbp[k]
            - lhs_ta[2, :, 1:-1] * xapxbp[:, :-2]   # sub   * xapxbp[k-1]
        )
        + rhs_tp_int
        + rhs_dp1_int
        + one_minus_gamma * (-lhs_dp1_int * xapxbp[:, 1:-1])
        + xpyp_forcing[:, 1:-1]
        + (1.0 / dt) * xapxbp[:, 1:-1]
    )

    # Boundary conditions: lower=current value, upper=threshold
    ngrdcol = Cn.shape[0]
    rhs_lb = xapxbp[:, 0:1]
    rhs_ub = jnp.full((ngrdcol, 1), threshold, dtype=Cn.dtype)

    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


# JIT-compiled production version
xp2_xpyp_rhs = jit(xp2_xpyp_rhs)


# ---------------------------------------------------------------------------
# update_xp2_mc — rain-evaporation effects on the second moments (l_morr_xp2_mc)
# ---------------------------------------------------------------------------
#
# Rain is assumed to fall through the moist (cold) portion of the PDF, treated as a double-delta with a
# precipitation fraction; evaporation makes the moist component moister and the cold component colder. The
# moment tendencies are computed on zt levels and interpolated to zm. Pure-jnp (top-down fill via lax.scan,
# zt2zm interpolation) → differentiable. l_morr_xp2_mc defaults to .false. and no gated case uses it; this is
# a completeness port validated against a literal NumPy transcription (`tests/test_update_xp2_mc.py`).


def pos_definite_variances(field, rho_ds_zm, dzm, threshold, hf_lower, hf_upper, fill_holes_type):
    """Hole-filling positive-definiteness for one variance field (advance_xp2_xpyp_module.F90:pos_definite_variances).

    Mirrors the Fortran subroutine's hole-filling core: a call to fill_holes_vertical over the zm interior
    (gr%k_lb_zm+dir .. gr%k_ub_zm-dir) when fill_holes_type /= 0. The Fortran subroutine also brackets this with
    the `<var>_pd` begin/finalize budget stats; in the JAX inlined solve those budget samples are taken at the
    caller (advance_clubb_core), so this routine returns just the filled field. Tracer-convention wrapper
    (`_asarray(fill_holes_vertical(jnp.asarray(field), ...))`)."""
    return _asarray(fill_holes_vertical(
        field=jnp.asarray(field), rho_ds=jnp.asarray(rho_ds_zm), dz=jnp.asarray(dzm),
        threshold=threshold, lower_k=hf_lower, upper_k=hf_upper, fill_holes_type=fill_holes_type))


def update_xp2_mc(gr, dt, cloud_frac, rcm, rvm, thlm, wm, exner, rrm_evap, pdf_params):
    """Rain-evaporation tendencies of the second moments on zm levels (advance_xp2_xpyp_module.F90:update_xp2_mc).

    Array inputs are (ngrdcol, nzt). pdf_params is a mapping (or object) providing mixt_frac, rt_1/rt_2,
    varnce_rt_1/2, thl_1/2, varnce_thl_1/2, w_1/2, varnce_w_1/2 (each (ngrdcol, nzt)). rrm_evap is the (negative)
    rain-evaporation rate. Returns (rtp2_mc, thlp2_mc, wprtp_mc, wpthlp_mc, rtpthlp_mc), each (ngrdcol, nzm)."""
    def _g(name):
        return jnp.asarray(pdf_params[name] if isinstance(pdf_params, dict) else getattr(pdf_params, name),
                           dtype=jnp.float64)

    cloud_frac = jnp.asarray(cloud_frac, dtype=jnp.float64)
    rcm = jnp.asarray(rcm, dtype=jnp.float64); rvm = jnp.asarray(rvm, dtype=jnp.float64)
    thlm = jnp.asarray(thlm, dtype=jnp.float64); wm = jnp.asarray(wm, dtype=jnp.float64)
    exner = jnp.asarray(exner, dtype=jnp.float64); rrm_evap = jnp.asarray(rrm_evap, dtype=jnp.float64)

    a = _g('mixt_frac')
    pf = precip_frac_double_delta_jax(cloud_frac)
    pf_const = jnp.where(pf > cloud_frac_min, (1.0 - pf) / jnp.where(pf > cloud_frac_min, pf, 1.0), 0.0)

    rt_tot = rcm + rvm
    temp_rtp2 = (a * ((_g('rt_1') - rt_tot) ** 2 + _g('varnce_rt_1'))
                 + (1.0 - a) * ((_g('rt_2') - rt_tot) ** 2 + _g('varnce_rt_2')))
    temp_thlp2 = (a * ((_g('thl_1') - thlm) ** 2 + _g('varnce_thl_1'))
                  + (1.0 - a) * ((_g('thl_2') - thlm) ** 2 + _g('varnce_thl_2')))
    temp_wp2 = (a * ((_g('w_1') - wm) ** 2 + _g('varnce_w_1'))
                + (1.0 - a) * ((_g('w_2') - wm) ** 2 + _g('varnce_w_2')))

    lvcpex = Lv / (Cp * exner)
    abse = jnp.abs(rrm_evap)

    rtp2_mc_zt = rrm_evap ** 2 * pf_const * dt + 2.0 * abse * jnp.sqrt(temp_rtp2 * pf_const)
    thlp2_mc_zt = (rrm_evap * lvcpex) ** 2 * pf_const * dt + 2.0 * abse * lvcpex * jnp.sqrt(temp_thlp2 * pf_const)
    wprtp_mc_zt = abse * jnp.sqrt(pf_const) * jnp.sqrt(temp_wp2)
    wpthlp_mc_zt = -lvcpex * abse * jnp.sqrt(pf_const) * jnp.sqrt(temp_wp2)
    rtpthlp_mc_zt = (-abse * jnp.sqrt(pf_const) * (lvcpex * jnp.sqrt(temp_rtp2) + jnp.sqrt(temp_thlp2))
                     - lvcpex * pf_const * rrm_evap ** 2 * dt)

    return (zt2zm_jax(rtp2_mc_zt, gr), zt2zm_jax(thlp2_mc_zt, gr), zt2zm_jax(wprtp_mc_zt, gr),
            zt2zm_jax(wpthlp_mc_zt, gr), zt2zm_jax(rtpthlp_mc_zt, gr))


def calc_xp2_xpyp_ta_lhs_jax(l_upwind, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, beta_col,
                             rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr):
    """Shared implicit turbulent-advection LHS for the xp2/xpyp equations, ADG1 path.

    Byte-identical port of the LHS half of advance_xp2_xpyp_module.F90:calc_xp2_xpyp_ta_terms — the operator
    depends only on the w-PDF (sign / coef of wp3_on_wp2), so it is the SAME (3, ngrdcol, nzm) array for all
    five moments (rtp2/thlp2/rtpthlp/up2/vp2); the caller computes it once. l_upwind selects the Godunov-upwind
    branch (gated default) vs the centered branch. `beta_col` is clubb_params[:, ibeta-1]."""
    _beta = _asarray(beta_col, dtype=np.float64)
    if l_upwind:
        _wp3 = _asarray(wp3_on_wp2, dtype=np.float64)
        _a1 = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
        _sgn = _xp.where(_wp3 >= 0, 1.0, -1.0)
        _coef_zm = (1.0 / 3.0) * _beta[:, np.newaxis] * _a1 * _wp3
        return _asarray(xpyp_term_ta_pdf_lhs_jax(
            None, None, jnp.asarray(invrs_rho_ds_zm), gr,
            l_upwind_xpyp_turbulent_adv=True,
            rho_ds_zm=jnp.asarray(rho_ds_zm), sgn_turbulent_vel=jnp.asarray(_sgn),
            coef_wpxpyp_implicit_zm=jnp.asarray(_coef_zm), grid_dir=float(gr.grid_dir),
        ), dtype=np.float64)
    else:
        _a1_zm = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
        _a1_zt = _asarray(zm2zt(jnp.array(_a1_zm), gr))
        _wp3_zt = _asarray(wp3_on_wp2_zt, dtype=np.float64)
        _coef_zt = (1.0 / 3.0) * _beta[:, np.newaxis] * _a1_zt * _wp3_zt
        return _asarray(xpyp_term_ta_pdf_lhs_jax(
            jnp.asarray(_coef_zt), jnp.asarray(rho_ds_zt), jnp.asarray(invrs_rho_ds_zm), gr,
        ))


def calc_xp2_xpyp_ta_rhs_jax(l_upwind, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w,
                             wp2, wp2_zt, beta_col, flux_a_zm, flux_b_zm,
                             rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr):
    """Turbulent-advection RHS term for one xp2/xpyp moment, ADG1 path.

    Byte-identical port of the RHS half of advance_xp2_xpyp_module.F90:calc_xp2_xpyp_ta_terms. The explicit
    term is wp_coef * <w'a'><w'b'> (variance: flux_a=flux_b → flux²; covariance e.g. rtpthlp: flux_a=wprtp,
    flux_b=wpthlp). The shared wp_coef (w-PDF only) is recomputed per call (deterministic), so the routine is
    self-contained per variable. l_upwind selects upwind (gated default) vs centered; `beta_col` is
    clubb_params[:, ibeta-1]. Returns rhs_ta on zm levels, NumPy float64."""
    _beta = _asarray(beta_col, dtype=np.float64)
    _fa = _asarray(flux_a_zm, dtype=np.float64)
    _fb = _asarray(flux_b_zm, dtype=np.float64)
    if l_upwind:
        _wp3 = _asarray(wp3_on_wp2, dtype=np.float64)
        _a1 = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
        _wp2n = _asarray(wp2, dtype=np.float64)
        _sgn = _xp.where(_wp3 >= 0, 1.0, -1.0)
        _wp_coef = (1.0 - (1.0 / 3.0) * _beta[:, np.newaxis]) * _a1 ** 2 * _wp3 / _wp2n
        return _asarray(xpyp_term_ta_pdf_rhs_jax(
            None, None, jnp.asarray(invrs_rho_ds_zm), gr,
            l_upwind_xpyp_turbulent_adv=True,
            rho_ds_zm=jnp.asarray(rho_ds_zm), sgn_turbulent_vel=jnp.asarray(_sgn),
            term_wpxpyp_explicit_zm=jnp.asarray(_wp_coef * _fa * _fb), grid_dir=float(gr.grid_dir),
        ), dtype=np.float64)
    else:
        _a1_zm = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
        _a1_zt = _asarray(zm2zt(jnp.array(_a1_zm), gr))
        _wp3_zt = _asarray(wp3_on_wp2_zt, dtype=np.float64)
        _wp2_zt = _asarray(wp2_zt, dtype=np.float64)
        _wp_coef_zt = (1.0 - (1.0 / 3.0) * _beta[:, np.newaxis]) * _a1_zt ** 2 * _wp3_zt / _wp2_zt
        _fa_zt = _asarray(zm2zt(jnp.array(_fa), gr))
        _fb_zt = _asarray(zm2zt(jnp.array(_fb), gr))
        return _asarray(xpyp_term_ta_pdf_rhs_jax(
            jnp.asarray(_wp_coef_zt * _fa_zt * _fb_zt),
            jnp.asarray(rho_ds_zt), jnp.asarray(invrs_rho_ds_zm), gr,
        ))


def calc_xp2_xpyp_ta_explicit_terms_jax(sigma_sqd_w, wp3_on_wp2_zt, wp2_zt, wprtp, wpthlp, beta, gr):
    """advance_xp2_xpyp_module.F90:calc_xp2_xpyp_ta_terms (lines 4603-4617) — the explicit w'x'w'y'
    turbulent-advection budget terms (the l_upwind_xpyp_ta + stats branch that overwrites the first
    assignment). With a1_zt = zm2zt(1/(1-sigma_sqd_w)) and
        wp_coef_zt = (1 - beta/3)·a1_zt²·wp3_on_wp2_zt / wp2_zt,
    returns (term_wprtp2 = wp_coef·wprtp_zt², term_wpthlp2 = wp_coef·wpthlp_zt²,
             term_wprtpthlp = wp_coef·wprtp_zt·wpthlp_zt). Tracer-transparent (numpy when concrete).
    """
    _a1_zm = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
    _a1_zt = _asarray(zm2zt(jnp.asarray(_a1_zm), gr))
    _beta = _asarray(beta, dtype=np.float64)
    _wp_coef_zt = ((1.0 - (1.0 / 3.0) * _beta[:, np.newaxis])
                   * _a1_zt**2
                   * _asarray(wp3_on_wp2_zt, dtype=np.float64)
                   / _asarray(wp2_zt, dtype=np.float64))
    _wpthlp_zt = _asarray(zm2zt(jnp.asarray(wpthlp), gr))
    _wprtp_zt = _asarray(zm2zt(jnp.asarray(wprtp), gr))
    return (_wp_coef_zt * _wprtp_zt**2,
            _wp_coef_zt * _wpthlp_zt**2,
            _wp_coef_zt * _wprtp_zt * _wpthlp_zt)


def calc_xp2_xpyp_lhs_jax(lhs_ta, lhs_ma, Kh_zt, c_K2, nu2,
                          invrs_rho_ds_zm, rho_ds_zt, Cn, invrs_tau_xp2_zm,
                          gamma, dt, gr):
    """rtp2/thlp2/rtpthlp shared implicit LHS assembly (advance_xp2_xpyp_module.F90 body).

    Kw2 = c_K2 * Kh_zt eddy diffusion + the Cn pressure-damping (dp1) term, combined with the
    shared turbulent-advection (lhs_ta) and mean-advection (lhs_ma) operators. The same LHS solves
    all three second moments (rtp2, thlp2, rtpthlp) under ADG1.

    Returns: (lhs, lhs_diff, dp1) — lhs_diff and dp1 are reused by the budget diagnostics.
    """
    Kw2 = jnp.asarray(c_K2) * jnp.asarray(Kh_zt)
    lhs_diff = diffusion_zm_lhs_jax(
        jnp.asarray(Kw2), jnp.asarray(nu2),
        jnp.asarray(invrs_rho_ds_zm), jnp.asarray(rho_ds_zt), gr)
    dp1 = term_dp1_lhs(jnp.asarray(Cn), jnp.asarray(invrs_tau_xp2_zm))
    lhs = xp2_xpyp_lhs(jnp.asarray(lhs_ta), jnp.asarray(lhs_ma),
                           lhs_diff, dp1 * gamma, dt)
    return lhs, lhs_diff, dp1


def term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm):
    """advance_xp2_xpyp_module.F90:term_tp_rhs — turbulent production of x_a'x_b' (explicit).

        rhs = -w'x_b'·d(x_am)/dz - w'x_a'·d(x_bm)/dz,   on interior momentum levels k=2..nzm-1.
    x_am/x_bm are on the zt grid (nzt=nzm-1); d/dz is the central momentum-level difference. Returns the
    interior slice (ngrdcol, nzm-2). Pure-jnp → differentiable.
    """
    return (-wpxbp[:, 1:-1] * invrs_dzm[:, 1:-1] * (xam[:, 1:] - xam[:, :-1])
            - wpxap[:, 1:-1] * invrs_dzm[:, 1:-1] * (xbm[:, 1:] - xbm[:, :-1]))


def term_pr1(C4, C14, xbp2, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd):
    """advance_xp2_xpyp_module.F90:term_pr1 — explicit pressure/dissipation term 1 for up2/vp2.

    The combined dissipation-term-1 + pressure-term-1 explicit RHS (interior momentum levels k=2..nzm-1):
        rhs = (1/3)·C4·(xbp2 + wp2)/tau_C4 - (1/3)·C14·(xbp2 + wp2)/tau_C14 + C14·w_tol^2/tau_C14,
    where `xbp2` is the *other* horizontal variance (v'^2 when solving u'^2, and vice versa). Returns the
    interior slice (ngrdcol, nzm-2). Pure-jnp → differentiable.
    """
    return ((1.0 / 3.0) * C4 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]
            - (1.0 / 3.0) * C14 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C14_zm[:, 1:-1]
            + C14 * invrs_tau_C14_zm[:, 1:-1] * w_tol_sqd)


def term_pr2(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, vpwp, um, vm, gr):
    """advance_xp2_xpyp_module.F90:term_pr2 — explicit pressure term 2 (PR2) for up2/vp2.

    The d(u'^2)/dt (and d(v'^2)/dt) equation's PR2 (same for both):
        rhs_pr2 = (2/3)[ C_uu_buoy (g/thv_ds) w'th_v'
                         + C_uu_shr ( -u'w' d(um)/dz - v'w' d(vm)/dz ) ],   floored to zero_threshold,
    on the interior momentum levels k=2..nzm-1 (Python [:, 1:-1]); d/dz is the central momentum-level
    difference. Returns the interior slice (ngrdcol, nzm-2). Pure-jnp → differentiable.
    """
    invrs_dzm = jnp.asarray(gr.invrs_dzm)
    um = jnp.asarray(um); vm = jnp.asarray(vm)
    du_dz = invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
    dv_dz = invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
    C_uu_buoy = jnp.asarray(C_uu_buoy); C_uu_shr = jnp.asarray(C_uu_shr)
    thv = jnp.asarray(thv_ds_zm); wpthvp = jnp.asarray(wpthvp)
    upwp = jnp.asarray(upwp); vpwp = jnp.asarray(vpwp)
    pr2 = (2.0 / 3.0) * (
        C_uu_buoy * (grav / thv[:, 1:-1]) * wpthvp[:, 1:-1]
        + C_uu_shr * (-upwp[:, 1:-1] * du_dz - vpwp[:, 1:-1] * dv_dz))
    return jnp.maximum(pr2, zero_threshold)


def calc_up2_vp2_lhs_jax(lhs_ta, lhs_ma, Kh_zt, c_K9, nu9,
                         invrs_rho_ds_zm, rho_ds_zt,
                         C4, C14, invrs_tau_C4_zm, invrs_tau_C14_zm,
                         gamma, dt, gr):
    """up2/vp2 implicit LHS assembly (advance_xp2_xpyp_module.F90:advance_xp2_xpyp body).

    Builds the shared LHS for the up2/vp2 (horizontal-velocity-variance) solve: the shared
    turbulent-advection (lhs_ta) and mean-advection (lhs_ma) operators plus the up2/vp2-specific
    Kw9 eddy diffusion and the C4/C14 pressure-damping (dp1) terms scaled by gamma-over-implicit.
    The same LHS solves both up2 and vp2 (ADG1).

    Returns: (lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14) — the latter three are reused by the
    up2/vp2 RHS build (xp2_xpyp_uv_rhs) and the dp2 budget diagnostic.
    """
    c_K9 = jnp.asarray(c_K9)
    Kw9_zt = c_K9[:, None] * jnp.asarray(Kh_zt)
    lhs_diff = diffusion_zm_lhs_jax(
        jnp.asarray(Kw9_zt), jnp.asarray(nu9),
        jnp.asarray(invrs_rho_ds_zm), jnp.asarray(rho_ds_zt), gr)
    inv_tau_C4 = jnp.asarray(invrs_tau_C4_zm)
    ng, nzm = inv_tau_C4.shape
    c4_1d = (2.0 / 3.0) * jnp.asarray(C4) * jnp.ones((ng, nzm))
    c14_1d = (1.0 / 3.0) * jnp.asarray(C14) * jnp.ones((ng, nzm))
    lhs_dp1_C4 = term_dp1_lhs(c4_1d, inv_tau_C4)
    lhs_dp1_C14 = term_dp1_lhs(c14_1d, jnp.asarray(invrs_tau_C14_zm))
    lhs_dp1 = (lhs_dp1_C4 + lhs_dp1_C14) * gamma
    lhs = xp2_xpyp_lhs(jnp.asarray(lhs_ta), jnp.asarray(lhs_ma),
                           lhs_diff, lhs_dp1, dt)
    return lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14


def xp2_xpyp_uv_rhs(rhs_ta_this, this_pre, other_pre, this_wp, this_dvel_dz,
                         lhs_splat, wp2, lhs_ta, C_uu_shr, C4, C14,
                         invrs_tau_C4_zm, invrs_tau_C14_zm, lhs_dp1_C4, lhs_dp1_C14,
                         pr2, omg, dt, w_tol_sqd, l_coriolis, fcor_y_col):
    """Explicit RHS for the up2 (or vp2) equation in advance_xp2_xpyp_module.F90:advance_xp2_xpyp — the
    pressure-rotation (C_uu) form. Byte-identical port of the inline build. Symmetric: for vp2 pass the
    v-quantities as `this_*`/`this_wp`/`this_dvel_dz` and the u-quantity's pre-solve value as `other_pre`
    (the C4/C14 isotropization term couples the two horizontal variances). All arrays NumPy float64 / tracer.

    `this_dvel_dz` is the interior wind-shear (ngrdcol, nzm-2); `pr2` is the interior PR2 term; `omg` = 1-gamma;
    `fcor_y_col` is fcor_y[:, newaxis] (only used when l_coriolis). Returns rhs (ngrdcol, nzm)."""
    ng, nzm = this_pre.shape
    rhs = np.zeros((ng, nzm), dtype=np.float64)
    rhs = _iset(rhs, np.s_[:, 1:-1], (
        rhs_ta_this[:, 1:-1]
        + 0.5 * lhs_splat[:, 1:-1] * wp2[:, 1:-1]
        + omg * (
            -lhs_ta[0, :, 1:-1] * this_pre[:, 2:]
            - lhs_ta[1, :, 1:-1] * this_pre[:, 1:-1]
            - lhs_ta[2, :, 1:-1] * this_pre[:, :-2]
        )
        + (1.0 - C_uu_shr) * (
            -this_wp[:, 1:-1] * this_dvel_dz
            - this_wp[:, 1:-1] * this_dvel_dz
        )
        + term_pr1(C4, C14, other_pre, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd)
        + omg * (-lhs_dp1_C4[:, 1:-1] - lhs_dp1_C14[:, 1:-1]) * this_pre[:, 1:-1]
        + pr2
        + (1.0 / dt) * this_pre[:, 1:-1]
    ))
    # Nontraditional Coriolis term (advance_xp2_xpyp_module.F90:772).
    if l_coriolis:
        rhs = _iset(rhs, np.s_[:, 1:-1], (rhs[:, 1:-1] - 2.0 * fcor_y_col * this_wp[:, 1:-1]))
    rhs = _iset(rhs, np.s_[:, 0], this_pre[:, 0])
    rhs = _iset(rhs, np.s_[:, -1], w_tol_sqd)
    return rhs


def xp2_xpyp_solve(lhs_assembled, rhs):
    """Tridiagonal solve of an assembled xp2/xpyp system — the JAX analog of
    advance_xp2_xpyp_module.F90:xp2_xpyp_solve, the typed wrapper that advance_xp2_xpyp's solve drivers call
    around the generic tridiag_solve (tridiag_lu_solver.F90). Returns the solution on zm levels, NumPy float64."""
    return _asarray(tridiag_lu_solve_jax(jnp.asarray(lhs_assembled), jnp.asarray(rhs)))


def solve_xp2_xpyp_jax(lhs_assembled, lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold,
                       xapxbp, xam, xbm, wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt,
                       gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Build the explicit RHS (xp2_xpyp_rhs) and tridiagonal-solve one xp2/xpyp moment with the
    pre-assembled shared LHS. Mirrors the per-variable solve inside advance_xp2_xpyp_module.F90:advance_xp2_xpyp
    (each of rtp2/thlp2/rtpthlp/up2/vp2 builds its RHS against the same LHS and tridiag-solves via xp2_xpyp_solve).
    Returns the solution on zm levels, NumPy float64. Pure (the helpers are pure-jnp + tracer-transparent) → differentiable."""
    rhs = xp2_xpyp_rhs(
        jnp.asarray(lhs_ta), jnp.asarray(rhs_ta), jnp.asarray(Cn), jnp.asarray(invrs_tau_zm),
        threshold, jnp.asarray(xapxbp), jnp.asarray(xam), jnp.asarray(xbm),
        jnp.asarray(wpxap), jnp.asarray(wpxbp), jnp.asarray(invrs_dzm), jnp.asarray(xpyp_forcing),
        dt, gamma,
    )
    return xp2_xpyp_solve(lhs_assembled, rhs)


def solve_xp2_xpyp_with_single_lhs(lhs_assembled, lhs_ta, Cn, invrs_tau_zm, invrs_dzm, dt, moments):
    """Solve a set of xp2/xpyp moments that all share one assembled LHS — the JAX analog of
    advance_xp2_xpyp_module.F90:solve_xp2_xpyp_with_single_lhs (the rtp2/thlp2/rtpthlp group, F90:664).
    Each entry of `moments` is the per-moment tuple
    ``(rhs_ta, threshold, xapxbp, xam, xbm, wpxap, wpxbp, xpyp_forcing)``; the LHS, lhs_ta, Cn,
    invrs_tau_zm, invrs_dzm and dt are shared. Returns one solved field per moment (NumPy float64),
    in the order given. Thin driver over `solve_xp2_xpyp_jax` — pure / differentiable."""
    return tuple(
        solve_xp2_xpyp_jax(lhs_assembled, lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold,
                           xapxbp, xam, xbm, wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt)
        for (rhs_ta, threshold, xapxbp, xam, xbm, wpxap, wpxbp, xpyp_forcing) in moments)



def advance_xp2_xpyp(
        Kh_zt,
        clubb_params,
        dt_advance,
        fcor_y,
        flags,
        gr,
        invrs_rho_ds_zm,
        invrs_tau_C14_zm,
        invrs_tau_C4_zm,
        invrs_tau_xp2_zm,
        l_sample,
        lhs_splat_wp2,
        ngrdcol,
        nu_vert_res_dep,
        nzm,
        rho_ds_zm,
        rho_ds_zt,
        rtm,
        rtp2,
        rtp2_forcing,
        rtpthlp,
        rtpthlp_forcing,
        sigma_sqd_w,
        stats_writer,
        thlm,
        thlp2,
        thlp2_forcing,
        thv_ds_zm,
        um,
        up2,
        upwp,
        vm,
        vp2,
        vpwp,
        wm_zm,
        wp2,
        wp2_zt,
        wp3_on_wp2,
        wp3_on_wp2_zt,
        wprtp,
        wpthlp,
        wpthvp,
):
    """Advance the rtp2/thlp2/rtpthlp/up2/vp2 second moments + their budget stats — the JAX driver
    mirroring advance_xp2_xpyp_module.F90:advance_xp2_xpyp (the solve via solve_xp2_xpyp_*_jax / the
    calc_*/term_* module routines, the post-solve pos_definite_variances + clip, and the interleaved
    budget-finalize stats which the Fortran keeps in stats_finalize_xp2_xpyp_terms). Returns the five
    updated variances/covariance (rtp2, thlp2, rtpthlp, up2, vp2); the caller does clip_covars_denom.
    Relocated verbatim from the inlined advance_clubb_core block (mirror-refactor iter 139)."""
    # ============================================================ #
    # xpyp turbulent-advection PDF LHS (xpyp_term_ta_pdf_lhs): uses
    # coef_wprtp2_implicit as the representative xpyp coefficient.
    # ============================================================ #

    # ============================================================ #
    # dp1 pressure-damping LHS term (_Cn_np/_inv_tau/_dp1_ref), assembled in
    # numpy here and fed (jnp.array'd) into the xp2_xpyp solve.
    # Uses C2rt (uniform in z) * invrs_tau_xp2_zm, boundaries zeroed.
    # ============================================================ #
    # inputs for the rtp2/thlp2/rtpthlp shared-LHS assembly. The Cn
    # pressure-damping (C2rt * invrs_tau_xp2_zm) and Kw2 = c_K2 * Kh_zt eddy diffusion are
    # combined with the shared TA/MA operators inside calc_xp2_xpyp_lhs_jax (below); _Cn_np /
    # _inv_tau are also passed to the per-moment solve, and _lhs_ma_f to the up2/vp2 LHS.
    _c2rt = float(clubb_params[0, iC2rt - 1])
    _Cn_np = np.full((ngrdcol, nzm), _c2rt, dtype=np.float64)
    _inv_tau = _asarray(invrs_tau_xp2_zm)
    _gamma = gamma_over_implicit_ts   # constants_clubb (advance_xp2_xpyp_module.F90:120)
    _dt_adv = float(dt_advance)
    _c_K2 = float(clubb_params[0, ic_K2 - 1])  # = 0.025
    _nu2_xp2 = _asarray(nu_vert_res_dep.nu2, dtype=np.float64)  # background eddy diff [m²/s]
    _lhs_ma_f = _asarray(term_ma_zm_lhs(jnp.asarray(wm_zm), gr))
    # lhs_ta already computed above

    # rtp2/thlp2/rtpthlp equation inputs (zm-level means/fluxes/forcing) for the
    # advance_xp2_xpyp_module.xp2_xpyp_rhs assembly below.
    _rtp2_np = _asarray(rtp2, dtype=np.float64).copy()
    _rtm_np  = _asarray(rtm,  dtype=np.float64).copy()
    _wprtp_np = _asarray(wprtp, dtype=np.float64).copy()
    _invrs_dzm_np = _asarray(gr.invrs_dzm, dtype=np.float64).copy()
    _rtp2_forcing_np = _asarray(rtp2_forcing, dtype=np.float64).copy()

    # ============================================================ #
    # advance_xp2_xpyp solve (upwind or centered ADG1 path): assemble the LHS/RHS from the module
    # routines (calc_xp2_xpyp_ta_lhs/rhs, term_dp1_lhs, diffusion_zm_lhs_jax, term_ma_zm_lhs, xp2_xpyp_lhs/rhs)
    # then tridiag-solve + clip. fill_holes/clip_variance are no-ops for the gated cases.
    # ============================================================ #
    # pre-solve up2/vp2 (the over-implicit term uses the t-level value)
    _up2_prev2 = _asarray(up2, dtype=np.float64).copy()
    _vp2_prev2 = _asarray(vp2, dtype=np.float64).copy()
    _thlm_np  = _asarray(thlm,  dtype=np.float64).copy()
    _thlp2_np = _asarray(thlp2, dtype=np.float64).copy()
    _wpthlp_np = _asarray(wpthlp, dtype=np.float64).copy()
    _thlp2_forcing_np = _asarray(thlp2_forcing, dtype=np.float64).copy()
    _rtpthlp_np = _asarray(rtpthlp, dtype=np.float64).copy()
    _rtpthlp_forcing_np = _asarray(rtpthlp_forcing, dtype=np.float64).copy()
    _threshold_thlp2 = float(thl_tol ** 2)
    _threshold_rtpthlp = float(zero_threshold)
    # Turbulent-advection LHS (shared, w-PDF only) + the 3 explicit RHS terms (rtp2: wprtp², thlp2:
    # wpthlp², rtpthlp: wprtp·wpthlp), relocated to advance_xp2_xpyp_module (mirror-refactor iter 28,
    # `calc_xp2_xpyp_ta_terms` — byte-identical, upwind+centered branches inside the helpers).
    _beta_colx2 = clubb_params[:, ibeta - 1]
    _lhs_ta_x2 = calc_xp2_xpyp_ta_lhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w,
        _beta_colx2, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    _rhs_tax2_rtp2 = calc_xp2_xpyp_ta_rhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, wp2, wp2_zt,
        _beta_colx2, wprtp, wprtp, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    _rhs_tax2_thlp2 = calc_xp2_xpyp_ta_rhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, wp2, wp2_zt,
        _beta_colx2, wpthlp, wpthlp, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    _rhs_tax2_rtpthlp = calc_xp2_xpyp_ta_rhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, wp2, wp2_zt,
        _beta_colx2, wprtp, wpthlp, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    # Shared rtp2/thlp2/rtpthlp LHS (Kw2 diffusion + Cn dp1 + TA/MA), via the module
    # routine (mirror-refactor iter 45). lhs_diff/dp1 are reused by the budget diagnostics.
    _lhsx2, _lhs_diff_f, _dp1_ref = calc_xp2_xpyp_lhs_jax(
        _lhs_ta_x2, _lhs_ma_f, Kh_zt, _c_K2, _nu2_xp2,
        invrs_rho_ds_zm, rho_ds_zt, _Cn_np, _inv_tau, _gamma, _dt_adv, gr)
    _lhsx2 = _asarray(_lhsx2)
    _lhs_diff_f = _asarray(_lhs_diff_f)
    _dp1_ref = _asarray(_dp1_ref)
    # Per-moment solve (build RHS via xp2_xpyp_rhs + tridiag-solve the shared LHS), relocated to
    # advance_xp2_xpyp_module.solve_xp2_xpyp_jax (mirror-refactor iter 31). rtpthlp is the covariance
    # (xam=rtm,xbm=thlm, wpxap=wprtp,wpxbp=wpthlp); rtp2/thlp2 are variances.
    # The 3 moments share one assembled LHS — Fortran solve_xp2_xpyp_with_single_lhs (F90:664).
    _solnx2_rtp2, _solnx2_thlp2, _solnx2_rtpthlp = solve_xp2_xpyp_with_single_lhs(
        _lhsx2, _lhs_ta_x2, _Cn_np, _inv_tau, _invrs_dzm_np, _dt_adv, (
            (_rhs_tax2_rtp2,    float(rt_tol ** 2),  _rtp2_np,    _rtm_np,  _rtm_np,  _wprtp_np, _wprtp_np,  _rtp2_forcing_np),
            (_rhs_tax2_thlp2,   _threshold_thlp2,    _thlp2_np,   _thlm_np, _thlm_np, _wpthlp_np, _wpthlp_np, _thlp2_forcing_np),
            (_rhs_tax2_rtpthlp, _threshold_rtpthlp,  _rtpthlp_np, _rtm_np,  _thlm_np, _wprtp_np, _wpthlp_np, _rtpthlp_forcing_np),
        ))
    # Apply l_lmm_stepping blending (0.5 * old + 0.5 * solution)
    if flags.l_lmm_stepping:
        _rtp2_jaxx2    = 0.5 * (_rtp2_np    + _solnx2_rtp2)
        _thlp2_jaxx2   = 0.5 * (_thlp2_np   + _solnx2_thlp2)
        _rtpthlp_jaxx2 = 0.5 * (_rtpthlp_np + _solnx2_rtpthlp)
    else:
        _rtp2_jaxx2    = _solnx2_rtp2
        _thlp2_jaxx2   = _solnx2_thlp2
        _rtpthlp_jaxx2 = _solnx2_rtpthlp

    # --- apply fill_holes to the rtp2/thlp2 solutions (mirrors the Fortran
    #     pos_definite_variances applied to rtp2/thlp2 after the solve) ---
    _hf_lower = gr.k_lb_zm + gr.grid_dir_indx  # Python 0-based
    _hf_upper = gr.k_ub_zm - gr.grid_dir_indx  # Python 0-based
    _rtp2_jaxx2_fh = pos_definite_variances(
        _rtp2_jaxx2.copy(), rho_ds_zm, gr.dzm, float(rt_tol**2),
        _hf_lower, _hf_upper, flags.fill_holes_type)
    _thlp2_jaxx2_fh = pos_definite_variances(
        _thlp2_jaxx2.copy(), rho_ds_zm, gr.dzm, float(thl_tol**2),
        _hf_lower, _hf_upper, flags.fill_holes_type)

    # Apply clip_variance (Fortran does this after pos_definite_variances).
    # When l_min_xp2_from_corr_wx, threshold is boosted to wpthlp^2/(wp2*corr^2)
    # to keep |corr(w,thl)| <= max_mag_correlation_flux=0.99 (constants_clubb.F90:348).
    # clip_variance (clip_explicit.F90): floor each variance over levels 0..nzm-2.
    if flags.l_min_xp2_from_corr_wx:
        _wp2_clip = _asarray(wp2, dtype=np.float64)
        _wpthlp_clip = _wpthlp_np  # zm-level, set at line ~1312
        _wprtp_clip = _asarray(wprtp, dtype=np.float64)
        _max_corr2 = max_mag_correlation_flux**2  # constants_clubb.F90:348
        _thr_thlp2 = _xp.maximum(_threshold_thlp2,
                                _wpthlp_clip**2 / (_wp2_clip * _max_corr2))
        _thr_rtp2 = _xp.maximum(float(rt_tol**2),
                               _wprtp_clip**2 / (_wp2_clip * _max_corr2))
        _thlp2_jaxx2_cv = _clip_variance(_thlp2_jaxx2_fh, jnp.asarray(_thr_thlp2))
        _rtp2_jaxx2_cv = _clip_variance(_rtp2_jaxx2_fh, jnp.asarray(_thr_rtp2))
    else:
        _thlp2_jaxx2_cv = _clip_variance(_thlp2_jaxx2_fh, _threshold_thlp2)
        _rtp2_jaxx2_cv = _clip_variance(_rtp2_jaxx2_fh, float(rt_tol**2))
    # Apply clip_covar to rtpthlp (Cauchy-Schwarz: |rtpthlp| <= 0.99*sqrt(rtp2*thlp2))
    # clip_covar for rtpthlp
    _rtpthlp_jaxx2_clip = _asarray(clip_covar(
        wpxp=jnp.asarray(_rtpthlp_jaxx2),
        wp2=jnp.asarray(_rtp2_jaxx2_cv),
        xp2=jnp.asarray(_thlp2_jaxx2_cv),
        max_mag_corr=max_mag_correlation,   # rtpthlp is not a flux → max_mag_correlation
    ))

    # ============================================================ #
    # Budget stats: rtp2/thlp2/rtpthlp + scalar TA terms           #
    # ============================================================ #
    if l_sample and stats_writer is not None:
        _dtx2 = float(dt_advance)
        _gx2 = _gamma   # 1.5
        _omgx2 = 1.0 - _gx2  # -0.5

        # Explicit w'x'w'y' TA budget terms (advance_xp2_xpyp_module:calc_xp2_xpyp_ta_terms,
        # the l_upwind_xpyp_ta + stats branch; ARM default)
        _term_wprtp2_st, _term_wpthlp2_st, _term_wprtpthlp_st = (
            calc_xp2_xpyp_ta_explicit_terms_jax(
                sigma_sqd_w, wp3_on_wp2_zt, wp2_zt, wprtp, wpthlp,
                clubb_params[:, ibeta - 1], gr))
        stats_writer.update("term_wprtp2_explicit", _term_wprtp2_st)
        stats_writer.update("term_wpthlp2_explicit", _term_wpthlp2_st)
        stats_writer.update("term_wprtpthlp_explicit", _term_wprtpthlp_st)

        # rtp2 budget terms
        _rt_thrx2 = float(rt_tol ** 2)
        _rtp2_mixx2 = _omgx2 * _rtp2_np + _gx2 * _solnx2_rtp2
        # rtp2 turbulent production (variance: xam=xbm=rtm, wpxap=wpxbp=wprtp) via term_tp_rhs
        _rtp2_tpx2 = _iset(_xp.zeros_like(_rtp2_np), np.s_[:, 1:-1],
            term_tp_rhs(_rtm_np, _rtm_np, _wprtp_np, _wprtp_np, _invrs_dzm_np))
        stats_writer.update("rtp2_ta",
            _rhs_tax2_rtp2 - apply_lhs_band3_interior_jax(_lhs_ta_x2, _rtp2_mixx2))
        stats_writer.update("rtp2_dp1",
            _dp1_ref * (_rt_thrx2 - _rtp2_mixx2))
        stats_writer.update("rtp2_dp2",
            -apply_lhs_band3_interior_jax(_lhs_diff_f, _solnx2_rtp2))
        stats_writer.update("rtp2_tp", _rtp2_tpx2)
        _rtp2_pdx2 = _xp.zeros_like(_rtp2_np)
        _rtp2_pdx2 = _iset(_rtp2_pdx2, np.s_[:, 1:-1], (
            (_rtp2_jaxx2_fh[:, 1:-1] - _solnx2_rtp2[:, 1:-1]) / _dtx2))
        stats_writer.update("rtp2_pd", _rtp2_pdx2)
        stats_writer.update("rtp2_zt",
            _asarray(jnp.maximum(
                zm2zt_jax(jnp.asarray(_rtp2_jaxx2_cv), gr),
                float(rt_tol ** 2)), dtype=np.float64))

        # thlp2 budget terms
        _thl_thrx2 = _threshold_thlp2
        _thlp2_mixx2 = _omgx2 * _thlp2_np + _gx2 * _solnx2_thlp2
        # thlp2 turbulent production (variance: xam=xbm=thlm, wpxap=wpxbp=wpthlp) via term_tp_rhs
        _thlp2_tpx2 = _iset(_xp.zeros_like(_thlp2_np), np.s_[:, 1:-1],
            term_tp_rhs(_thlm_np, _thlm_np, _wpthlp_np, _wpthlp_np, _invrs_dzm_np))
        stats_writer.update("thlp2_ta",
            _rhs_tax2_thlp2 - apply_lhs_band3_interior_jax(_lhs_ta_x2, _thlp2_mixx2))
        stats_writer.update("thlp2_dp1",
            _dp1_ref * (_thl_thrx2 - _thlp2_mixx2))
        stats_writer.update("thlp2_dp2",
            -apply_lhs_band3_interior_jax(_lhs_diff_f, _solnx2_thlp2))
        stats_writer.update("thlp2_tp", _thlp2_tpx2)
        _thlp2_pdx2 = _xp.zeros_like(_thlp2_np)
        _thlp2_pdx2 = _iset(_thlp2_pdx2, np.s_[:, 1:-1], (
            (_thlp2_jaxx2_fh[:, 1:-1] - _solnx2_thlp2[:, 1:-1]) / _dtx2))
        stats_writer.update("thlp2_pd", _thlp2_pdx2)
        stats_writer.update("thlp2_zt",
            _asarray(jnp.maximum(
                zm2zt_jax(jnp.asarray(_thlp2_jaxx2_cv), gr),
                float(thl_tol ** 2)), dtype=np.float64))

        # rtpthlp budget terms
        _rtpthlp_mixx2 = _omgx2 * _rtpthlp_np + _gx2 * _solnx2_rtpthlp
        # rtpthlp turbulent-production decomposition (advance_xp2_xpyp_module.F90:3760-3770): exactly as the
        # Fortran does, the two TP contributions are obtained by calling term_tp_rhs twice with one field-pair
        # zeroed — tp1 ← (xm_zeros, thlm, wprtp, wpxp_zeros); tp2 ← (rtm, xm_zeros, wpxp_zeros, wpthlp).
        _xm_zeros_zt = _xp.zeros_like(_rtm_np)
        _wpxp_zeros_zm = _xp.zeros_like(_wprtp_np)
        _tp1_intx2 = term_tp_rhs(_xm_zeros_zt, _thlm_np, _wprtp_np, _wpxp_zeros_zm, _invrs_dzm_np)
        _tp2_intx2 = term_tp_rhs(_rtm_np, _xm_zeros_zt, _wpxp_zeros_zm, _wpthlp_np, _invrs_dzm_np)
        _rtpthlp_tp1_x2 = _iset(_xp.zeros_like(_rtpthlp_np), np.s_[:, 1:-1], _tp1_intx2)
        _rtpthlp_tp2_x2 = _iset(_xp.zeros_like(_rtpthlp_np), np.s_[:, 1:-1], _tp2_intx2)
        stats_writer.update("rtpthlp_ta",
            _rhs_tax2_rtpthlp - apply_lhs_band3_interior_jax(_lhs_ta_x2, _rtpthlp_mixx2))
        stats_writer.update("rtpthlp_dp1",
            _dp1_ref * (_threshold_rtpthlp - _rtpthlp_mixx2))
        stats_writer.update("rtpthlp_dp2",
            -apply_lhs_band3_interior_jax(_lhs_diff_f, _solnx2_rtpthlp))
        stats_writer.update("rtpthlp_tp1", _rtpthlp_tp1_x2)
        stats_writer.update("rtpthlp_tp2", _rtpthlp_tp2_x2)
        # rtpthlp_cl: Cauchy-Schwarz clip_covar effect
        stats_writer.update("rtpthlp_cl",
            (_rtpthlp_jaxx2_clip - _solnx2_rtpthlp) / _dtx2)

    # ============================================================ #
    # advance_xp2_xpyp for up2/vp2 (horizontal velocity variances) #
    # LHS: same TA as rtp2 (ADG1); Kw9 diffusion; C4/C14 dp1.      #
    # ============================================================ #
    # up2/vp2 LHS assembly via advance_xp2_xpyp_module (mirror-refactor iter 42):
    # shared TA/MA (as rtp2) + Kw9 diffusion + C4/C14 dp1, same LHS for both.
    _C4_v2 = _asarray(clubb_params[:, iC4 - 1], dtype=np.float64)[:, np.newaxis]
    _C14_v2 = _asarray(clubb_params[:, iC14 - 1], dtype=np.float64)[:, np.newaxis]
    _invrs_tau_C4_zm_v2 = _asarray(invrs_tau_C4_zm, dtype=np.float64)
    _invrs_tau_C14_zm_v2 = _asarray(invrs_tau_C14_zm, dtype=np.float64)
    _lhs_uvv2, _lhs_diff_uvv2, _lhs_dp1_C4_v2, _lhs_dp1_C14_v2 = (
        calc_up2_vp2_lhs_jax(
            _lhs_ta_x2, _lhs_ma_f, Kh_zt,
            _asarray(clubb_params[:, ic_K9 - 1], dtype=np.float64),
            _asarray(nu_vert_res_dep.nu9, dtype=np.float64),
            _asarray(invrs_rho_ds_zm, dtype=np.float64),
            _asarray(rho_ds_zt, dtype=np.float64),
            _C4_v2, _C14_v2, _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2,
            _gamma, _dt_adv, gr))
    _lhs_uvv2 = _asarray(_lhs_uvv2)
    _lhs_diff_uvv2 = _asarray(_lhs_diff_uvv2)
    _lhs_dp1_C4_v2 = _asarray(_lhs_dp1_C4_v2)
    _lhs_dp1_C14_v2 = _asarray(_lhs_dp1_C14_v2)
    # RHS TA for up2/vp2 (ADG1 path) — same w-PDF coefficient as rtp2 but uses upwp/vpwp.
    # Relocated to advance_xp2_xpyp_module.calc_xp2_xpyp_ta_rhs_jax (mirror-refactor iter 27):
    # the helper recomputes the shared sign/wp_coef internally (deterministic → byte-identical),
    # which decouples this block from the rtp2 block's _sgnx2/_wp_coef/_wp_coef_zt_x2 intermediates.
    _upwp_npv2 = _asarray(upwp, dtype=np.float64)   # also used by the TP term below
    _vpwp_npv2 = _asarray(vpwp, dtype=np.float64)
    _beta_colv2 = clubb_params[:, ibeta - 1]
    _rhs_tav2_up2 = calc_xp2_xpyp_ta_rhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, wp2, wp2_zt,
        _beta_colv2, upwp, upwp, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    _rhs_tav2_vp2 = calc_xp2_xpyp_ta_rhs_jax(
        flags.l_upwind_xpyp_ta, wp3_on_wp2, wp3_on_wp2_zt, sigma_sqd_w, wp2, wp2_zt,
        _beta_colv2, vpwp, vpwp, rho_ds_zm, invrs_rho_ds_zm, rho_ds_zt, gr)
    # Auxiliary inputs for pressure and production terms
    _C_uu_shrv2 = _asarray(clubb_params[:, iC_uu_shr - 1], dtype=np.float64)[:, np.newaxis]
    _C_uu_buoyv2 = _asarray(clubb_params[:, iC_uu_buoy - 1], dtype=np.float64)[:, np.newaxis]
    _um_npv2 = _asarray(um, dtype=np.float64)
    _vm_npv2 = _asarray(vm, dtype=np.float64)
    _wp2_npv2 = _asarray(wp2, dtype=np.float64)
    _wpthvp_npv2 = _asarray(wpthvp, dtype=np.float64)
    _thv_ds_zm_npv2 = _asarray(thv_ds_zm, dtype=np.float64)
    _lhs_splat_v2 = _asarray(lhs_splat_wp2, dtype=np.float64)
    _omgv2 = 1.0 - _gamma
    # Wind gradients at interior zm levels: d(um)/dz = (um[k] - um[k-1]) * invrs_dzm[k]
    _du_dz_v2 = _invrs_dzm_np[:, 1:-1] * (_um_npv2[:, 1:] - _um_npv2[:, :-1])
    _dv_dz_v2 = _invrs_dzm_np[:, 1:-1] * (_vm_npv2[:, 1:] - _vm_npv2[:, :-1])
    # PR2 (same for up2 and vp2) via the module routine (advance_xp2_xpyp_module.F90:term_pr2)
    _pr2_v2 = _asarray(term_pr2(
        _C_uu_shrv2, _C_uu_buoyv2, _thv_ds_zm_npv2, _wpthvp_npv2,
        _upwp_npv2, _vpwp_npv2, _um_npv2, _vm_npv2, gr))
    # up2/vp2 explicit RHS (pressure-rotation form) via the module routine; symmetric (vp2 swaps the
    # u-quantities for v-quantities; the C4/C14 isotropization term couples to the *other* variance).
    _coriolisv2 = bool(getattr(flags, 'l_ho_nontrad_coriolis', False))
    _fcyv2 = _asarray(fcor_y, dtype=np.float64)[:, np.newaxis] if _coriolisv2 else None
    _rhsv2_up2 = xp2_xpyp_uv_rhs(
        _rhs_tav2_up2, _up2_prev2, _vp2_prev2, _upwp_npv2, _du_dz_v2,
        _lhs_splat_v2, _wp2_npv2, _lhs_ta_x2, _C_uu_shrv2, _C4_v2, _C14_v2,
        _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, _lhs_dp1_C4_v2, _lhs_dp1_C14_v2,
        _pr2_v2, _omgv2, _dt_adv, float(w_tol_sqd), _coriolisv2, _fcyv2)
    _rhsv2_vp2 = xp2_xpyp_uv_rhs(
        _rhs_tav2_vp2, _vp2_prev2, _up2_prev2, _vpwp_npv2, _dv_dz_v2,
        _lhs_splat_v2, _wp2_npv2, _lhs_ta_x2, _C_uu_shrv2, _C4_v2, _C14_v2,
        _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, _lhs_dp1_C4_v2, _lhs_dp1_C14_v2,
        _pr2_v2, _omgv2, _dt_adv, float(w_tol_sqd), _coriolisv2, _fcyv2)
    # Solve both with same LHS (ADG1)
    # up2/vp2 share the assembled LHS — bare tridiag solve (Fortran xp2_xpyp_solve, F90:2726)
    _lhs_uvv2_jax = jnp.asarray(_lhs_uvv2)
    _solnv2_up2 = xp2_xpyp_solve(_lhs_uvv2_jax, _rhsv2_up2)
    _solnv2_vp2 = xp2_xpyp_solve(_lhs_uvv2_jax, _rhsv2_vp2)
    if flags.l_lmm_stepping:
        _up2_jaxv2 = 0.5 * (_up2_prev2 + _solnv2_up2)
        _vp2_jaxv2 = 0.5 * (_vp2_prev2 + _solnv2_vp2)
    else:
        _up2_jaxv2 = _solnv2_up2
        _vp2_jaxv2 = _solnv2_vp2
    # Post-solve up2/vp2: pos_definite_variances (fill_holes) then clip_variance, per field —
    # the two distinct Fortran calls (advance_xp2_xpyp_module.F90:pos_definite_variances +
    # clip_explicit.F90:clip_variance). _fh feeds the PD budget term (fh-soln), _cv is the final field.
    _up2_jaxv2_fh = pos_definite_variances(
        _up2_jaxv2, rho_ds_zm, gr.dzm, float(w_tol_sqd),
        _hf_lower, _hf_upper, flags.fill_holes_type)
    _up2_jaxv2_cv = _clip_variance(_up2_jaxv2_fh, float(w_tol_sqd))
    _vp2_jaxv2_fh = pos_definite_variances(
        _vp2_jaxv2, rho_ds_zm, gr.dzm, float(w_tol_sqd),
        _hf_lower, _hf_upper, flags.fill_holes_type)
    _vp2_jaxv2_cv = _clip_variance(_vp2_jaxv2_fh, float(w_tol_sqd))

    # ============================================================ #
    # Budget stats: up2/vp2 + pressure rotation (upwp_pr4/vpwp_pr4) #
    # ============================================================ #
    if l_sample and stats_writer is not None:
        _dtv2 = float(dt_advance)
        _gv2 = _gamma   # 1.5
        _omgv2b = 1.0 - _gv2  # -0.5

        # up2/vp2 mixed values (pre-solve and post-solve)
        _up2_mixv2 = _omgv2b * _up2_prev2 + _gv2 * _solnv2_up2
        _vp2_mixv2 = _omgv2b * _vp2_prev2 + _gv2 * _solnv2_vp2

        # TA terms: rhs_ta - lhs_ta @ mixed (shared LHS with rtp2)
        _up2_tav2 = _rhs_tav2_up2 - apply_lhs_band3_interior_jax(_lhs_ta_x2, _up2_mixv2)
        _vp2_tav2 = _rhs_tav2_vp2 - apply_lhs_band3_interior_jax(_lhs_ta_x2, _vp2_mixv2)
        stats_writer.update("up2_ta", _up2_tav2)
        stats_writer.update("vp2_ta", _vp2_tav2)

        # TP: (1-C_uu_shr) * (-2*upwp*invrs_dzm*d(um)) — the inner is the variance term_tp_rhs
        # (xam=xbm=um, wpxap=wpxbp=upwp), scaled by the shear partition (1-C_uu_shr)
        _up2_tpinv2 = term_tp_rhs(_um_npv2, _um_npv2, _upwp_npv2, _upwp_npv2, _invrs_dzm_np)
        _vp2_tpinv2 = term_tp_rhs(_vm_npv2, _vm_npv2, _vpwp_npv2, _vpwp_npv2, _invrs_dzm_np)
        _up2_tpv2 = _iset(_xp.zeros_like(_up2_prev2), np.s_[:, 1:-1],
            (1.0 - _C_uu_shrv2[:, 0:1]) * _up2_tpinv2)
        _vp2_tpv2 = _iset(_xp.zeros_like(_vp2_prev2), np.s_[:, 1:-1],
            (1.0 - _C_uu_shrv2[:, 0:1]) * _vp2_tpinv2)
        stats_writer.update("up2_tp", _up2_tpv2)
        stats_writer.update("vp2_tp", _vp2_tpv2)

        # PR2: max((2/3)*(C_uu_buoy*g/thv*wpthvp + C_uu_shr*(-upwp*du_dz - vpwp*dv_dz)), 0)
        _pr2_fullv2 = _xp.zeros_like(_up2_prev2)
        _pr2_fullv2 = _iset(_pr2_fullv2, np.s_[:, 1:-1], _pr2_v2)
        stats_writer.update("up2_pr2", _pr2_fullv2)
        stats_writer.update("vp2_pr2", _pr2_fullv2)

        # PR1 stats: rhs_pr1_C4 - lhs_dp1_C4 * mixed
        # stats_pr1 (C4 only) = (1/3)*C4*(vp2_old+wp2)*invrs_tau_C4
        # stats_pr2_C14 (C14 only) = -(1/3)*C14*(vp2_old+wp2)*invrs_tau_C14 + C14*invrs_tau_C14*w_tol_sqd
        # PR1 C4/C14 decomposition (advance_xp2_xpyp_module.F90:3346-3352): exactly as the Fortran does, the
        # C4 (pressure) and C14 (dissipation) parts are obtained by calling term_pr1 twice with the other
        # coefficient zeroed (C14_zeros → C4 part; C4_zeros → C14 part).  xbp2 is the *other* variance.
        _C4_zeros_v2 = _xp.zeros_like(_C4_v2[:, 0:1])
        _C14_zeros_v2 = _xp.zeros_like(_C14_v2[:, 0:1])
        _c4u_intv2  = term_pr1(_C4_v2[:, 0:1], _C14_zeros_v2, _vp2_prev2, _wp2_npv2,
                               _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, float(w_tol_sqd))
        _c14u_intv2 = term_pr1(_C4_zeros_v2, _C14_v2[:, 0:1], _vp2_prev2, _wp2_npv2,
                               _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, float(w_tol_sqd))
        _c4v_intv2  = term_pr1(_C4_v2[:, 0:1], _C14_zeros_v2, _up2_prev2, _wp2_npv2,
                               _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, float(w_tol_sqd))
        _c14v_intv2 = term_pr1(_C4_zeros_v2, _C14_v2[:, 0:1], _up2_prev2, _wp2_npv2,
                               _invrs_tau_C4_zm_v2, _invrs_tau_C14_zm_v2, float(w_tol_sqd))
        _rhs_pr1_C4_up  = _iset(_xp.zeros_like(_up2_prev2), np.s_[:, 1:-1], _c4u_intv2)
        _rhs_pr1_C14_up = _iset(_xp.zeros_like(_up2_prev2), np.s_[:, 1:-1], _c14u_intv2)
        _rhs_pr1_C4_vp  = _iset(_xp.zeros_like(_vp2_prev2), np.s_[:, 1:-1], _c4v_intv2)
        _rhs_pr1_C14_vp = _iset(_xp.zeros_like(_vp2_prev2), np.s_[:, 1:-1], _c14v_intv2)
        # _lhs_dp1_C4_v2 = (2/3)*C4*invrs_tau_C4 (already unscaled, no gamma factor)
        # _lhs_dp1_C14_v2 = (1/3)*C14*invrs_tau_C14 (already unscaled)
        _up2_pr1_v2 = _rhs_pr1_C4_up - _lhs_dp1_C4_v2 * _up2_mixv2
        _up2_dp1_v2 = _rhs_pr1_C14_up - _lhs_dp1_C14_v2 * _up2_mixv2
        _vp2_pr1_v2 = _rhs_pr1_C4_vp - _lhs_dp1_C4_v2 * _vp2_mixv2
        _vp2_dp1_v2 = _rhs_pr1_C14_vp - _lhs_dp1_C14_v2 * _vp2_mixv2
        stats_writer.update("up2_pr1", _up2_pr1_v2)
        stats_writer.update("up2_dp1", _up2_dp1_v2)
        stats_writer.update("vp2_pr1", _vp2_pr1_v2)
        stats_writer.update("vp2_dp1", _vp2_dp1_v2)

        # DP2 (diffusion): -lhs_diff_uvv2 @ new
        stats_writer.update("up2_dp2", -apply_lhs_band3_interior_jax(_lhs_diff_uvv2, _solnv2_up2))
        stats_writer.update("vp2_dp2", -apply_lhs_band3_interior_jax(_lhs_diff_uvv2, _solnv2_vp2))

        # PD (fill_holes effect): (fh - solve) / dt
        _up2_pdv2 = _xp.zeros_like(_up2_prev2)
        _vp2_pdv2 = _xp.zeros_like(_vp2_prev2)
        _up2_pdv2 = _iset(_up2_pdv2, np.s_[:, 1:-1], (
            (_up2_jaxv2_fh[:, 1:-1] - _solnv2_up2[:, 1:-1]) / _dtv2))
        _vp2_pdv2 = _iset(_vp2_pdv2, np.s_[:, 1:-1], (
            (_vp2_jaxv2_fh[:, 1:-1] - _solnv2_vp2[:, 1:-1]) / _dtv2))
        stats_writer.update("up2_pd", _up2_pdv2)
        stats_writer.update("vp2_pd", _vp2_pdv2)

        # ZT: post-clip value interpolated to zt
        stats_writer.update("up2_zt",
            _asarray(jnp.maximum(
                zm2zt_jax(jnp.asarray(_up2_jaxv2_cv), gr),
                float(w_tol_sqd)), dtype=np.float64))
        stats_writer.update("vp2_zt",
            _asarray(jnp.maximum(
                zm2zt_jax(jnp.asarray(_vp2_jaxv2_cv), gr),
                float(w_tol_sqd)), dtype=np.float64))

        # upwp_pr4/vpwp_pr4 are written in the advance_xm_wpxp budget stats block
        # using ddzt_um/ddzt_vm (zt-level gradient), not du_dz_zm here.

    # ============================================================ #
    # Override advance_xp2_xpyp state with JAX values             #
    # rtp2/thlp2/rtpthlp computed in JAX.                           #
    # up2/vp2 computed in JAX.                                      #
    # ============================================================ #
    rtp2    = _asarray(_rtp2_jaxx2_cv,      dtype=np.float64).copy()
    thlp2   = _asarray(_thlp2_jaxx2_cv,     dtype=np.float64).copy()
    rtpthlp = _asarray(_rtpthlp_jaxx2_clip, dtype=np.float64).copy()
    up2     = _asarray(_up2_jaxv2_cv,        dtype=np.float64).copy()
    vp2     = _asarray(_vp2_jaxv2_cv,        dtype=np.float64).copy()
    return rtp2, thlp2, rtpthlp, up2, vp2

__all__ = [
    "term_dp1_lhs",
    "xp2_xpyp_lhs",
    "term_dp1_rhs",
    "xp2_xpyp_rhs",
    "update_xp2_mc",
    "calc_xp2_xpyp_ta_lhs_jax",
    "calc_xp2_xpyp_ta_rhs_jax",
    "xp2_xpyp_uv_rhs",
    "xp2_xpyp_solve",
    "solve_xp2_xpyp_jax",
    "solve_xp2_xpyp_with_single_lhs",
    "advance_xp2_xpyp",
]
