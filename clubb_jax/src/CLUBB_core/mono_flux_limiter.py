"""Monotonic turbulent flux limiter — port of mono_flux_limiter.F90.

Limits w'x' and re-solves xm when the turbulent advection of xm would create new
extrema (a monotonic turbulent-advection scheme). Active for um/vm/rtm/thlm when the
corresponding l_mono_flux_lim_* flag is set (all default .true.). It is a no-op when
w'x' stays within the monotonic bounds — which is the case for weak-shear BLs (ARM,
bomex, dycoms2_rf01, …) but NOT for strong-shear / stable BLs (atex, gabls3_night).

This port targets the standalone configuration used by the SCM cases:
  - ascending grid (grid_dir = +1, grid_dir_indx = +1, grid_dir_str k_zt = k-1)
  - l_implemented = .false.  (mean advection IS included in the re-solve LHS)
  - l_mfl_xm_imp_adj = .true. (implicit tridiagonal re-solve of xm)
  - l_force_descending_solves = .false. (no band/level flip before the solve)
Descending grid / host-model (l_implemented) paths are not ported (not used here).

Reference: clubb_release/src/CLUBB_core/mono_flux_limiter.F90
  monotonic_turbulent_flux_limit / mfl_xm_lhs / mfl_xm_rhs / mfl_xm_solve
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zt2zm_jax
from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import term_ma_zt_lhs_jax
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_lu_solve_jax

# solve_type tags
MFL_RTM = "rtm"
MFL_THLM = "thlm"
MFL_UM = "um"
MFL_VM = "vm"

_MAX_XP2 = {MFL_RTM: 5.0e-6, MFL_THLM: 5.0, MFL_UM: 10.0, MFL_VM: 10.0}
_EPS = np.finfo(np.float64).eps
_UNUSED = -999.9
_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)


def _mean_w_up_down(w_i, varnce_w_i, wm):
    """mean_vert_vel_up_down for one PDF component (zm grid).

    Port of mono_flux_limiter.F90:mean_vert_vel_up_down. Returns (mwd, mwu):
    the mean downward and upward vertical velocity of the component, with the
    domain boundaries zeroed. Matches the stats-block `_mwc_mfl`.
    """
    import jax
    wi = jnp.asarray(w_i)
    sig = jnp.sqrt(jnp.maximum(jnp.asarray(varnce_w_i), 0.0))
    sig_s = jnp.where(sig > 0.0, sig, 1.0)
    z = (0.0 - wi) / (_SQRT2 * sig_s)
    ev = jnp.exp(-z ** 2)
    ef = jax.scipy.special.erf(z)
    too_weak = jnp.abs(wi) + 3.0 * sig <= wm
    all_dn = (~too_weak) & (wi + 3.0 * sig <= 0.0)
    all_up = (~too_weak) & (~all_dn) & (wi - 3.0 * sig >= 0.0)
    mwd_m = -sig / _SQRT2PI * ev + wi * 0.5 * (1.0 + ef)
    mwu_m = sig / _SQRT2PI * ev + wi * 0.5 * (1.0 - ef)
    mwd = jnp.where(too_weak, 0.0, jnp.where(all_dn, wi, jnp.where(all_up, 0.0, mwd_m)))
    mwu = jnp.where(too_weak, 0.0, jnp.where(all_dn, 0.0, jnp.where(all_up, wi, mwu_m)))
    mwd = mwd.at[:, 0].set(0.0).at[:, -1].set(0.0)
    mwu = mwu.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return np.asarray(mwd, dtype=np.float64), np.asarray(mwu, dtype=np.float64)


def calc_turb_adv_range(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                        mixt_frac_zm, gr, dt):
    """Range of zt levels reachable by turbulent advection in one timestep.

    Port of mono_flux_limiter.F90:calc_turb_adv_range (l_constant_thickness=.false.,
    ascending grid). Returns (low_lev_effect, high_lev_effect), each (ngrdcol, nzt)
    int (0-based zt indices). Mirrors the stats-block `_ll_mfl`/`_hl_mfl` computation.
    """
    ngrdcol, nzm = w_1_zm.shape
    nzt = nzm - 1
    gd = float(gr.grid_dir)
    dzm = np.asarray(gr.dzm, dtype=np.float64)
    wm = gd * dzm * (1.0 / dt)   # w_min(k) = grid_dir * dzm * invrs_dt
    mwd1, mwu1 = _mean_w_up_down(w_1_zm, varnce_w_1_zm, wm)
    mwd2, mwu2 = _mean_w_up_down(w_2_zm, varnce_w_2_zm, wm)
    mf = np.asarray(mixt_frac_zm, dtype=np.float64)
    vvd = mf * mwd1 + (1.0 - mf) * mwd2   # mean_w_down (zm)
    vvu = mf * mwu1 + (1.0 - mf) * mwu2   # mean_w_up   (zm)

    lle = np.zeros((ngrdcol, nzt), dtype=np.int64)
    hle = np.zeros((ngrdcol, nzt), dtype=np.int64)
    for i in range(ngrdcol):
        lle[i, 0] = 0;        hle[i, 0] = 0
        lle[i, nzt - 2] = nzt - 2;  hle[i, nzt - 2] = nzt - 1
        lle[i, nzt - 1] = nzt - 1;  hle[i, nzt - 1] = nzt - 1
        for k in range(1, nzt - 2):
            da = 0.0
            for j in range(k - 1, -1, -1):       # downward: low_lev_effect
                lle[i, k] = j
                ja = j + 1                        # zm index (ascending)
                vu = vvu[i, ja]
                if vu > 0.0:
                    da += gd * dzm[i, ja] / vu
                    if da >= dt:
                        break
                else:
                    lle[i, k] = min(j + 1, nzt - 1)
                    break
            da = 0.0
            for j in range(k + 1, nzt):          # upward: high_lev_effect
                hle[i, k] = j
                ja = j                            # zm index (ascending)
                vd = vvd[i, ja]
                if vd < 0.0:
                    da += -gd * dzm[i, ja] / vd
                    if da >= dt:
                        break
                else:
                    hle[i, k] = max(j - 1, 0)
                    break
    return lle, hle


def monotonic_turbulent_flux_limit(
    solve_type, xm, wpxp, xm_old, xp2, wm_zt, xm_forcing,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
    xp2_threshold, xm_tol, low_lev_effect, high_lev_effect,
    gr, dt, l_mono_flux_lim_spikefix=True,
):
    """Apply the monotonic turbulent flux limiter to (xm, wpxp).

    All arrays are NumPy float64. xm/xm_old/wm_zt/xm_forcing/rho_ds_zt/invrs_rho_ds_zt
    are (ngrdcol, nzt); wpxp/xp2/rho_ds_zm/invrs_rho_ds_zm are (ngrdcol, nzm).
    low_lev_effect/high_lev_effect are (ngrdcol, nzt) int (0-based zt indices).

    Returns (xm_new, wpxp_new) as NumPy float64.
    """
    ngrdcol, nzt = xm.shape
    nzm = nzt + 1
    is_uv = solve_type in (MFL_UM, MFL_VM)
    max_xp2 = _MAX_XP2[solve_type]
    invrs_dt = 1.0 / dt
    grid_dir = float(gr.grid_dir)           # +1 ascending
    dzt = np.asarray(gr.dzt, dtype=np.float64)      # (ngrdcol, nzt)
    dzm = np.asarray(gr.dzm, dtype=np.float64)      # (ngrdcol, nzm)
    invrs_dzt = np.asarray(gr.invrs_dzt, dtype=np.float64)  # (ngrdcol, nzt)

    xm = xm.copy()
    wpxp = wpxp.copy()

    # Interpolate x'^2 to zt, clip to [xp2_threshold, max_xp2].
    xp2_zt = np.asarray(zm2zt_jax(jnp.asarray(xp2), gr), dtype=np.float64)
    xp2_zt = np.minimum(np.maximum(xp2_zt, xp2_threshold), max_xp2)

    xm_enter_mfl = xm.copy()

    # max_dev and xm_without_ta (m_adv_term = 0); min/max allowable per level.
    max_dev = np.maximum(2.0 * np.sqrt(xp2_zt), xm_tol)
    xm_without_ta = xm_old + dt * xm_forcing      # (ngrdcol, nzt)
    if is_uv:
        min_x_allowable_lev = xm_without_ta - max_dev
    else:
        min_x_allowable_lev = np.maximum(xm_without_ta - max_dev, 0.0)
    max_x_allowable_lev = xm_without_ta + max_dev

    # min/max allowable over the turbulent-advection range [low_lev, high_lev].
    # Ascending grid: clamp to [k_lb_zt=0, k_ub_zt=nzt-1].
    min_x_allowable = np.empty((ngrdcol, nzt))
    max_x_allowable = np.empty((ngrdcol, nzt))
    for i in range(ngrdcol):
        for k in range(nzt):
            lo = max(int(low_lev_effect[i, k]), 0)
            hi = min(int(high_lev_effect[i, k]), nzt - 1)
            min_x_allowable[i, k] = np.min(min_x_allowable_lev[i, lo:hi + 1])
            max_x_allowable[i, k] = np.max(max_x_allowable_lev[i, lo:hi + 1])

    wpxp_thresh_term_zt = (invrs_dt * grid_dir * dzt
                           * (xm_without_ta - min_x_allowable))
    wpxp_mfl_max_term_zt = rho_ds_zt * wpxp_thresh_term_zt
    wpxp_mfl_min_term_zt = (rho_ds_zt * invrs_dt * grid_dir * dzt
                            * (xm_without_ta - max_x_allowable))
    wpxp_thresh_term = np.asarray(
        zt2zm_jax(jnp.asarray(wpxp_thresh_term_zt), gr), dtype=np.float64)

    # Sequential clip of wpxp over interior zm levels (k = 1 .. nzm-2; k_zt = k-1).
    wpxp_net_adjust = np.zeros((ngrdcol, nzm))
    l_xm_adjustment_needed = np.zeros(ngrdcol, dtype=bool)
    spikefix_rtm = l_mono_flux_lim_spikefix and solve_type == MFL_RTM
    for i in range(ngrdcol):
        for k in range(1, nzm - 1):
            k_zt = k - 1
            if (spikefix_rtm
                    and abs(wpxp[i, k - 1]) > wpxp_thresh_term[i, k - 1]
                    and wpxp[i, k - 1] < 0.0):
                wpxp_mfl_max = 0.0
            else:
                wpxp_mfl_max = invrs_rho_ds_zm[i, k] * (
                    wpxp_mfl_max_term_zt[i, k_zt]
                    + rho_ds_zm[i, k - 1] * wpxp[i, k - 1])
            if wpxp[i, k] > wpxp_mfl_max:
                wpxp_net_adjust[i, k] = wpxp_mfl_max - wpxp[i, k]
                wpxp[i, k] = wpxp_mfl_max
                if abs(wpxp_net_adjust[i, k]) > _EPS:
                    l_xm_adjustment_needed[i] = True
            else:
                wpxp_mfl_min = invrs_rho_ds_zm[i, k] * (
                    wpxp_mfl_min_term_zt[i, k_zt]
                    + rho_ds_zm[i, k - 1] * wpxp[i, k - 1])
                if wpxp[i, k] < wpxp_mfl_min:
                    wpxp_net_adjust[i, k] = wpxp_mfl_min - wpxp[i, k]
                    wpxp[i, k] = wpxp_mfl_min
                    if abs(wpxp_net_adjust[i, k]) > _EPS:
                        l_xm_adjustment_needed[i] = True

    if not l_xm_adjustment_needed.any():
        return xm, wpxp

    # Re-solve xm implicitly with the limited wpxp (mfl_xm_lhs/rhs/solve).
    # LHS: mean advection (l_implemented=False) + 1/dt on the main diagonal.
    lhs = np.array(term_ma_zt_lhs_jax(jnp.asarray(wm_zt), gr), dtype=np.float64)
    lhs[1, :, :] = lhs[1, :, :] + invrs_dt
    # RHS: xm_old/dt - d(rho_ds_zm wpxp)/dz / rho_ds_zt + xm_forcing.
    rhs = xm_old * invrs_dt + xm_forcing
    rhs = rhs - invrs_rho_ds_zt * invrs_dzt * (
        rho_ds_zm[:, 1:] * wpxp[:, 1:] - rho_ds_zm[:, :-1] * wpxp[:, :-1])
    xm_mfl = np.asarray(
        tridiag_lu_solve_jax(jnp.asarray(lhs), jnp.asarray(rhs)), dtype=np.float64)
    for i in range(ngrdcol):
        if l_xm_adjustment_needed[i]:
            xm[i, :] = xm_mfl[i, :]

    # Spike fix at the domain top: conserve column xm if the top level moved a lot.
    k_ub_zt = nzt - 1
    k_ub_zm = nzm - 1
    for i in range(ngrdcol):
        if abs(xm[i, k_ub_zt] - xm_enter_mfl[i, k_ub_zt]) > 10.0 * xm_tol:
            dz = np.asarray(gr.zm)[i, k_ub_zm] - np.asarray(gr.zm)[i, k_ub_zm - 1]
            xm_density_weighted = (rho_ds_zt[i, k_ub_zt]
                                   * (xm[i, k_ub_zt] - xm_enter_mfl[i, k_ub_zt]) * dz)
            xm_vert_integral = np.sum(
                rho_ds_zt[i, 0:k_ub_zt] * xm[i, 0:k_ub_zt]
                * grid_dir * dzt[i, 0:k_ub_zt])
            if abs(xm_vert_integral) < _EPS:
                xm[i, k_ub_zt] = xm_enter_mfl[i, k_ub_zt]
            else:
                xm_adj_coef = xm_density_weighted / xm_vert_integral
                if xm_adj_coef < -0.99:
                    xm_adj_coef = -0.99
                xm[i, :] = xm[i, :] * (1.0 + xm_adj_coef)
                xm[i, k_ub_zt] = xm_enter_mfl[i, k_ub_zt]

    return xm, wpxp


def monotonic_turbulent_flux_limit_jax(
    solve_type, xm, wpxp, xm_old, xp2, wm_zt, xm_forcing,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
    xp2_threshold, xm_tol, low_lev_effect, high_lev_effect,
    gr, dt, l_mono_flux_lim_spikefix=True,
):
    """JAX (lax.scan) port of `monotonic_turbulent_flux_limit` — bit-exact to the NumPy version and
    differentiable w.r.t. the fields (REFACTOR B2, iter11). `low_lev_effect`/`high_lev_effect` are the
    integer turbulent-advection ranges from `calc_turb_adv_range`; they derive from the w-PDF (not the
    limited fields), so they are structural CONSTANTS for the limiter's gradient. The sequential per-level
    clip (level k reads the possibly-adjusted wpxp[k-1]) is a `lax.scan` with the adjusted wpxp as carry."""
    xm = jnp.asarray(xm, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    xm_old = jnp.asarray(xm_old, dtype=jnp.float64)
    xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    wm_zt = jnp.asarray(wm_zt, dtype=jnp.float64)
    xm_forcing = jnp.asarray(xm_forcing, dtype=jnp.float64)
    rho_ds_zm = jnp.asarray(rho_ds_zm, dtype=jnp.float64)
    rho_ds_zt = jnp.asarray(rho_ds_zt, dtype=jnp.float64)
    invrs_rho_ds_zm = jnp.asarray(invrs_rho_ds_zm, dtype=jnp.float64)
    invrs_rho_ds_zt = jnp.asarray(invrs_rho_ds_zt, dtype=jnp.float64)
    lle = jnp.asarray(low_lev_effect, dtype=jnp.int64)
    hle = jnp.asarray(high_lev_effect, dtype=jnp.int64)

    ngrdcol, nzt = xm.shape
    nzm = nzt + 1
    is_uv = solve_type in (MFL_UM, MFL_VM)
    max_xp2 = _MAX_XP2[solve_type]
    invrs_dt = 1.0 / dt
    grid_dir = float(gr.grid_dir)
    dzt = jnp.asarray(gr.dzt, dtype=jnp.float64)
    invrs_dzt = jnp.asarray(gr.invrs_dzt, dtype=jnp.float64)
    spikefix_rtm = bool(l_mono_flux_lim_spikefix and solve_type == MFL_RTM)

    xm_enter = xm

    # x'^2 -> zt, clip; per-level min/max allowable.
    xp2_zt = jnp.minimum(jnp.maximum(zm2zt_jax(xp2, gr), xp2_threshold), max_xp2)
    max_dev = jnp.maximum(2.0 * jnp.sqrt(xp2_zt), xm_tol)
    xm_without_ta = xm_old + dt * xm_forcing
    min_lev = (xm_without_ta - max_dev) if is_uv else jnp.maximum(xm_without_ta - max_dev, 0.0)
    max_lev = xm_without_ta + max_dev

    # min/max allowable over the turbulent-advection window [lle, hle] (clamped to [0, nzt-1]).
    lo = jnp.maximum(lle, 0)
    hi = jnp.minimum(hle, nzt - 1)
    jj = jnp.arange(nzt)
    win = (jj[None, None, :] >= lo[:, :, None]) & (jj[None, None, :] <= hi[:, :, None])  # (ngrdcol,nzt,nzt)
    min_x_allowable = jnp.min(jnp.where(win, min_lev[:, None, :], jnp.inf), axis=2)
    max_x_allowable = jnp.max(jnp.where(win, max_lev[:, None, :], -jnp.inf), axis=2)

    wpxp_thresh_term_zt = invrs_dt * grid_dir * dzt * (xm_without_ta - min_x_allowable)
    wpxp_mfl_max_term_zt = rho_ds_zt * wpxp_thresh_term_zt
    wpxp_mfl_min_term_zt = rho_ds_zt * invrs_dt * grid_dir * dzt * (xm_without_ta - max_x_allowable)
    wpxp_thresh_term = zt2zm_jax(wpxp_thresh_term_zt, gr)   # (ngrdcol, nzm)

    # Sequential clip of wpxp over interior zm levels k = 1..nzm-2 (carry = adjusted wpxp[k-1]).
    ks = jnp.arange(1, nzm - 1)
    inp = (wpxp[:, ks].T, wpxp_mfl_max_term_zt[:, ks - 1].T, wpxp_mfl_min_term_zt[:, ks - 1].T,
           wpxp_thresh_term[:, ks - 1].T, rho_ds_zm[:, ks - 1].T, invrs_rho_ds_zm[:, ks].T)

    def clip_step(carry, level):
        wk, mxt, mnt, thr, rkm1, irk = level
        spikefix_cond = (jnp.abs(carry) > thr) & (carry < 0.0)
        mfl_max_raw = irk * (mxt + rkm1 * carry)
        mfl_max = jnp.where(spikefix_cond, 0.0, mfl_max_raw) if spikefix_rtm else mfl_max_raw
        over_max = wk > mfl_max
        mfl_min = irk * (mnt + rkm1 * carry)
        under_min = (~over_max) & (wk < mfl_min)
        wk_new = jnp.where(over_max, mfl_max, jnp.where(under_min, mfl_min, wk))
        net = jnp.where(over_max, mfl_max - wk, jnp.where(under_min, mfl_min - wk, 0.0))
        adj = (over_max | under_min) & (jnp.abs(net) > _EPS)
        return wk_new, (wk_new, adj)

    _, (wk_new_T, adj_T) = jax.lax.scan(clip_step, wpxp[:, 0], inp)
    wpxp_new = jnp.concatenate([wpxp[:, :1], wk_new_T.T, wpxp[:, -1:]], axis=1)
    adj_needed = jnp.any(adj_T, axis=0)                    # (ngrdcol,)

    # Re-solve xm implicitly with the limited wpxp; apply per-column where adjustment was needed.
    lhs = term_ma_zt_lhs_jax(wm_zt, gr).at[1, :, :].add(invrs_dt)
    rhs = (xm_old * invrs_dt + xm_forcing
           - invrs_rho_ds_zt * invrs_dzt
           * (rho_ds_zm[:, 1:] * wpxp_new[:, 1:] - rho_ds_zm[:, :-1] * wpxp_new[:, :-1]))
    xm_mfl = tridiag_lu_solve_jax(lhs, rhs)
    xm = jnp.where(adj_needed[:, None], xm_mfl, xm)

    # Spike fix at the domain top: conserve column xm if the top level moved a lot.
    k_ub_zt = nzt - 1
    zm = jnp.asarray(gr.zm, dtype=jnp.float64)
    dz = zm[:, nzm - 1] - zm[:, nzm - 2]
    cond_top = jnp.abs(xm[:, k_ub_zt] - xm_enter[:, k_ub_zt]) > 10.0 * xm_tol
    xm_density_weighted = rho_ds_zt[:, k_ub_zt] * (xm[:, k_ub_zt] - xm_enter[:, k_ub_zt]) * dz
    xm_vert_integral = jnp.sum(rho_ds_zt[:, :k_ub_zt] * xm[:, :k_ub_zt] * grid_dir * dzt[:, :k_ub_zt], axis=1)
    small_int = jnp.abs(xm_vert_integral) < _EPS
    safe_int = jnp.where(jnp.abs(xm_vert_integral) > 0.0, xm_vert_integral, 1.0)
    adj_coef = jnp.maximum(xm_density_weighted / safe_int, -0.99)
    xm_scaled = (xm * (1.0 + adj_coef)[:, None]).at[:, k_ub_zt].set(xm_enter[:, k_ub_zt])
    xm_small = xm.at[:, k_ub_zt].set(xm_enter[:, k_ub_zt])
    xm = jnp.where(cond_top[:, None], jnp.where(small_int[:, None], xm_small, xm_scaled), xm)

    return xm, wpxp_new
