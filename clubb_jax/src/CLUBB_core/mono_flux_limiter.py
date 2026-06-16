"""JAX port of `src/CLUBB_core/mono_flux_limiter.F90`.

Description:
  Limits the value of w'x' and corrects the value of xm when the xm turbulent
  advection term is not monotonic.

  A monotonic turbulent advection scheme does not allow new extrema for variable
  x to be created by turbulent advection.  When w'x' would move xm outside the
  allowable range implied by nearby previous-time values, forcings, and mean
  advection, w'x' is limited and xm is re-solved.

JAX adaptation:
  `calc_turb_adv_range` keeps the Fortran routine boundary, but its ascending
  and descending grid loops are split into adjacent private helpers so the
  branch-specific `lax.while_loop` carries stay small and static.
"""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import erf

from clubb_jax.src.CLUBB_core.clubb_constants import (
    eps,
    l_force_descending_solves,
    one,
    unused_var,
    zero,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.grid_class import zm2zt, zt2zm
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zt_lhs
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_solve


# Private named constants to avoid string comparisons.
# NOTE: These values must match the values for xm_wpxp_thlm
# and xm_wpxp_rtm given in advance_xm_wpxp_module!
mono_flux_thlm = 1
mono_flux_rtm = 2
mono_flux_um = 4
mono_flux_vm = 5

ndiags3 = 3


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "solve_type",
        "tridiag_solve_method",
        "l_implemented",
        "l_upwind_xm_ma",
        "l_mono_flux_lim_spikefix",
    ),
)
def monotonic_turbulent_flux_limit(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    solve_type: int,
    dt: float,
    xm_old,
    xp2,
    wm_zt,
    xm_forcing,
    rho_ds_zm,
    rho_ds_zt,
    invrs_rho_ds_zm,
    invrs_rho_ds_zt,
    xp2_threshold: float,
    xm_tol: float,
    l_implemented: bool,
    low_lev_effect,
    high_lev_effect,
    tridiag_solve_method: int,
    l_upwind_xm_ma: bool,
    l_mono_flux_lim_spikefix: bool,
    stats,
    xm,
    wpxp,
    err_info,
):
    """Limit w'x' and correct xm when turbulent advection is not monotonic."""
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
    low_lev_effect = jnp.asarray(low_lev_effect, dtype=jnp.int32)
    high_lev_effect = jnp.asarray(high_lev_effect, dtype=jnp.int32)

    name_wpxp_mfl = ""
    name_xm_mfl = ""
    name_xm_enter_mfl = ""
    name_xm_old = ""
    name_wpxp_enter_mfl = ""
    name_xm_without_ta = ""
    name_xm_mfl_min = ""
    name_xm_mfl_max = ""
    name_wpxp_mfl_min = ""
    name_wpxp_mfl_max = ""
    name_xm_exit_mfl = ""
    name_wpxp_exit_mfl = ""

    if solve_type == mono_flux_rtm:
        max_xp2 = 5.0e-6
        name_wpxp_mfl = "wprtp_mfl"
        name_xm_mfl = "rtm_mfl"
        name_xm_enter_mfl = "rtm_enter_mfl"
        name_xm_old = "rtm_old"
        name_wpxp_enter_mfl = "wprtp_enter_mfl"
        name_xm_without_ta = "rtm_without_ta"
        name_xm_mfl_min = "rtm_mfl_min"
        name_xm_mfl_max = "rtm_mfl_max"
        name_wpxp_mfl_min = "wprtp_mfl_min"
        name_wpxp_mfl_max = "wprtp_mfl_max"
        name_xm_exit_mfl = "rtm_exit_mfl"
        name_wpxp_exit_mfl = "wprtp_exit_mfl"
    elif solve_type == mono_flux_thlm:
        max_xp2 = 5.0
        name_wpxp_mfl = "wpthlp_mfl"
        name_xm_mfl = "thlm_mfl"
        name_xm_enter_mfl = "thlm_enter_mfl"
        name_xm_old = "thlm_old"
        name_wpxp_enter_mfl = "wpthlp_enter_mfl"
        name_xm_without_ta = "thlm_without_ta"
        name_xm_mfl_min = "thlm_mfl_min"
        name_xm_mfl_max = "thlm_mfl_max"
        name_wpxp_mfl_min = "wpthlp_mfl_min"
        name_wpxp_mfl_max = "wpthlp_mfl_max"
        name_xm_exit_mfl = "thlm_exit_mfl"
        name_wpxp_exit_mfl = "wpthlp_exit_mfl"
    elif solve_type == mono_flux_um:
        max_xp2 = 10.0
        name_wpxp_mfl = "upwp_mfl"
        name_xm_mfl = "um_mfl"
    elif solve_type == mono_flux_vm:
        max_xp2 = 10.0
        name_wpxp_mfl = "vpwp_mfl"
        name_xm_mfl = "vm_mfl"
    else:
        max_xp2 = 5.0

    if stats.l_sample:
        stats = stats.begin_budget(name_wpxp_mfl, wpxp / dt)
        stats = stats.begin_budget(name_xm_mfl, xm / dt)
        stats = stats.update(name_xm_enter_mfl, xm)
        stats = stats.update(name_xm_old, xm_old)
        stats = stats.update(name_wpxp_enter_mfl, wpxp)

    invrs_dt = one / dt

    # Interpolate x'^2 to thermodynamic levels.
    xp2_zt = zm2zt(nzm, nzt, ngrdcol, gr, xp2)

    # Place an upper limit on xp2_zt.
    xp2_zt = jnp.minimum(jnp.maximum(xp2_zt, xp2_threshold), max_xp2)

    # Store the value of xm as it enters the mfl.
    xm_enter_mfl = xm

    # Find the maximum and minimum usable values of variable x at each
    # vertical level.
    max_dev = jnp.maximum(2.0 * jnp.sqrt(xp2_zt), xm_tol)

    # Shut off to avoid using new, possibly corrupt mean advection term.
    xm_without_ta = xm_old + dt * xm_forcing

    # Since variable x must be one of theta_l, r_t, or a scalar, all of which
    # are positive definite quantities, the value must be >= 0.  Variable x
    # may also be u or v, in which case it is not positive definite.
    is_uv = solve_type in (mono_flux_um, mono_flux_vm)
    min_x_allowable_lev = jnp.where(
        is_uv,
        xm_without_ta - max_dev,
        jnp.maximum(xm_without_ta - max_dev, zero_threshold),
    )
    max_x_allowable_lev = xm_without_ta + max_dev

    # Find the smallest and largest value of all relevant levels for variable x.
    low_lev = jnp.minimum(low_lev_effect, high_lev_effect)
    high_lev = jnp.maximum(low_lev_effect, high_lev_effect)
    low_lev = jnp.maximum(low_lev, 0)
    high_lev = jnp.minimum(high_lev, nzt - 1)
    level_idx = jnp.arange(nzt)
    window = (
        (level_idx[None, None, :] >= low_lev[:, :, None])
        & (level_idx[None, None, :] <= high_lev[:, :, None])
    )
    min_x_allowable = jnp.min(
        jnp.where(window, min_x_allowable_lev[:, None, :], jnp.inf),
        axis=2,
    )
    max_x_allowable = jnp.max(
        jnp.where(window, max_x_allowable_lev[:, None, :], -jnp.inf),
        axis=2,
    )

    wpxp_thresh_term_zt = (
        invrs_dt * gr.grid_dir * gr.dzt * (xm_without_ta - min_x_allowable)
    )
    wpxp_mfl_max_term_zt = rho_ds_zt * wpxp_thresh_term_zt
    wpxp_mfl_min_term_zt = (
        rho_ds_zt * invrs_dt * gr.grid_dir * gr.dzt * (xm_without_ta - max_x_allowable)
    )

    # Interpolate wpxp_thresh_term_zt to momentum levels.
    wpxp_thresh_term = zt2zm(nzm, nzt, ngrdcol, gr, wpxp_thresh_term_zt)

    if gr.grid_dir_indx > 0:
        scan_levels = jnp.arange(1, nzm - 1)
        initial_neighbor = wpxp[:, gr.k_lb_zm]
    else:
        scan_levels = jnp.arange(nzm - 2, 0, -1)
        initial_neighbor = wpxp[:, gr.k_lb_zm]

    spikefix_rtm = bool(l_mono_flux_lim_spikefix and solve_type == mono_flux_rtm)

    def clip_step(carry, k):
        previous_wpxp = carry
        k_zt = jnp.where(gr.grid_dir_indx > 0, k - 1, k)
        previous_k = k - gr.grid_dir_indx

        previous_thresh = jnp.take(wpxp_thresh_term, previous_k, axis=1)
        previous_rho = jnp.take(rho_ds_zm, previous_k, axis=1)
        invrs_rho = jnp.take(invrs_rho_ds_zm, k, axis=1)
        max_term = jnp.take(wpxp_mfl_max_term_zt, k_zt, axis=1)
        min_term = jnp.take(wpxp_mfl_min_term_zt, k_zt, axis=1)
        current_wpxp = jnp.take(wpxp, k, axis=1)

        spikefix_cond = (
            (jnp.abs(previous_wpxp) > previous_thresh)
            & (previous_wpxp < zero)
        )
        wpxp_mfl_max_raw = invrs_rho * (max_term + previous_rho * previous_wpxp)
        wpxp_mfl_max_k = jnp.where(spikefix_cond & spikefix_rtm, zero, wpxp_mfl_max_raw)

        over_max = current_wpxp > wpxp_mfl_max_k
        wpxp_mfl_min_k = invrs_rho * (min_term + previous_rho * previous_wpxp)
        under_min = (~over_max) & (current_wpxp < wpxp_mfl_min_k)

        adjusted_wpxp = jnp.where(
            over_max,
            wpxp_mfl_max_k,
            jnp.where(under_min, wpxp_mfl_min_k, current_wpxp),
        )
        wpxp_net_adjust_k = jnp.where(
            over_max,
            wpxp_mfl_max_k - current_wpxp,
            jnp.where(under_min, wpxp_mfl_min_k - current_wpxp, zero),
        )
        wpxp_mfl_min_stat = jnp.where(over_max, unused_var, wpxp_mfl_min_k)
        l_xm_adjustment_needed = jnp.abs(wpxp_net_adjust_k) > eps
        return adjusted_wpxp, (
            k,
            adjusted_wpxp,
            wpxp_net_adjust_k,
            wpxp_mfl_min_stat,
            wpxp_mfl_max_k,
            l_xm_adjustment_needed,
        )

    _carry, (
        k_scan,
        wpxp_scan,
        wpxp_net_adjust_scan,
        wpxp_mfl_min_scan,
        wpxp_mfl_max_scan,
        l_xm_adjustment_scan,
    ) = jax.lax.scan(clip_step, initial_neighbor, scan_levels)

    wpxp = wpxp.at[:, k_scan].set(wpxp_scan.T)
    wpxp_net_adjust = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    wpxp_mfl_min = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    wpxp_mfl_max = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    wpxp_net_adjust = wpxp_net_adjust.at[:, k_scan].set(wpxp_net_adjust_scan.T)
    wpxp_mfl_min = wpxp_mfl_min.at[:, k_scan].set(wpxp_mfl_min_scan.T)
    wpxp_mfl_max = wpxp_mfl_max.at[:, k_scan].set(wpxp_mfl_max_scan.T)
    l_xm_adjustment_needed = jnp.any(l_xm_adjustment_scan, axis=0)
    l_any_adjustment_needed = jnp.any(l_xm_adjustment_needed)

    def adjust_xm(args):
        xm_in, err_info_in = args
        lhs_mfl_xm = mfl_xm_lhs(
            nzm,
            nzt,
            ngrdcol,
            dt,
            gr.weights_zt2zm,
            gr.invrs_dzt,
            gr.invrs_dzm,
            wm_zt,
            l_implemented,
            l_upwind_xm_ma,
            gr.grid_dir,
        )
        rhs_mfl_xm = mfl_xm_rhs(
            nzm,
            nzt,
            ngrdcol,
            dt,
            xm_old,
            wpxp,
            xm_forcing,
            gr.invrs_dzt,
            rho_ds_zm,
            invrs_rho_ds_zt,
        )
        xm_mfl, err_info_out = mfl_xm_solve(
            nzt,
            ngrdcol,
            gr,
            solve_type,
            tridiag_solve_method,
            l_implemented,
            lhs_mfl_xm,
            rhs_mfl_xm,
            err_info_in,
        )
        xm_out = jnp.where(l_xm_adjustment_needed[:, None], xm_mfl, xm_in)

        # Ensure there are no spikes at the top of the domain.
        dz = gr.zm[:, gr.k_ub_zm] - gr.zm[:, gr.k_ub_zm - gr.grid_dir_indx]
        xm_density_weighted = (
            rho_ds_zt[:, gr.k_ub_zt]
            * (xm_out[:, gr.k_ub_zt] - xm_enter_mfl[:, gr.k_ub_zt])
            * dz
        )
        if gr.grid_dir_indx > 0:
            integral_mask = jnp.arange(nzt) < gr.k_ub_zt
        else:
            integral_mask = jnp.arange(nzt) > gr.k_ub_zt
        xm_vert_integral = jnp.sum(
            jnp.where(
                integral_mask[None, :],
                rho_ds_zt * xm_out * gr.grid_dir * gr.dzt,
                zero,
            ),
            axis=1,
        )
        l_top_spike = (
            jnp.abs(xm_out[:, gr.k_ub_zt] - xm_enter_mfl[:, gr.k_ub_zt])
            > 10.0 * xm_tol
        )
        l_small_integral = jnp.abs(xm_vert_integral) < eps
        xm_top_removed = xm_out.at[:, gr.k_ub_zt].set(xm_enter_mfl[:, gr.k_ub_zt])
        xm_adj_coef = xm_density_weighted / jnp.where(
            jnp.abs(xm_vert_integral) > zero,
            xm_vert_integral,
            one,
        )
        xm_adj_coef = jnp.maximum(xm_adj_coef, -0.99)
        xm_scaled = (xm_out * (one + xm_adj_coef)[:, None]).at[:, gr.k_ub_zt].set(
            xm_enter_mfl[:, gr.k_ub_zt]
        )
        xm_out = jnp.where(
            l_top_spike[:, None],
            jnp.where(l_small_integral[:, None], xm_top_removed, xm_scaled),
            xm_out,
        )
        return xm_out, err_info_out

    xm, err_info = jax.lax.cond(
        l_any_adjustment_needed,
        adjust_xm,
        lambda args: args,
        (xm, err_info),
    )

    if stats.l_sample:
        stats = stats.update(name_xm_without_ta, xm_without_ta)
        stats = stats.update(name_xm_mfl_min, min_x_allowable)
        stats = stats.update(name_xm_mfl_max, max_x_allowable)
        stats = stats.update(name_wpxp_mfl_min, wpxp_mfl_min)
        stats = stats.update(name_wpxp_mfl_max, wpxp_mfl_max)
        stats = stats.finalize_budget(name_wpxp_mfl, wpxp / dt)
        stats = stats.finalize_budget(name_xm_mfl, xm / dt)
        stats = stats.update(name_xm_exit_mfl, xm)
        stats = stats.update(name_wpxp_exit_mfl, wpxp)

    return xm, wpxp, err_info, stats


@partial(
    jax.jit,
    static_argnames=("nzm", "nzt", "ngrdcol", "l_implemented", "l_upwind_xm_ma"),
)
def mfl_xm_lhs(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    dt: float,
    weights_zt2zm,
    invrs_dzt,
    invrs_dzm,
    wm_zt,
    l_implemented: bool,
    l_upwind_xm_ma: bool,
    grid_dir: float,
):
    """Set up the left-hand side of the monotonic-flux xm tridiagonal system."""
    if not l_implemented:
        lhs = term_ma_zt_lhs(
            nzm,
            nzt,
            ngrdcol,
            wm_zt,
            weights_zt2zm,
            invrs_dzt,
            invrs_dzm,
            l_upwind_xm_ma,
            grid_dir,
        )
    else:
        lhs = jnp.zeros((ndiags3, ngrdcol, nzt), dtype=jnp.float64)

    # LHS xm time tendency.
    lhs = lhs.at[1, :, :].add(one / dt)
    return lhs


@partial(jax.jit, static_argnames=("nzm", "nzt", "ngrdcol"))
def mfl_xm_rhs(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    dt: float,
    xm_old,
    wpxp,
    xm_forcing,
    invrs_dzt,
    rho_ds_zm,
    invrs_rho_ds_zt,
):
    """Set up the right-hand side of the monotonic-flux xm tridiagonal system."""
    del nzm, nzt, ngrdcol
    invrs_dt = one / dt
    return (
        jnp.asarray(xm_old) * invrs_dt
        - jnp.asarray(invrs_rho_ds_zt)
        * jnp.asarray(invrs_dzt)
        * (
            jnp.asarray(rho_ds_zm)[:, 1:] * jnp.asarray(wpxp)[:, 1:]
            - jnp.asarray(rho_ds_zm)[:, :-1] * jnp.asarray(wpxp)[:, :-1]
        )
        + jnp.asarray(xm_forcing)
    )


@partial(
    jax.jit,
    static_argnames=("nzt", "ngrdcol", "solve_type", "tridiag_solve_method", "l_implemented"),
)
def mfl_xm_solve(
    nzt: int,
    ngrdcol: int,
    gr,
    solve_type: int,
    tridiag_solve_method: int,
    l_implemented: bool,
    lhs,
    rhs,
    err_info,
):
    """Solve the monotonic-flux xm tridiagonal matrix equation."""
    if solve_type == mono_flux_rtm:
        solve_type_str = "rtm"
    elif solve_type == mono_flux_thlm:
        solve_type_str = "thlm"
    else:
        solve_type_str = "scalars"

    if l_force_descending_solves and gr.grid_dir_indx > 0:
        lhs = lhs[::-1, :, ::-1]
        rhs = rhs[:, ::-1]

    err_info, xm, _rcond = tridiag_solve(
        solve_type_str,
        tridiag_solve_method,
        ngrdcol,
        nzt,
        lhs,
        rhs,
        err_info,
        l_implemented=l_implemented,
    )

    if l_force_descending_solves and gr.grid_dir_indx > 0:
        xm = xm[:, ::-1]

    return xm, err_info


@partial(jax.jit, static_argnames=("nzm", "nzt", "ngrdcol"))
def calc_turb_adv_range(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    dt: float,
    w_1_zm,
    w_2_zm,
    varnce_w_1_zm,
    varnce_w_2_zm,
    mixt_frac_zm,
    stats,
):
    """Calculate the lowermost and uppermost levels affected by turbulent advection."""
    # Toggle constant or variable thickness. The Fortran parameter is .false.
    l_constant_thickness = False
    del l_constant_thickness

    invrs_dt = one / dt
    w_min = gr.grid_dir * gr.dzm * invrs_dt

    # Find the average upwards vertical velocity and the average downwards
    # vertical velocity.
    mean_w_down_zm, mean_w_up_zm, stats = mean_vert_vel_up_down(
        nzm,
        ngrdcol,
        w_1_zm,
        w_2_zm,
        varnce_w_1_zm,
        varnce_w_2_zm,
        mixt_frac_zm,
        zero,
        w_min,
        stats,
    )

    if gr.grid_dir_indx > 0:
        low_lev_effect, high_lev_effect = _calc_turb_adv_range_ascending(
            nzt, ngrdcol, gr, dt, mean_w_down_zm, mean_w_up_zm,
        )
    else:
        low_lev_effect, high_lev_effect = _calc_turb_adv_range_descending(
            nzt, ngrdcol, gr, dt, mean_w_down_zm, mean_w_up_zm,
        )

    return low_lev_effect, high_lev_effect, stats


# Target-only split of the two grid-direction branches inside Fortran
# `calc_turb_adv_range`; keep these helpers adjacent to that routine.
def _calc_turb_adv_range_ascending(nzt, ngrdcol, gr, dt, mean_w_down_zm, mean_w_up_zm):
    """Ascending-grid branch of calc_turb_adv_range."""
    low_lev_effect = jnp.zeros((ngrdcol, nzt), dtype=jnp.int32)
    high_lev_effect = jnp.zeros((ngrdcol, nzt), dtype=jnp.int32)

    for k in range(1, nzt - 2):
        start_low = jnp.full((ngrdcol,), k - 1, dtype=jnp.int32)

        def low_cond(carry):
            j, _dt_all, _low, done = carry
            return jnp.any((j >= 0) & (~done))

        def low_body(carry):
            j, dt_all, low, done = carry
            active = (j >= 0) & (~done)
            j_adj = j + 1
            vert_vel_up = jnp.take_along_axis(mean_w_up_zm, j_adj[:, None], axis=1)[:, 0]
            dzm = jnp.take_along_axis(gr.dzm, j_adj[:, None], axis=1)[:, 0]
            has_up = vert_vel_up > zero
            dt_one = gr.grid_dir * dzm / jnp.where(has_up, vert_vel_up, one)
            dt_next = jnp.where(active & has_up, dt_all + dt_one, dt_all)
            reached = active & has_up & (dt_next >= dt)
            no_up = active & (~has_up)
            low_next = jnp.where(active, j, low)
            low_next = jnp.where(no_up, j + gr.grid_dir_indx, low_next)
            done_next = done | reached | no_up
            j_next = jnp.where(active & (~done_next), j - 1, j)
            return j_next, dt_next, low_next, done_next

        _j, _dt_all, low_k, _done = jax.lax.while_loop(
            low_cond,
            low_body,
            (
                start_low,
                jnp.zeros((ngrdcol,), dtype=jnp.float64),
                start_low,
                jnp.zeros((ngrdcol,), dtype=bool),
            ),
        )
        low_lev_effect = low_lev_effect.at[:, k].set(low_k)

        start_high = jnp.full((ngrdcol,), k + 1, dtype=jnp.int32)

        def high_cond(carry):
            j, _dt_all, _high, done = carry
            return jnp.any((j <= nzt - 1) & (~done))

        def high_body(carry):
            j, dt_all, high, done = carry
            active = (j <= nzt - 1) & (~done)
            j_adj = j
            vert_vel_down = jnp.take_along_axis(mean_w_down_zm, j_adj[:, None], axis=1)[:, 0]
            dzm = jnp.take_along_axis(gr.dzm, j_adj[:, None], axis=1)[:, 0]
            has_down = vert_vel_down < zero
            dt_one = -gr.grid_dir * dzm / jnp.where(has_down, vert_vel_down, -one)
            dt_next = jnp.where(active & has_down, dt_all + dt_one, dt_all)
            reached = active & has_down & (dt_next >= dt)
            no_down = active & (~has_down)
            high_next = jnp.where(active, j, high)
            high_next = jnp.where(no_down, j - gr.grid_dir_indx, high_next)
            done_next = done | reached | no_down
            j_next = jnp.where(active & (~done_next), j + 1, j)
            return j_next, dt_next, high_next, done_next

        _j, _dt_all, high_k, _done = jax.lax.while_loop(
            high_cond,
            high_body,
            (
                start_high,
                jnp.zeros((ngrdcol,), dtype=jnp.float64),
                start_high,
                jnp.zeros((ngrdcol,), dtype=bool),
            ),
        )
        high_lev_effect = high_lev_effect.at[:, k].set(high_k)

    # Information for levels gr%k_lb_zt, gr%k_ub_zt-gr%grid_dir_indx, and
    # gr%k_ub_zt is not needed. However, set the values at these levels for
    # purposes of not having odd values in the arrays.
    low_lev_effect = low_lev_effect.at[:, gr.k_lb_zt].set(gr.k_lb_zt)
    high_lev_effect = high_lev_effect.at[:, gr.k_lb_zt].set(gr.k_lb_zt)
    low_lev_effect = low_lev_effect.at[:, gr.k_ub_zt - gr.grid_dir_indx].set(
        gr.k_ub_zt - gr.grid_dir_indx
    )
    high_lev_effect = high_lev_effect.at[:, gr.k_ub_zt - gr.grid_dir_indx].set(gr.k_ub_zt)
    low_lev_effect = low_lev_effect.at[:, gr.k_ub_zt].set(gr.k_ub_zt)
    high_lev_effect = high_lev_effect.at[:, gr.k_ub_zt].set(gr.k_ub_zt)

    return low_lev_effect, high_lev_effect


def _calc_turb_adv_range_descending(nzt, ngrdcol, gr, dt, mean_w_down_zm, mean_w_up_zm):
    """Descending-grid branch of calc_turb_adv_range."""
    low_lev_effect = jnp.zeros((ngrdcol, nzt), dtype=jnp.int32)
    high_lev_effect = jnp.zeros((ngrdcol, nzt), dtype=jnp.int32)

    for k in range(nzt - 2, 1, -1):
        start_low = jnp.full((ngrdcol,), k - gr.grid_dir_indx, dtype=jnp.int32)

        def low_cond(carry):
            j, _dt_all, _low, done = carry
            return jnp.any((j <= gr.k_lb_zt) & (~done))

        def low_body(carry):
            j, dt_all, low, done = carry
            active = (j <= gr.k_lb_zt) & (~done)
            j_adj = j
            vert_vel_up = jnp.take_along_axis(mean_w_up_zm, j_adj[:, None], axis=1)[:, 0]
            dzm = jnp.take_along_axis(gr.dzm, j_adj[:, None], axis=1)[:, 0]
            has_up = vert_vel_up > zero
            dt_one = gr.grid_dir * dzm / jnp.where(has_up, vert_vel_up, one)
            dt_next = jnp.where(active & has_up, dt_all + dt_one, dt_all)
            reached = active & has_up & (dt_next >= dt)
            no_up = active & (~has_up)
            low_next = jnp.where(active, j, low)
            low_next = jnp.where(no_up, j + gr.grid_dir_indx, low_next)
            done_next = done | reached | no_up
            j_next = jnp.where(active & (~done_next), j - gr.grid_dir_indx, j)
            return j_next, dt_next, low_next, done_next

        _j, _dt_all, low_k, _done = jax.lax.while_loop(
            low_cond,
            low_body,
            (
                start_low,
                jnp.zeros((ngrdcol,), dtype=jnp.float64),
                start_low,
                jnp.zeros((ngrdcol,), dtype=bool),
            ),
        )
        low_lev_effect = low_lev_effect.at[:, k].set(low_k)

        start_high = jnp.full((ngrdcol,), k + gr.grid_dir_indx, dtype=jnp.int32)

        def high_cond(carry):
            j, _dt_all, _high, done = carry
            return jnp.any((j >= gr.k_ub_zt) & (~done))

        def high_body(carry):
            j, dt_all, high, done = carry
            active = (j >= gr.k_ub_zt) & (~done)
            j_adj = j + 1
            vert_vel_down = jnp.take_along_axis(mean_w_down_zm, j_adj[:, None], axis=1)[:, 0]
            dzm = jnp.take_along_axis(gr.dzm, j_adj[:, None], axis=1)[:, 0]
            has_down = vert_vel_down < zero
            dt_one = -gr.grid_dir * dzm / jnp.where(has_down, vert_vel_down, -one)
            dt_next = jnp.where(active & has_down, dt_all + dt_one, dt_all)
            reached = active & has_down & (dt_next >= dt)
            no_down = active & (~has_down)
            high_next = jnp.where(active, j, high)
            high_next = jnp.where(no_down, j - gr.grid_dir_indx, high_next)
            done_next = done | reached | no_down
            j_next = jnp.where(active & (~done_next), j + gr.grid_dir_indx, j)
            return j_next, dt_next, high_next, done_next

        _j, _dt_all, high_k, _done = jax.lax.while_loop(
            high_cond,
            high_body,
            (
                start_high,
                jnp.zeros((ngrdcol,), dtype=jnp.float64),
                start_high,
                jnp.zeros((ngrdcol,), dtype=bool),
            ),
        )
        high_lev_effect = high_lev_effect.at[:, k].set(high_k)

    low_lev_effect = low_lev_effect.at[:, gr.k_lb_zt].set(gr.k_lb_zt)
    high_lev_effect = high_lev_effect.at[:, gr.k_lb_zt].set(gr.k_lb_zt)
    low_lev_effect = low_lev_effect.at[:, gr.k_ub_zt - gr.grid_dir_indx].set(
        gr.k_ub_zt - gr.grid_dir_indx
    )
    high_lev_effect = high_lev_effect.at[:, gr.k_ub_zt - gr.grid_dir_indx].set(gr.k_ub_zt)
    low_lev_effect = low_lev_effect.at[:, gr.k_ub_zt].set(gr.k_ub_zt)
    high_lev_effect = high_lev_effect.at[:, gr.k_ub_zt].set(gr.k_ub_zt)
    return low_lev_effect, high_lev_effect


@partial(jax.jit, static_argnames=("nzm", "ngrdcol"))
def mean_vert_vel_up_down(
    nzm: int,
    ngrdcol: int,
    w_1_zm,
    w_2_zm,
    varnce_w_1_zm,
    varnce_w_2_zm,
    mixt_frac_zm,
    w_ref: float,
    w_min,
    stats,
):
    """Calculate the mean up and down vertical velocities from the w PDF."""
    mean_w_down_1st, mean_w_up_1st = calc_mean_w_up_down_component(
        nzm, ngrdcol, w_1_zm, varnce_w_1_zm, w_ref, w_min,
    )
    mean_w_down_2nd, mean_w_up_2nd = calc_mean_w_up_down_component(
        nzm, ngrdcol, w_2_zm, varnce_w_2_zm, w_ref, w_min,
    )

    mixt_frac_zm = jnp.asarray(mixt_frac_zm, dtype=jnp.float64)

    # Overall mean of downwards w.
    mean_w_down = (
        mixt_frac_zm * mean_w_down_1st
        + (one - mixt_frac_zm) * mean_w_down_2nd
    )

    # Overall mean of upwards w.
    mean_w_up = (
        mixt_frac_zm * mean_w_up_1st
        + (one - mixt_frac_zm) * mean_w_up_2nd
    )

    if stats.l_sample:
        stats = stats.update("mean_w_up", mean_w_up)
        stats = stats.update("mean_w_down", mean_w_down)

    return mean_w_down, mean_w_up, stats


@partial(jax.jit, static_argnames=("nzm", "ngrdcol"))
def calc_mean_w_up_down_component(
    nzm: int,
    ngrdcol: int,
    w_i_zm,
    varnce_w_i_zm,
    w_ref: float,
    w_min,
):
    """Split one PDF component of vertical velocity into upward and downward components."""
    del ngrdcol
    w_i_zm = jnp.asarray(w_i_zm, dtype=jnp.float64)
    varnce_w_i_zm = jnp.asarray(varnce_w_i_zm, dtype=jnp.float64)
    w_min = jnp.asarray(w_min, dtype=jnp.float64)

    invrs_sqrt_2pi = one / jnp.sqrt(2.0 * jnp.pi)
    sqrt_2 = jnp.sqrt(2.0)

    sigma_w_i_zm = jnp.sqrt(jnp.maximum(varnce_w_i_zm, zero))
    sigma_safe = jnp.where(sigma_w_i_zm > zero, sigma_w_i_zm, one)

    too_weak = jnp.abs(w_i_zm) + 3.0 * sigma_w_i_zm <= w_min
    all_down = (~too_weak) & (w_i_zm + 3.0 * sigma_w_i_zm <= w_ref)
    all_up = (~too_weak) & (~all_down) & (w_i_zm - 3.0 * sigma_w_i_zm >= w_ref)

    exp_cache = jnp.exp(-((w_ref - w_i_zm) ** 2) / (2.0 * sigma_safe ** 2))
    erf_cache = erf((w_ref - w_i_zm) / (sqrt_2 * sigma_safe))

    mean_w_down_mixed = (
        -sigma_w_i_zm * invrs_sqrt_2pi * exp_cache
        + w_i_zm * 0.5 * (one + erf_cache)
    )
    mean_w_up_mixed = (
        sigma_w_i_zm * invrs_sqrt_2pi * exp_cache
        + w_i_zm * 0.5 * (one - erf_cache)
    )

    mean_w_down_i = jnp.where(
        too_weak,
        zero,
        jnp.where(all_down, w_i_zm, jnp.where(all_up, zero, mean_w_down_mixed)),
    )
    mean_w_up_i = jnp.where(
        too_weak,
        zero,
        jnp.where(all_down, zero, jnp.where(all_up, w_i_zm, mean_w_up_mixed)),
    )

    # Upper and lower levels are not used, set to 0 to be safe and avoid NaN problems.
    boundary = (jnp.arange(nzm)[None, :] == 0) | (jnp.arange(nzm)[None, :] == nzm - 1)
    mean_w_down_i = jnp.where(boundary, zero, mean_w_down_i)
    mean_w_up_i = jnp.where(boundary, zero, mean_w_up_i)
    return mean_w_down_i, mean_w_up_i

