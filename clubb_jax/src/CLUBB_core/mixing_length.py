"""JAX port of ``src/CLUBB_core/mixing_length.F90``.

Description:
  Compute CLUBB's mixing length and dissipation time scales, including
  Richardson-number-dependent damping terms and Lscale-related stats.

Adaptation notes:
- Stats are threaded explicitly with JaxStats because the Fortran routine uses
  global stats side effects that are not JAX-compatible state.
- ``compute_mixing_length`` uses small scan helpers to express the Fortran
  parcel while-loops in JAX. Those helpers are local language adapters for the
  dynamic loop structure, not physics abstractions.
- The direct parcel-length branch currently mirrors the ascending-grid path
  used by the SCM cases. The diagnosed-Lscale branch is fully vectorized.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.advance_helper_module import (
    calc_Ri_zm,
    smooth_heaviside_peskin,
    smooth_max,
)
from clubb_jax.src.CLUBB_core.clubb_constants import (
    Cp,
    Lv,
    Rd,
    em_min,
    ep,
    ep1,
    ep2,
    eps,
    grav,
    iC_invrs_tau_N2,
    iC_invrs_tau_N2_clear_wp3,
    iC_invrs_tau_N2_wp2,
    iC_invrs_tau_N2_wpxp,
    iC_invrs_tau_N2_xp2,
    iC_invrs_tau_bkgnd,
    iC_invrs_tau_sfc,
    iC_invrs_tau_shear,
    iC_invrs_tau_wpxp_N2_thresh,
    iC_invrs_tau_wpxp_Ri,
    ialtitude_threshold,
    imu,
    itaumax,
    iwpxp_Ri_exp,
    iz_displace,
    min_max_smth_mag,
    one,
    one_fourth,
    one_half,
    rt_tol,
    thl_tol,
    two,
    unused_var,
    vonk,
    zero,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.error_code import clubb_at_least_debug_level
from clubb_jax.src.CLUBB_core.grid_class import zm2zt, zm2zt2zm, zt2zm
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq


def calc_Lscale(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    l_implemented: bool,
    host_dx,
    host_dy,
    p_in_Pa,
    exner,
    rtm,
    thlm,
    thvm,
    thlp2,
    rtp2,
    rtpthlp,
    pdf_params,
    em,
    thv_ds_zt,
    lmin,
    upwp_sfc,
    vpwp_sfc,
    ddzt_umvm_sqd,
    ice_supersat_frac,
    ufmin,
    tau_const,
    sfc_elevation,
    clubb_params,
    saturation_formula: int,
    l_Lscale_plume_centered: bool,
    l_diag_Lscale_from_tau: bool,
    l_e3sm_config: bool,
    l_smooth_Heaviside_tau_wpxp: bool,
    l_modify_limiters_for_cnvg_test: bool,
    l_use_invrs_tau_N2_iso: bool,
    brunt_vaisala_freq_sqd_smth,
    stats: JaxStats,
    err_info,
):
    """Compute CLUBB's mixing length and dissipation time scales."""
    l_smooth_min_max = False

    Lscale_max = set_Lscale_max(ngrdcol, l_implemented, host_dx, host_dy)

    # Interpolate rtpthlp here so the diagnostic stats value is generated at
    # the same point as the Lscale calculation that consumes it.
    rtpthlp_zt = zm2zt(nzm, nzt, ngrdcol, gr, rtpthlp)
    stats = stats.update("rtpthlp_zt", rtpthlp_zt)

    # Calculate Richardson number Ri_zm.
    if l_modify_limiters_for_cnvg_test:
        Ri_zm = calc_Ri_zm(
            nzm,
            ngrdcol,
            brunt_vaisala_freq_sqd_smth,
            ddzt_umvm_sqd,
            0.0,
            1.0e-12,
        )
        Ri_zm = zm2zt2zm(nzm, nzt, ngrdcol, gr, Ri_zm)
    else:
        if l_smooth_min_max:
            brunt_vaisala_freq_clipped = smooth_max(
                nzm,
                ngrdcol,
                1.0e-7,
                brunt_vaisala_freq_sqd_smth,
                1.0e-4 * min_max_smth_mag,
            )
            ddzt_umvm_sqd_clipped = smooth_max(
                nzm,
                ngrdcol,
                ddzt_umvm_sqd,
                1.0e-7,
                1.0e-6 * min_max_smth_mag,
            )
            Ri_zm = calc_Ri_zm(
                nzm,
                ngrdcol,
                brunt_vaisala_freq_clipped,
                ddzt_umvm_sqd_clipped,
                0.0,
                0.0,
            )
        else:
            Ri_zm = calc_Ri_zm(
                nzm,
                ngrdcol,
                brunt_vaisala_freq_sqd_smth,
                ddzt_umvm_sqd,
                1.0e-7,
                1.0e-7,
            )

    stats = stats.update("Ri_zm", Ri_zm)

    if not l_diag_Lscale_from_tau:
        # Compute Lscale first using the buoyant parcel calculation.
        newmu = clubb_params[:, imu]

        # Interpolate thlp2 and rtp2 to thermodynamic levels.
        thlp2_zt = zm2zt(nzm, nzt, ngrdcol, gr, thlp2, thl_tol**2)
        rtp2_zt = zm2zt(nzm, nzt, ngrdcol, gr, rtp2, rt_tol**2)

        err_info, Lscale, Lscale_up, Lscale_down, stats = calc_Lscale_directly(
            ngrdcol,
            nzm,
            nzt,
            gr,
            l_implemented,
            p_in_Pa,
            exner,
            rtm,
            thlm,
            thvm,
            newmu,
            rtp2_zt,
            thlp2_zt,
            rtpthlp_zt,
            pdf_params,
            em,
            thv_ds_zt,
            Lscale_max,
            lmin,
            clubb_params,
            saturation_formula,
            l_Lscale_plume_centered,
            stats,
            err_info,
        )

        # Calculate CLUBB's turbulent eddy-turnover time scale as CLUBB's
        # length scale divided by a velocity scale.
        em_zt = zm2zt(nzm, nzt, ngrdcol, gr, em, em_min)
        tau_zt = jnp.minimum(
            Lscale / jnp.sqrt(em_zt),
            clubb_params[:, itaumax][:, None],
        )

        Lscale_zm_for_tau = zt2zm(nzm, nzt, ngrdcol, gr, Lscale, zero_threshold)
        tau_zm = jnp.minimum(
            Lscale_zm_for_tau / jnp.sqrt(jnp.maximum(em_min, em)),
            clubb_params[:, itaumax][:, None],
        )

        invrs_tau_zm = one / tau_zm
        invrs_tau_wp2_zm = invrs_tau_zm
        invrs_tau_xp2_zm = invrs_tau_zm
        invrs_tau_wpxp_zm = invrs_tau_zm
        invrs_tau_wp3_zm = invrs_tau_zm
        tau_max_zm = jnp.broadcast_to(clubb_params[:, itaumax][:, None], (ngrdcol, nzm))

        invrs_tau_zt = one / tau_zt
        invrs_tau_wp3_zt = invrs_tau_zt
        tau_max_zt = jnp.broadcast_to(clubb_params[:, itaumax][:, None], (ngrdcol, nzt))

        invrs_tau_sfc = jnp.zeros((ngrdcol, nzm))
        invrs_tau_no_N2_zm = jnp.zeros((ngrdcol, nzm))
        invrs_tau_bkgnd = jnp.zeros((ngrdcol, nzm))
        invrs_tau_shear = jnp.zeros((ngrdcol, nzm))
        invrs_tau_N2_iso = jnp.zeros((ngrdcol, nzm))
    else:
        # Diagnose simple tau and Lscale.
        (
            err_info,
            invrs_tau_zt,
            invrs_tau_zm,
            invrs_tau_sfc,
            invrs_tau_no_N2_zm,
            invrs_tau_bkgnd,
            invrs_tau_shear,
            invrs_tau_N2_iso,
            invrs_tau_wp2_zm,
            invrs_tau_xp2_zm,
            invrs_tau_wp3_zm,
            invrs_tau_wp3_zt,
            invrs_tau_wpxp_zm,
            tau_max_zm,
            tau_max_zt,
            tau_zm,
            tau_zt,
            Lscale,
            Lscale_up,
            Lscale_down,
            stats,
        ) = diagnose_Lscale_from_tau(
            nzm,
            nzt,
            ngrdcol,
            gr,
            upwp_sfc,
            vpwp_sfc,
            ddzt_umvm_sqd,
            ice_supersat_frac,
            em,
            ufmin,
            tau_const,
            sfc_elevation,
            Lscale_max,
            clubb_params,
            stats,
            l_e3sm_config,
            l_smooth_Heaviside_tau_wpxp,
            brunt_vaisala_freq_sqd_smth,
            Ri_zm,
            err_info,
        )

    Lscale_zm = zt2zm(nzm, nzt, ngrdcol, gr, Lscale)

    invrs_tau_C6_zm = invrs_tau_wpxp_zm
    invrs_tau_C1_zm = invrs_tau_wp2_zm
    invrs_tau_C14_zm = invrs_tau_wp2_zm

    if (not l_diag_Lscale_from_tau) and l_use_invrs_tau_N2_iso:
        err_info = err_info.set_fatal()

    if not l_use_invrs_tau_N2_iso:
        invrs_tau_C4_zm = invrs_tau_wp2_zm
    else:
        invrs_tau_C4_zm = invrs_tau_N2_iso

    stats = stats.update("Lscale", Lscale)
    stats = stats.update("Lscale_up", Lscale_up)
    stats = stats.update("Lscale_down", Lscale_down)
    stats = stats.update("tau_zm", tau_zm)
    stats = stats.update("invrs_tau_zm", invrs_tau_zm)
    stats = stats.update("invrs_tau_xp2_zm", invrs_tau_xp2_zm)
    stats = stats.update("invrs_tau_wp2_zm", invrs_tau_wp2_zm)
    stats = stats.update("invrs_tau_wpxp_zm", invrs_tau_wpxp_zm)
    stats = stats.update("invrs_tau_wp3_zm", invrs_tau_wp3_zm)
    stats = stats.update("tau_zt", tau_zt)

    if l_diag_Lscale_from_tau:
        stats = stats.update("invrs_tau_no_N2_zm", invrs_tau_no_N2_zm)
        stats = stats.update("invrs_tau_bkgnd", invrs_tau_bkgnd)
        stats = stats.update("invrs_tau_sfc", invrs_tau_sfc)
        stats = stats.update("invrs_tau_shear", invrs_tau_shear)

    return (
        err_info,
        invrs_tau_zt,
        invrs_tau_zm,
        invrs_tau_xp2_zm,
        invrs_tau_wp3_zt,
        invrs_tau_C1_zm,
        invrs_tau_C4_zm,
        invrs_tau_C6_zm,
        invrs_tau_C14_zm,
        tau_max_zm,
        tau_max_zt,
        tau_zm,
        Lscale,
        Lscale_zm,
        Lscale_up,
        Lscale_down,
        stats,
    )


def set_Lscale_max(
    ngrdcol: int,
    l_implemented: bool,
    host_dx,
    host_dy,
):
    """Set the maximum allowable value of Lscale."""
    if l_implemented:
        return 0.25 * jnp.minimum(host_dx, host_dy)
    return jnp.full((ngrdcol,), 1.0e5)


def _bounded_while(cond_fn, body_fn, init_state, max_iters: int):
    """Run a fixed scan that freezes state after the Fortran while condition fails."""

    def step(state, _):
        run = cond_fn(state)
        new_state = body_fn(state)
        state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(run, new, old),
            state,
            new_state,
        )
        return state, None

    final, _ = jax.lax.scan(step, init_state, None, length=int(max_iters))
    return final


def _parcel_thv(thl_par, rt_par, exner_j, p_j, thv_ds_j, Lv_coef_j, saturation_formula):
    tl_par_j = thl_par * exner_j
    rsatl_par_j = sat_mixrat_liq(p_j, tl_par_j, saturation_formula)
    tl_par_j_sqd = tl_par_j**2
    s_par_j = (
        (rt_par - rsatl_par_j)
        * tl_par_j_sqd
        / (tl_par_j_sqd + ep * Lv**2 / (Rd * Cp) * rsatl_par_j)
    )
    rc_par_j = jnp.maximum(s_par_j, zero_threshold)
    return thl_par + ep1 * thv_ds_j * rt_par + Lv_coef_j * rc_par_j


def _upward_inner_while(
    k,
    tke,
    thl_par_j,
    rt_par_j,
    dCAPE_dz_j_minus_1,
    thl_par_j_precalc,
    rt_par_j_precalc,
    exp_mu_dzm,
    grav_on_thvm,
    Lv_coef,
    thv_ds,
    exner,
    p_in_Pa,
    thvm,
    dzm,
    invrs_dzm,
    zt,
    k_ub_zt,
    saturation_formula,
):
    init_state = (
        k + 2,
        tke,
        thl_par_j,
        rt_par_j,
        dCAPE_dz_j_minus_1,
        jnp.asarray(False),
        k + 1,
        tke,
        dCAPE_dz_j_minus_1,
        jnp.asarray(0.0),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcape_prev, done, *_rest = state
        return (~done) & (j < k_ub_zt)

    def body_fn(state):
        j, tke, thl_par_j, rt_par_j, dCAPE_dz_j_minus_1, done, j_last, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = state
        thl_par_j_new = thl_par_j_precalc[j] + thl_par_j * exp_mu_dzm[j]
        rt_par_j_new = rt_par_j_precalc[j] + rt_par_j * exp_mu_dzm[j]
        thv_par_j = _parcel_thv(
            thl_par_j_new,
            rt_par_j_new,
            exner[j],
            p_in_Pa[j],
            thv_ds[j],
            Lv_coef[j],
            saturation_formula,
        )
        dCAPE_dz_j = grav_on_thvm[j] * (thv_par_j - thvm[j])
        CAPE_incr = one_half * (dCAPE_dz_j + dCAPE_dz_j_minus_1) * dzm[j]
        tke_new = tke + CAPE_incr
        exhausted = tke_new <= zero
        newly_exhausted = exhausted & (~done)

        return (
            jnp.where(exhausted, j, j + 1),
            jnp.where(exhausted, tke, tke_new),
            jnp.where(exhausted, thl_par_j, thl_par_j_new),
            jnp.where(exhausted, rt_par_j, rt_par_j_new),
            jnp.where(exhausted, dCAPE_dz_j_minus_1, dCAPE_dz_j),
            done | exhausted,
            jnp.where(exhausted, j_last, j),
            jnp.where(newly_exhausted, tke, tke_exit),
            jnp.where(newly_exhausted, dCAPE_dz_j_minus_1, dCAPE_exit_prev),
            jnp.where(newly_exhausted, dCAPE_dz_j, dCAPE_exit_j),
        )

    final = _bounded_while(cond_fn, body_fn, init_state, k_ub_zt)
    j, _tke, _thl, _rt, _dcape_prev, done, j_last, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = final
    return j_last, done, j, tke_exit, dCAPE_exit_prev, dCAPE_exit_j


def _downward_inner_while(
    k,
    tke,
    thl_par_j,
    rt_par_j,
    dCAPE_dz_j_plus_1,
    thl_par_j_precalc,
    rt_par_j_precalc,
    exp_mu_dzm,
    grav_on_thvm,
    Lv_coef,
    thv_ds,
    exner,
    p_in_Pa,
    thvm,
    dzm,
    invrs_dzm,
    zt,
    k_lb_zt,
    k_ub_zt,
    saturation_formula,
):
    init_state = (
        k - 2,
        tke,
        thl_par_j,
        rt_par_j,
        dCAPE_dz_j_plus_1,
        jnp.asarray(False),
        k - 1,
        tke,
        dCAPE_dz_j_plus_1,
        jnp.asarray(0.0),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcape_next, done, *_rest = state
        return (~done) & (j >= k_lb_zt)

    def body_fn(state):
        j, tke, thl_par_j, rt_par_j, dCAPE_dz_j_plus_1, done, j_last, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = state
        thl_par_j_new = thl_par_j_precalc[j] + thl_par_j * exp_mu_dzm[j + 1]
        rt_par_j_new = rt_par_j_precalc[j] + rt_par_j * exp_mu_dzm[j + 1]
        thv_par_j = _parcel_thv(
            thl_par_j_new,
            rt_par_j_new,
            exner[j],
            p_in_Pa[j],
            thv_ds[j],
            Lv_coef[j],
            saturation_formula,
        )
        dCAPE_dz_j = grav_on_thvm[j] * (thv_par_j - thvm[j])
        CAPE_incr = one_half * (dCAPE_dz_j + dCAPE_dz_j_plus_1) * dzm[j + 1]
        tke_new = tke - CAPE_incr
        exhausted = tke_new <= zero
        newly_exhausted = exhausted & (~done)

        return (
            jnp.where(exhausted, j, j - 1),
            jnp.where(exhausted, tke, tke_new),
            jnp.where(exhausted, thl_par_j, thl_par_j_new),
            jnp.where(exhausted, rt_par_j, rt_par_j_new),
            jnp.where(exhausted, dCAPE_dz_j_plus_1, dCAPE_dz_j),
            done | exhausted,
            jnp.where(exhausted, j_last, j),
            jnp.where(newly_exhausted, tke, tke_exit),
            jnp.where(newly_exhausted, dCAPE_dz_j_plus_1, dCAPE_exit_plus1),
            jnp.where(newly_exhausted, dCAPE_dz_j, dCAPE_exit_j),
        )

    final = _bounded_while(cond_fn, body_fn, init_state, k_ub_zt)
    j, _tke, _thl, _rt, _dcape_next, done, j_last, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = final
    return j_last, done, j, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j


def _compute_lscale_up_col(
    tke_i,
    thl_par_1,
    rt_par_1,
    dCAPE_dz_1,
    CAPE_incr_1,
    thl_par_j_precalc,
    rt_par_j_precalc,
    exp_mu_dzm,
    grav_on_thvm,
    Lv_coef,
    thv_ds,
    exner,
    p_in_Pa,
    thvm,
    dzm,
    invrs_dzm,
    zt,
    k_ub_zt,
    saturation_formula,
    nzt,
):
    zlmin = 0.1

    def outer_step(Lscale_up_max_alt, k):
        tke_after_first_level = tke_i[k] + CAPE_incr_1[k + 1]
        dCAPE_dz_k_plus_1 = dCAPE_dz_1[k + 1]
        dCAPE_safe = jnp.where(jnp.abs(dCAPE_dz_k_plus_1) > 0.0, dCAPE_dz_k_plus_1, 1.0)
        frac_first = (
            -jnp.sqrt(jnp.maximum(zero_threshold, -two * tke_i[k] * dzm[k + 1] * dCAPE_dz_k_plus_1))
            / dCAPE_safe
        )

        (
            j_last,
            exited_early,
            j,
            tke_exit,
            dCAPE_exit_prev,
            dCAPE_exit_j,
        ) = _upward_inner_while(
            k,
            tke_after_first_level,
            thl_par_1[k + 1],
            rt_par_1[k + 1],
            dCAPE_dz_1[k + 1],
            thl_par_j_precalc,
            rt_par_j_precalc,
            exp_mu_dzm,
            grav_on_thvm,
            Lv_coef,
            thv_ds,
            exner,
            p_in_Pa,
            thvm,
            dzm,
            invrs_dzm,
            zt,
            k_ub_zt,
            saturation_formula,
        )

        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_prev
        linear_case = jnp.abs(dCAPE_diff) * two <= jnp.abs(dCAPE_exit_j + dCAPE_exit_prev) * eps
        dCAPE_j_safe = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = -tke_exit / dCAPE_j_safe

        dCAPE_diff_safe = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_dCAPE_diff = one / dCAPE_diff_safe
        discriminant = dCAPE_exit_prev**2 - two * tke_exit * invrs_dzm[j] * dCAPE_diff
        frac_quadratic = (
            -dCAPE_exit_prev * invrs_dCAPE_diff * dzm[j]
            - jnp.sqrt(jnp.maximum(zero_threshold, discriminant)) * invrs_dCAPE_diff * dzm[j]
        )
        frac_exhausted = jnp.where(linear_case, frac_linear, frac_quadratic)
        frac_exhausted = jnp.where(exited_early, frac_exhausted, zero)

        Lscale_up_k = jnp.where(
            tke_after_first_level > zero,
            zlmin + zt[j_last] - zt[k] + frac_exhausted,
            zlmin + frac_first,
        )
        parcel_top = zt[k] + Lscale_up_k
        Lscale_up_k = jnp.where(
            parcel_top < Lscale_up_max_alt,
            Lscale_up_max_alt - zt[k],
            Lscale_up_k,
        )
        Lscale_up_max_alt = jnp.where(
            parcel_top < Lscale_up_max_alt,
            Lscale_up_max_alt,
            parcel_top,
        )
        return Lscale_up_max_alt, Lscale_up_k

    _, values = jax.lax.scan(outer_step, jnp.asarray(0.0), jnp.arange(nzt - 2))
    return jnp.concatenate([values, jnp.full((2,), zlmin)])


def _compute_lscale_down_col(
    tke_i,
    thl_par_1,
    rt_par_1,
    dCAPE_dz_1,
    CAPE_incr_1,
    thl_par_j_precalc,
    rt_par_j_precalc,
    exp_mu_dzm,
    grav_on_thvm,
    Lv_coef,
    thv_ds,
    exner,
    p_in_Pa,
    thvm,
    dzm,
    invrs_dzm,
    zt,
    k_lb_zt,
    k_ub_zt,
    saturation_formula,
    nzt,
):
    zlmin = 0.1

    def outer_step(Lscale_down_min_alt, offset):
        k = nzt - 1 - offset
        tke_after_first_level = tke_i[k] - CAPE_incr_1[k - 1]
        dCAPE_dz_k_minus_1 = dCAPE_dz_1[k - 1]
        dCAPE_safe = jnp.where(jnp.abs(dCAPE_dz_k_minus_1) > 0.0, dCAPE_dz_k_minus_1, 1.0)
        frac_first = (
            jnp.sqrt(jnp.maximum(zero_threshold, two * tke_i[k] * dzm[k] * dCAPE_dz_k_minus_1))
            / dCAPE_safe
        )

        (
            j_last,
            exited_early,
            j,
            tke_exit,
            dCAPE_exit_plus1,
            dCAPE_exit_j,
        ) = _downward_inner_while(
            k,
            tke_after_first_level,
            thl_par_1[k - 1],
            rt_par_1[k - 1],
            dCAPE_dz_1[k - 1],
            thl_par_j_precalc,
            rt_par_j_precalc,
            exp_mu_dzm,
            grav_on_thvm,
            Lv_coef,
            thv_ds,
            exner,
            p_in_Pa,
            thvm,
            dzm,
            invrs_dzm,
            zt,
            k_lb_zt,
            k_ub_zt,
            saturation_formula,
        )

        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_plus1
        linear_case = jnp.abs(dCAPE_diff) * two <= jnp.abs(dCAPE_exit_j + dCAPE_exit_plus1) * eps
        dCAPE_j_safe = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = tke_exit / dCAPE_j_safe

        dCAPE_diff_safe = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_dCAPE_diff = one / dCAPE_diff_safe
        discriminant = dCAPE_exit_plus1**2 + two * tke_exit * invrs_dzm[j + 1] * dCAPE_diff
        frac_quadratic = (
            -dCAPE_exit_plus1 * invrs_dCAPE_diff * dzm[j + 1]
            + jnp.sqrt(jnp.maximum(zero_threshold, discriminant)) * invrs_dCAPE_diff * dzm[j + 1]
        )
        frac_exhausted = jnp.where(linear_case, frac_linear, frac_quadratic)
        frac_exhausted = jnp.where(exited_early, frac_exhausted, zero)

        Lscale_down_k = jnp.where(
            tke_after_first_level > zero,
            zlmin + zt[k] - zt[j_last] + frac_exhausted,
            zlmin + frac_first,
        )
        parcel_bottom = zt[k] - Lscale_down_k
        Lscale_down_k = jnp.where(
            parcel_bottom > Lscale_down_min_alt,
            zt[k] - Lscale_down_min_alt,
            Lscale_down_k,
        )
        Lscale_down_min_alt = jnp.where(
            parcel_bottom > Lscale_down_min_alt,
            Lscale_down_min_alt,
            parcel_bottom,
        )
        return Lscale_down_min_alt, (k, Lscale_down_k)

    _, (indices, values) = jax.lax.scan(
        outer_step,
        zt[k_ub_zt],
        jnp.arange(nzt - 1),
    )
    Lscale_down = jnp.full((nzt,), zlmin)
    return Lscale_down.at[indices].set(values)


def compute_mixing_length(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    thvm,
    thlm,
    rtm,
    em,
    Lscale_max,
    p_in_Pa,
    exner,
    thv_ds,
    mu,
    lmin,
    saturation_formula: int,
    l_implemented: bool,
    err_info,
):
    """Larson's fifth moist, nonlocal length scale."""
    if gr.grid_dir_indx != 1:
        raise NotImplementedError(
            "JAX compute_mixing_length currently supports ascending grids only."
        )

    # The scan helpers index these arrays with traced loop indices, so the
    # direct branch must establish JAX arrays at this boundary.
    thvm = jnp.asarray(thvm)
    thlm = jnp.asarray(thlm)
    rtm = jnp.asarray(rtm)
    em = jnp.asarray(em)
    Lscale_max = jnp.asarray(Lscale_max)
    p_in_Pa = jnp.asarray(p_in_Pa)
    exner = jnp.asarray(exner)
    thv_ds = jnp.asarray(thv_ds)
    mu = jnp.asarray(mu)

    mu_bad = jnp.abs(mu) < eps
    err_info = err_info.set_fatal(mask=mu_bad)
    mu = jnp.where(mu_bad, one, mu)

    zlmin = 0.1
    Lscale_sfclyr_depth = 500.0
    invrs_Lscale_sfclyr_depth = one / Lscale_sfclyr_depth

    tke_i = zm2zt(nzm, nzt, ngrdcol, gr, em)
    grav_on_thvm = grav / thvm
    Lv_coef = Lv / (exner * Cp) - ep2 * thv_ds

    exp_mu_dzm = jnp.exp(-mu[:, None] * gr.dzm)
    invrs_dzm_on_mu = gr.invrs_dzm / mu[:, None]
    entrain_coef = (one - exp_mu_dzm) * invrs_dzm_on_mu

    thl_par_j_precalc_up = jnp.concatenate(
        [
            jnp.zeros((ngrdcol, 1)),
            thlm[:, 1:nzt - 1]
            - thlm[:, :nzt - 2] * exp_mu_dzm[:, 1:nzt - 1]
            - (thlm[:, 1:nzt - 1] - thlm[:, :nzt - 2])
            * entrain_coef[:, 1:nzt - 1],
            jnp.zeros((ngrdcol, 2)),
        ],
        axis=1,
    )
    rt_par_j_precalc_up = jnp.concatenate(
        [
            jnp.zeros((ngrdcol, 1)),
            rtm[:, 1:nzt - 1]
            - rtm[:, :nzt - 2] * exp_mu_dzm[:, 1:nzt - 1]
            - (rtm[:, 1:nzt - 1] - rtm[:, :nzt - 2])
            * entrain_coef[:, 1:nzt - 1],
            jnp.zeros((ngrdcol, 2)),
        ],
        axis=1,
    )

    thl_par_1_up_int = (
        thlm[:, 1:] - (thlm[:, 1:] - thlm[:, :-1]) * entrain_coef[:, 1:nzt]
    )
    rt_par_1_up_int = (
        rtm[:, 1:] - (rtm[:, 1:] - rtm[:, :-1]) * entrain_coef[:, 1:nzt]
    )
    tl_par_1_up_int = thl_par_1_up_int * exner[:, 1:]
    rsatl_par_1_up_int = sat_mixrat_liq(
        p_in_Pa[:, 1:],
        tl_par_1_up_int,
        saturation_formula,
    )
    tl_par_1_up_sqd = tl_par_1_up_int**2
    s_par_1_up_int = (
        (rt_par_1_up_int - rsatl_par_1_up_int)
        * tl_par_1_up_sqd
        / (tl_par_1_up_sqd + ep * Lv**2 / (Rd * Cp) * rsatl_par_1_up_int)
    )
    rc_par_1_up_int = jnp.maximum(s_par_1_up_int, zero_threshold)
    thv_par_1_up_int = (
        thl_par_1_up_int
        + ep1 * thv_ds[:, 1:] * rt_par_1_up_int
        + Lv_coef[:, 1:] * rc_par_1_up_int
    )
    dCAPE_dz_1_up_int = grav_on_thvm[:, 1:] * (
        thv_par_1_up_int - thvm[:, 1:]
    )
    CAPE_incr_1_up_int = one_half * dCAPE_dz_1_up_int * gr.dzm[:, 1:nzt]

    thl_par_1_up = jnp.concatenate(
        [jnp.zeros((ngrdcol, 1)), thl_par_1_up_int],
        axis=1,
    )
    rt_par_1_up = jnp.concatenate(
        [jnp.zeros((ngrdcol, 1)), rt_par_1_up_int],
        axis=1,
    )
    dCAPE_dz_1_up = jnp.concatenate(
        [jnp.zeros((ngrdcol, 1)), dCAPE_dz_1_up_int],
        axis=1,
    )
    CAPE_incr_1_up = jnp.concatenate(
        [jnp.zeros((ngrdcol, 1)), CAPE_incr_1_up_int],
        axis=1,
    )

    thl_par_j_precalc_down = jnp.concatenate(
        [
            thlm[:, :-1]
            - thlm[:, 1:] * exp_mu_dzm[:, 1:nzt]
            - (thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:nzt],
            jnp.zeros((ngrdcol, 2)),
        ],
        axis=1,
    )
    rt_par_j_precalc_down = jnp.concatenate(
        [
            rtm[:, :-1]
            - rtm[:, 1:] * exp_mu_dzm[:, 1:nzt]
            - (rtm[:, :-1] - rtm[:, 1:]) * entrain_coef[:, 1:nzt],
            jnp.zeros((ngrdcol, 2)),
        ],
        axis=1,
    )

    thl_par_1_down_int = (
        thlm[:, :-1] - (thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:nzt]
    )
    rt_par_1_down_int = (
        rtm[:, :-1] - (rtm[:, :-1] - rtm[:, 1:]) * entrain_coef[:, 1:nzt]
    )
    tl_par_1_down_int = thl_par_1_down_int * exner[:, :-1]
    rsatl_par_1_down_int = sat_mixrat_liq(
        p_in_Pa[:, :-1],
        tl_par_1_down_int,
        saturation_formula,
    )
    tl_par_1_down_sqd = tl_par_1_down_int**2
    s_par_1_down_int = (
        (rt_par_1_down_int - rsatl_par_1_down_int)
        * tl_par_1_down_sqd
        / (tl_par_1_down_sqd + ep * Lv**2 / (Rd * Cp) * rsatl_par_1_down_int)
    )
    rc_par_1_down_int = jnp.maximum(s_par_1_down_int, zero_threshold)
    thv_par_1_down_int = (
        thl_par_1_down_int
        + ep1 * thv_ds[:, :-1] * rt_par_1_down_int
        + Lv_coef[:, :-1] * rc_par_1_down_int
    )
    dCAPE_dz_1_down_int = grav_on_thvm[:, :-1] * (
        thv_par_1_down_int - thvm[:, :-1]
    )
    CAPE_incr_1_down_int = (
        one_half * dCAPE_dz_1_down_int * gr.dzm[:, 1:nzt]
    )

    thl_par_1_down = jnp.concatenate(
        [thl_par_1_down_int, jnp.zeros((ngrdcol, 1))],
        axis=1,
    )
    rt_par_1_down = jnp.concatenate(
        [rt_par_1_down_int, jnp.zeros((ngrdcol, 1))],
        axis=1,
    )
    dCAPE_dz_1_down = jnp.concatenate(
        [dCAPE_dz_1_down_int, jnp.zeros((ngrdcol, 1))],
        axis=1,
    )
    CAPE_incr_1_down = jnp.concatenate(
        [CAPE_incr_1_down_int, jnp.zeros((ngrdcol, 1))],
        axis=1,
    )

    Lscale_up_cols = []
    Lscale_down_cols = []
    for i in range(ngrdcol):
        Lscale_up_cols.append(
            _compute_lscale_up_col(
                tke_i[i],
                thl_par_1_up[i],
                rt_par_1_up[i],
                dCAPE_dz_1_up[i],
                CAPE_incr_1_up[i],
                thl_par_j_precalc_up[i],
                rt_par_j_precalc_up[i],
                exp_mu_dzm[i],
                grav_on_thvm[i],
                Lv_coef[i],
                thv_ds[i],
                exner[i],
                p_in_Pa[i],
                thvm[i],
                gr.dzm[i],
                gr.invrs_dzm[i],
                gr.zt[i],
                gr.k_ub_zt,
                saturation_formula,
                nzt,
            )
        )
        Lscale_down_cols.append(
            _compute_lscale_down_col(
                tke_i[i],
                thl_par_1_down[i],
                rt_par_1_down[i],
                dCAPE_dz_1_down[i],
                CAPE_incr_1_down[i],
                thl_par_j_precalc_down[i],
                rt_par_j_precalc_down[i],
                exp_mu_dzm[i],
                grav_on_thvm[i],
                Lv_coef[i],
                thv_ds[i],
                exner[i],
                p_in_Pa[i],
                thvm[i],
                gr.dzm[i],
                gr.invrs_dzm[i],
                gr.zt[i],
                gr.k_lb_zt,
                gr.k_ub_zt,
                saturation_formula,
                nzt,
            )
        )

    Lscale_up = jnp.stack(Lscale_up_cols)
    Lscale_down = jnp.stack(Lscale_down_cols)

    if l_implemented:
        lminh = (
            jnp.maximum(
                zero_threshold,
                Lscale_sfclyr_depth - (gr.zt - gr.zm[:, gr.k_lb_zm][:, None]),
            )
            * lmin
            * invrs_Lscale_sfclyr_depth
        )
    else:
        lminh = (
            jnp.maximum(zero_threshold, Lscale_sfclyr_depth - gr.zt)
            * lmin
            * invrs_Lscale_sfclyr_depth
        )

    Lscale_up = jnp.maximum(lminh, Lscale_up)
    Lscale_down = jnp.maximum(lminh, Lscale_down)
    Lscale = jnp.sqrt(jnp.maximum(zero_threshold, Lscale_up * Lscale_down))
    Lscale = Lscale.at[:, gr.k_ub_zt].set(Lscale[:, gr.k_ub_zt - gr.grid_dir_indx])
    Lscale = jnp.minimum(Lscale, Lscale_max[:, None])

    if clubb_at_least_debug_level(1):
        err_info = err_info.set_fatal(mask=jnp.any(jnp.isnan(Lscale), axis=1))
        err_info = err_info.set_fatal(mask=jnp.any(jnp.isnan(Lscale_up), axis=1))
        err_info = err_info.set_fatal(mask=jnp.any(jnp.isnan(Lscale_down), axis=1))

    return err_info, Lscale, Lscale_up, Lscale_down


def calc_Lscale_directly(
    ngrdcol: int,
    nzm: int,
    nzt: int,
    gr,
    l_implemented: bool,
    p_in_Pa,
    exner,
    rtm,
    thlm,
    thvm,
    newmu,
    rtp2_zt,
    thlp2_zt,
    rtpthlp_zt,
    pdf_params,
    em,
    thv_ds_zt,
    Lscale_max,
    lmin,
    clubb_params,
    saturation_formula: int,
    l_Lscale_plume_centered: bool,
    stats: JaxStats,
    err_info,
):
    """Diagnose Lscale directly from thermodynamic profiles and PDF data."""
    del rtp2_zt, thlp2_zt, rtpthlp_zt, pdf_params

    l_avg_Lscale = False

    if clubb_at_least_debug_level(0):
        if l_Lscale_plume_centered and not l_avg_Lscale:
            err_info = err_info.set_fatal()
            return (
                err_info,
                jnp.zeros((ngrdcol, nzt)),
                jnp.zeros((ngrdcol, nzt)),
                jnp.zeros((ngrdcol, nzt)),
                stats,
            )

    Lscale_pert_1 = jnp.full((ngrdcol, nzt), unused_var)
    Lscale_pert_2 = jnp.full((ngrdcol, nzt), unused_var)

    stats = stats.update("Lscale_pert_1", Lscale_pert_1)
    stats = stats.update("Lscale_pert_2", Lscale_pert_2)

    # This call to compute_mixing_length must be last. Otherwise, the values of
    # Lscale_up and Lscale_down in stats will be based on perturbation length
    # scales rather than the mean length scale.
    err_info, Lscale, Lscale_up, Lscale_down = compute_mixing_length(
        nzm,
        nzt,
        ngrdcol,
        gr,
        thvm,
        thlm,
        rtm,
        em,
        Lscale_max,
        p_in_Pa,
        exner,
        thv_ds_zt,
        newmu,
        lmin,
        saturation_formula,
        l_implemented,
        err_info,
    )

    return err_info, Lscale, Lscale_up, Lscale_down, stats


def diagnose_Lscale_from_tau(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    upwp_sfc,
    vpwp_sfc,
    ddzt_umvm_sqd,
    ice_supersat_frac,
    em,
    ufmin,
    tau_const,
    sfc_elevation,
    Lscale_max,
    clubb_params,
    stats: JaxStats,
    l_e3sm_config: bool,
    l_smooth_Heaviside_tau_wpxp: bool,
    brunt_vaisala_freq_sqd_smth,
    Ri_zm,
    err_info,
):
    """Diagnose inverse damping time scales and turbulent mixing length."""
    l_smooth_min_max = False
    heaviside_smth_range = 1.0

    em_zt = zm2zt(nzm, nzt, ngrdcol, gr, em, em_min)

    lowest_zm_below_ground = (
        gr.zm[:, gr.k_lb_zm]
        - sfc_elevation
        + clubb_params[:, iz_displace]
        < eps
    )
    err_info = err_info.set_fatal(mask=lowest_zm_below_ground)

    tmp_calc_ngrdcol = (upwp_sfc**2 + vpwp_sfc**2) ** one_fourth

    if l_smooth_min_max:
        ustar = smooth_max(
            ngrdcol,
            ngrdcol,
            tmp_calc_ngrdcol,
            ufmin,
            min_max_smth_mag,
        )
    else:
        ustar = jnp.maximum(tmp_calc_ngrdcol, ufmin)

    invrs_tau_bkgnd = (
        clubb_params[:, iC_invrs_tau_bkgnd][:, None] / tau_const
        + jnp.zeros((ngrdcol, nzm))
    )

    norm_ddzt_umvm = jnp.sqrt(ddzt_umvm_sqd)
    smooth_norm_ddzt_umvm = zm2zt2zm(nzm, nzt, ngrdcol, gr, norm_ddzt_umvm)

    invrs_tau_shear_smooth = (
        clubb_params[:, iC_invrs_tau_shear][:, None] * smooth_norm_ddzt_umvm
    )

    # Enforce that invrs_tau_shear is positive.
    invrs_tau_shear = smooth_max(
        nzm,
        ngrdcol,
        invrs_tau_shear_smooth,
        zero_threshold,
        min_max_smth_mag,
    )

    z_eff = (
        gr.zm
        - sfc_elevation[:, None]
        + clubb_params[:, iz_displace][:, None]
    )
    z_eff = jnp.where(jnp.abs(z_eff) < eps, eps, z_eff)
    invrs_tau_sfc = (
        clubb_params[:, iC_invrs_tau_sfc][:, None]
        * (ustar[:, None] / vonk)
        / z_eff
    )

    invrs_tau_no_N2_zm = invrs_tau_bkgnd + invrs_tau_sfc + invrs_tau_shear

    if l_smooth_min_max:
        brunt_vaisala_freq_clipped = smooth_max(
            nzm,
            ngrdcol,
            zero_threshold,
            brunt_vaisala_freq_sqd_smth,
            1.0e-4 * min_max_smth_mag,
        )
        brunt_freq_pos = jnp.sqrt(brunt_vaisala_freq_clipped)
    else:
        brunt_freq_pos = jnp.sqrt(
            jnp.maximum(zero_threshold, brunt_vaisala_freq_sqd_smth)
        )

    ice_supersat_frac_zm = zt2zm(
        nzm,
        nzt,
        ngrdcol,
        gr,
        ice_supersat_frac,
        zero_threshold,
    )

    if l_smooth_min_max:
        raise NotImplementedError("l_smooth_min_max is a disabled Fortran parameter.")
    else:
        brunt_freq_out_cloud = brunt_freq_pos * jnp.minimum(
            one,
            jnp.maximum(zero_threshold, one - ice_supersat_frac_zm / 0.001),
        )

    brunt_freq_out_cloud = jnp.where(
        gr.zm < clubb_params[:, ialtitude_threshold][:, None],
        zero,
        brunt_freq_out_cloud,
    )

    # Write both bv extra terms to disk.
    stats = stats.update("bv_freq_pos", brunt_freq_pos)
    stats = stats.update("bv_freq_out_cloud", brunt_freq_out_cloud)

    # This time scale is used optionally for the return-to-isotropy term. It
    # omits invrs_tau_sfc based on the rationale that the isotropization
    # rate shouldn't be enhanced near the ground.
    invrs_tau_N2_iso = (
        invrs_tau_bkgnd
        + invrs_tau_shear
        + clubb_params[:, iC_invrs_tau_N2_wp2][:, None] * brunt_freq_pos
    )

    invrs_tau_wp2_zm = (
        invrs_tau_no_N2_zm
        + clubb_params[:, iC_invrs_tau_N2][:, None] * brunt_freq_pos
        + clubb_params[:, iC_invrs_tau_N2_wp2][:, None] * brunt_freq_out_cloud
    )

    invrs_tau_zm = (
        invrs_tau_no_N2_zm
        + clubb_params[:, iC_invrs_tau_N2][:, None] * brunt_freq_pos
    )

    if l_e3sm_config:
        invrs_tau_zm = one_half * invrs_tau_zm
        invrs_tau_xp2_zm = (
            invrs_tau_no_N2_zm
            + clubb_params[:, iC_invrs_tau_N2_xp2][:, None] * brunt_freq_pos
            + clubb_params[:, iC_invrs_tau_sfc][:, None]
            * two
            * jnp.sqrt(em)
            / z_eff
        )
        if l_smooth_min_max:
            raise NotImplementedError("l_smooth_min_max is a disabled Fortran parameter.")
        else:
            invrs_tau_xp2_zm = (
                jnp.minimum(
                    jnp.maximum(
                        jnp.sqrt(
                            ddzt_umvm_sqd
                            / jnp.maximum(1.0e-7, brunt_vaisala_freq_sqd_smth)
                        ),
                        0.3,
                    ),
                    one,
                )
                * invrs_tau_xp2_zm
            )
        invrs_tau_wpxp_zm = (
            two * invrs_tau_zm
            + clubb_params[:, iC_invrs_tau_N2_wpxp][:, None]
            * brunt_freq_out_cloud
        )
    else:
        invrs_tau_xp2_zm = (
            invrs_tau_no_N2_zm
            + clubb_params[:, iC_invrs_tau_N2][:, None] * brunt_freq_pos
            + clubb_params[:, iC_invrs_tau_N2_xp2][:, None] * brunt_freq_out_cloud
        )
        ice_supersat_frac_zm = zt2zm(
            nzm,
            nzt,
            ngrdcol,
            gr,
            ice_supersat_frac,
            zero_threshold,
        )
        invrs_tau_wpxp_zm = (
            invrs_tau_no_N2_zm
            + clubb_params[:, iC_invrs_tau_N2][:, None] * brunt_freq_pos
            + clubb_params[:, iC_invrs_tau_N2_wpxp][:, None]
            * brunt_freq_out_cloud
        )

    if l_smooth_Heaviside_tau_wpxp:
        bvf_thresh = (
            brunt_vaisala_freq_sqd_smth
            / clubb_params[:, iC_invrs_tau_wpxp_N2_thresh][:, None]
            - one
        )
        H_invrs_tau_wpxp_N2 = smooth_heaviside_peskin(
            nzm,
            ngrdcol,
            bvf_thresh,
            heaviside_smth_range,
        )
    else:
        H_invrs_tau_wpxp_N2 = jnp.where(
            brunt_vaisala_freq_sqd_smth
            > clubb_params[:, iC_invrs_tau_wpxp_N2_thresh][:, None],
            one,
            zero,
        )

    if l_smooth_min_max:
        raise NotImplementedError("l_smooth_min_max is a disabled Fortran parameter.")
    else:
        invrs_tau_wpxp_zm = jnp.where(
            gr.zm > clubb_params[:, ialtitude_threshold][:, None],
            invrs_tau_wpxp_zm
            * (
                one
                + H_invrs_tau_wpxp_N2
                * jnp.minimum(
                    clubb_params[:, iC_invrs_tau_wpxp_Ri][:, None]
                    * jnp.maximum(Ri_zm, zero)
                    ** clubb_params[:, iwpxp_Ri_exp][:, None],
                    12.0,
                )
            ),
            invrs_tau_wpxp_zm,
        )

    invrs_tau_wp3_zm = (
        invrs_tau_wp2_zm
        + clubb_params[:, iC_invrs_tau_N2_clear_wp3][:, None]
        * brunt_freq_out_cloud
    )

    if l_smooth_min_max:
        raise NotImplementedError("l_smooth_min_max is a disabled Fortran parameter.")
    else:
        tau_max_zt = Lscale_max[:, None] / jnp.sqrt(em_zt)
        tau_max_zm = Lscale_max[:, None] / jnp.sqrt(jnp.maximum(em, em_min))

    if l_smooth_min_max:
        raise NotImplementedError("l_smooth_min_max is a disabled Fortran parameter.")
    else:
        tau_zm = jnp.minimum(one / invrs_tau_zm, tau_max_zm)
        tau_zt = zm2zt(nzm, nzt, ngrdcol, gr, tau_zm)
        tau_zt = jnp.minimum(tau_zt, tau_max_zt)

    invrs_tau_zt = zm2zt(nzm, nzt, ngrdcol, gr, invrs_tau_zm)
    invrs_tau_wp3_zt = zm2zt(nzm, nzt, ngrdcol, gr, invrs_tau_wp3_zm)

    Lscale = tau_zt * jnp.sqrt(em_zt)

    # Lscale_up and Lscale_down aren't calculated with this option. They are
    # set to 0 for stats output.
    Lscale_up = jnp.zeros((ngrdcol, nzt))
    Lscale_down = jnp.zeros((ngrdcol, nzt))

    return (
        err_info,
        invrs_tau_zt,
        invrs_tau_zm,
        invrs_tau_sfc,
        invrs_tau_no_N2_zm,
        invrs_tau_bkgnd,
        invrs_tau_shear,
        invrs_tau_N2_iso,
        invrs_tau_wp2_zm,
        invrs_tau_xp2_zm,
        invrs_tau_wp3_zm,
        invrs_tau_wp3_zt,
        invrs_tau_wpxp_zm,
        tau_max_zm,
        tau_max_zt,
        tau_zm,
        tau_zt,
        Lscale,
        Lscale_up,
        Lscale_down,
        stats,
    )
