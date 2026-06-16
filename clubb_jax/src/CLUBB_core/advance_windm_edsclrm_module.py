"""JAX-side entry point for ``src/CLUBB_core/advance_windm_edsclrm_module.F90``.

Description:
  Solves for both mean horizontal wind components, um and vm, and for the
  eddy-scalars (passive scalars that do not use the high-order closure).

References:
  Eqn. 8 & 9 on p. 3545 of
  ``A PDF-Based Model for Boundary Layer Clouds. Part I:
    Method and Model Description'' Golaz, et al. (2002)
    JAS, Vol. 59, pp. 3540--3551.

Adaptation notes:
- Sponge damping blocks are unsupported here because clubb_case_initalization
  rejects sponge-enabled Python/JAX driver cases before this routine is called.
- The detailed Fortran error-print diagnostics and diagnostic-only early
  returns are reduced until full JAX diagnostic state is available; fatal
  conditions still mark err_info.
- Stats are threaded explicitly with JaxStats because the Fortran routine uses
  global stats side effects that are not JAX-compatible state.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.advance_helper_module import calc_xpwp, pvertinterp
from clubb_jax.src.CLUBB_core.clip_explicit import clip_covar
from clubb_jax.src.CLUBB_core.clubb_constants import (
    clip_upwp,
    clip_vpwp,
    eps,
    ic_K10,
    ic_K10h,
    l_force_descending_solves,
    one,
    one_half,
    upwp_cl_max,
    vpwp_cl_max,
    zero,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.diffusion import diffusion_zt_lhs
from clubb_jax.src.CLUBB_core.fill_holes import fill_holes_vertical
from clubb_jax.src.CLUBB_core.grid_class import zm2zt
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_solve
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zt_lhs
from clubb_jax.src.derived_types import ErrInfo, Grid, NuVertResDep


windm_edsclrm_um = 1
windm_edsclrm_vm = 2
windm_edsclrm_scalar = 3

ndiags3 = 3


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "edsclr_dim",
        "tridiag_solve_method",
        "l_predict_upwp_vpwp",
        "l_upwind_xm_ma",
        "l_uv_nudge",
        "l_tke_aniso",
        "l_lmm_stepping",
        "l_linearize_pbl_winds",
        "l_do_expldiff_rtm_thlm",
        "fill_holes_type",
        "order_xp2_xpyp",
        "order_wp2_wp3",
        "order_windm",
        "upwp_cl_num",
        "vpwp_cl_num",
        "l_implemented",
    ),
)
def advance_windm_edsclrm(
    nzm: int, nzt: int, ngrdcol: int, edsclr_dim: int, gr: Grid, dt,
    wm_zt, Kh_zm, clubb_params,
    ug, vg, um_ref, vm_ref,
    wp2, up2, vp2, um_forcing, vm_forcing,
    edsclrm_forcing, p_in_Pa,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zt,
    fcor, l_implemented: bool,
    nu_vert_res_dep: NuVertResDep, ts_nudge,
    tridiag_solve_method: int,
    l_predict_upwp_vpwp: bool,
    l_upwind_xm_ma: bool,
    l_uv_nudge: bool,
    l_tke_aniso: bool,
    l_lmm_stepping: bool,
    l_linearize_pbl_winds: bool,
    l_do_expldiff_rtm_thlm: bool,
    fill_holes_type: int,
    order_xp2_xpyp: int,
    order_wp2_wp3: int,
    order_windm: int,
    upwp_cl_num: int,
    vpwp_cl_num: int,
    stats: JaxStats,
    um, vm, thlm, rtm, edsclrm,
    upwp, vpwp, wpedsclrp,
    um_pert, vm_pert, upwp_pert, vpwp_pert,
    err_info: ErrInfo,
):
    """Advance mean winds and eddy scalars one model timestep."""
    del order_xp2_xpyp, order_wp2_wp3, order_windm

    nu_zero = jnp.zeros((ngrdcol,), dtype=jnp.float64)

    Km_zm = Kh_zm * clubb_params[:, ic_K10][:, None]
    Kmh_zm = Kh_zm * clubb_params[:, ic_K10h][:, None]
    Km_zm_p_nu10 = Km_zm + nu_vert_res_dep.nu10[:, None]

    if edsclr_dim > 1 and l_do_expldiff_rtm_thlm:
        edsclrm = edsclrm.at[:, :, edsclr_dim - 2].set(thlm)
        edsclrm = edsclrm.at[:, :, edsclr_dim - 1].set(rtm)

    l_perturbed_wind = (not l_predict_upwp_vpwp) and l_linearize_pbl_winds

    if not l_implemented:
        lhs_ma_zt = term_ma_zt_lhs(
            nzm, nzt, ngrdcol, wm_zt, gr.weights_zt2zm,
            gr.invrs_dzt, gr.invrs_dzm, l_upwind_xm_ma, gr.grid_dir,
        )
    else:
        lhs_ma_zt = jnp.zeros((ndiags3, ngrdcol, nzt), dtype=jnp.float64)

    lhs = jnp.zeros((ndiags3, ngrdcol, nzt), dtype=jnp.float64)
    rhs = jnp.zeros((ngrdcol, nzt, max(2, edsclr_dim)), dtype=jnp.float64)
    solution = jnp.zeros_like(rhs)
    upwp_chnge = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    vpwp_chnge = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    wind_speed = jnp.ones((ngrdcol, nzt), dtype=jnp.float64)
    u_star_sqd = jnp.zeros((ngrdcol,), dtype=jnp.float64)

    if not l_predict_upwp_vpwp:
        Km_zt = zm2zt(nzm, nzt, ngrdcol, gr, Km_zm, zero)

        lhs_diff = diffusion_zt_lhs(
            nzm, nzt, ngrdcol, gr, Km_zm, Km_zt, nu_vert_res_dep.nu10,
            invrs_rho_ds_zt, rho_ds_zm,
        )

        if l_lmm_stepping:
            um_old = um
            vm_old = vm
        else:
            um_old = jnp.zeros_like(um)
            vm_old = jnp.zeros_like(vm)

        um_tndcy, stats = compute_uv_tndcy(
            nzt, ngrdcol, windm_edsclrm_um,
            fcor, vm, vg,
            um_forcing, l_implemented,
            stats,
        )

        vm_tndcy, stats = compute_uv_tndcy(
            nzt, ngrdcol, windm_edsclrm_vm,
            fcor, um, ug,
            vm_forcing, l_implemented,
            stats,
        )

        l_imp_sfc_momentum_flux = True

        wind_speed = jnp.maximum(jnp.sqrt(um ** 2 + vm ** 2), eps)
        u_star_sqd = jnp.sqrt(
            upwp[:, gr.k_lb_zm] ** 2 + vpwp[:, gr.k_lb_zm] ** 2
        )

        rhs_um, stats = windm_edsclrm_rhs(
            nzm, nzt, ngrdcol, gr, windm_edsclrm_um, dt,
            lhs_diff, um, um_tndcy,
            rho_ds_zm, invrs_rho_ds_zt,
            l_imp_sfc_momentum_flux, upwp[:, gr.k_lb_zm],
            stats,
        )
        rhs = rhs.at[:, :, windm_edsclrm_um - 1].set(rhs_um)

        rhs_vm, stats = windm_edsclrm_rhs(
            nzm, nzt, ngrdcol, gr, windm_edsclrm_vm, dt,
            lhs_diff, vm, vm_tndcy,
            rho_ds_zm, invrs_rho_ds_zt,
            l_imp_sfc_momentum_flux, vpwp[:, gr.k_lb_zm],
            stats,
        )
        rhs = rhs.at[:, :, windm_edsclrm_vm - 1].set(rhs_vm)

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, um)
        upwp = upwp.at[:, 1:nzm - 1].set(-one_half * xpwp[:, 1:nzm - 1])

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, vm)
        vpwp = vpwp.at[:, 1:nzm - 1].set(-one_half * xpwp[:, 1:nzm - 1])

        upwp = upwp.at[:, gr.k_ub_zm].set(zero)
        vpwp = vpwp.at[:, gr.k_ub_zm].set(zero)

        lhs = windm_edsclrm_lhs(
            nzm, nzt, ngrdcol, gr, dt,
            lhs_ma_zt, lhs_diff,
            wind_speed, u_star_sqd,
            rho_ds_zm, invrs_rho_ds_zt,
            l_implemented, l_imp_sfc_momentum_flux,
        )

        nrhs = 2
        l_need_rcond = bool(
            stats.l_sample and stats.var_on_stats_list("windm_matrix_condt_num")
        )
        solution_wind, err_info, stats = windm_edsclrm_solve(
            nzt, ngrdcol, gr, nrhs,
            tridiag_solve_method,
            l_implemented,
            stats, l_need_rcond,
            lhs, rhs[:, :, :nrhs], err_info,
        )
        solution = solution.at[:, :, :nrhs].set(solution_wind)

        um = solution[:, :, windm_edsclrm_um - 1]
        vm = solution[:, :, windm_edsclrm_vm - 1]

        if stats.l_sample:
            stats = windm_edsclrm_implicit_stats(
                nzm, nzt, ngrdcol, windm_edsclrm_um, gr,
                um, gr.invrs_dzt,
                lhs_diff, lhs_ma_zt,
                invrs_rho_ds_zt, u_star_sqd,
                rho_ds_zm, wind_speed,
                l_imp_sfc_momentum_flux,
                stats,
            )

            stats = windm_edsclrm_implicit_stats(
                nzm, nzt, ngrdcol, windm_edsclrm_vm, gr,
                vm, gr.invrs_dzt,
                lhs_diff, lhs_ma_zt,
                invrs_rho_ds_zt, u_star_sqd,
                rho_ds_zm, wind_speed,
                l_imp_sfc_momentum_flux,
                stats,
            )

        if l_lmm_stepping:
            um = one_half * (um_old + um)
            vm = one_half * (vm_old + vm)

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, um)
        upwp = upwp.at[:, 1:nzm - 1].add(-one_half * xpwp[:, 1:nzm - 1])

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, vm)
        vpwp = vpwp.at[:, 1:nzm - 1].add(-one_half * xpwp[:, 1:nzm - 1])

        if l_uv_nudge:
            if stats.l_sample:
                stats = stats.begin_budget("um_ndg", um / dt)
                stats = stats.begin_budget("vm_ndg", vm / dt)

            um = um - ((um - um_ref) * (dt / ts_nudge))
            vm = vm - ((vm - vm_ref) * (dt / ts_nudge))

            if stats.l_sample:
                stats = stats.finalize_budget("um_ndg", um / dt)
                stats = stats.finalize_budget("vm_ndg", vm / dt)

        if stats.l_sample:
            stats = stats.update("um_ref", um_ref)
            stats = stats.update("vm_ref", vm_ref)

        if l_tke_aniso:
            if stats.l_sample and l_predict_upwp_vpwp:
                if upwp_cl_num == 0:
                    stats = stats.begin_budget("upwp_cl", upwp / dt)
                else:
                    stats = stats.update_budget("upwp_cl", -upwp / dt)
            upwp_cl_num = upwp_cl_num + 1
            upwp, upwp_chnge = clip_covar(
                nzm, ngrdcol, clip_upwp, wp2, up2, upwp,
            )
            if stats.l_sample and l_predict_upwp_vpwp:
                if upwp_cl_num == upwp_cl_max:
                    stats = stats.finalize_budget("upwp_cl", upwp / dt)
                else:
                    stats = stats.update_budget("upwp_cl", upwp / dt)

            if stats.l_sample and l_predict_upwp_vpwp:
                if vpwp_cl_num == 0:
                    stats = stats.begin_budget("vpwp_cl", vpwp / dt)
                else:
                    stats = stats.update_budget("vpwp_cl", -vpwp / dt)
            vpwp_cl_num = vpwp_cl_num + 1
            vpwp, vpwp_chnge = clip_covar(
                nzm, ngrdcol, clip_vpwp, wp2, vp2, vpwp,
            )
            if stats.l_sample and l_predict_upwp_vpwp:
                if vpwp_cl_num == vpwp_cl_max:
                    stats = stats.finalize_budget("vpwp_cl", vpwp / dt)
                else:
                    stats = stats.update_budget("vpwp_cl", vpwp / dt)
        else:
            if stats.l_sample and l_predict_upwp_vpwp:
                if upwp_cl_num == 0:
                    stats = stats.begin_budget("upwp_cl", upwp / dt)
                else:
                    stats = stats.update_budget("upwp_cl", -upwp / dt)
            upwp_cl_num = upwp_cl_num + 1
            upwp, upwp_chnge = clip_covar(
                nzm, ngrdcol, clip_upwp, wp2, wp2, upwp,
            )
            if stats.l_sample and l_predict_upwp_vpwp:
                if upwp_cl_num == upwp_cl_max:
                    stats = stats.finalize_budget("upwp_cl", upwp / dt)
                else:
                    stats = stats.update_budget("upwp_cl", upwp / dt)

            if stats.l_sample and l_predict_upwp_vpwp:
                if vpwp_cl_num == 0:
                    stats = stats.begin_budget("vpwp_cl", vpwp / dt)
                else:
                    stats = stats.update_budget("vpwp_cl", -vpwp / dt)
            vpwp_cl_num = vpwp_cl_num + 1
            vpwp, vpwp_chnge = clip_covar(
                nzm, ngrdcol, clip_vpwp, wp2, wp2, vpwp,
            )
            if stats.l_sample and l_predict_upwp_vpwp:
                if vpwp_cl_num == vpwp_cl_max:
                    stats = stats.finalize_budget("vpwp_cl", vpwp / dt)
                else:
                    stats = stats.update_budget("vpwp_cl", vpwp / dt)

    if l_perturbed_wind:
        l_imp_sfc_momentum_flux = True

        wind_speed_pert = jnp.maximum(jnp.sqrt(um_pert ** 2 + vm_pert ** 2), eps)
        u_star_sqd_pert = jnp.sqrt(
            upwp_pert[:, gr.k_lb_zm] ** 2 + vpwp_pert[:, gr.k_lb_zm] ** 2
        )

        rhs_um, stats = windm_edsclrm_rhs(
            nzm, nzt, ngrdcol, gr, windm_edsclrm_um, dt,
            lhs_diff, um_pert, um_tndcy,
            rho_ds_zm, invrs_rho_ds_zt,
            l_imp_sfc_momentum_flux, upwp_pert[:, gr.k_lb_zm],
            stats,
        )
        rhs = rhs.at[:, :, windm_edsclrm_um - 1].set(rhs_um)

        rhs_vm, stats = windm_edsclrm_rhs(
            nzm, nzt, ngrdcol, gr, windm_edsclrm_vm, dt,
            lhs_diff, vm_pert, vm_tndcy,
            rho_ds_zm, invrs_rho_ds_zt,
            l_imp_sfc_momentum_flux, vpwp_pert[:, gr.k_lb_zm],
            stats,
        )
        rhs = rhs.at[:, :, windm_edsclrm_vm - 1].set(rhs_vm)

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, um_pert)
        upwp_pert = upwp_pert.at[:, 1:nzm - 1].set(-one_half * xpwp[:, 1:nzm - 1])

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, vm_pert)
        vpwp_pert = vpwp_pert.at[:, 1:nzm - 1].set(-one_half * xpwp[:, 1:nzm - 1])

        upwp_pert = upwp_pert.at[:, gr.k_ub_zm].set(zero)
        vpwp_pert = vpwp_pert.at[:, gr.k_ub_zm].set(zero)

        lhs = windm_edsclrm_lhs(
            nzm, nzt, ngrdcol, gr, dt,
            lhs_ma_zt, lhs_diff,
            wind_speed_pert, u_star_sqd_pert,
            rho_ds_zm, invrs_rho_ds_zt,
            l_implemented, l_imp_sfc_momentum_flux,
        )

        nrhs = 2
        l_need_rcond = bool(
            stats.l_sample and stats.var_on_stats_list("windm_matrix_condt_num")
        )
        solution_wind, err_info, stats = windm_edsclrm_solve(
            nzt, ngrdcol, gr, nrhs,
            tridiag_solve_method,
            l_implemented,
            stats, l_need_rcond,
            lhs, rhs[:, :, :nrhs], err_info,
        )
        solution = solution.at[:, :, :nrhs].set(solution_wind)

        um_pert = solution[:, :, windm_edsclrm_um - 1]
        vm_pert = solution[:, :, windm_edsclrm_vm - 1]

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, um_pert)
        upwp_pert = upwp_pert.at[:, 1:nzm - 1].add(-one_half * xpwp[:, 1:nzm - 1])

        xpwp = calc_xpwp(nzm, nzt, ngrdcol, gr, Km_zm_p_nu10, vm_pert)
        vpwp_pert = vpwp_pert.at[:, 1:nzm - 1].add(-one_half * xpwp[:, 1:nzm - 1])

        if l_tke_aniso:
            upwp_pert, upwp_chnge = clip_covar(
                nzm, ngrdcol, clip_upwp, wp2, up2, upwp_pert,
            )
            vpwp_pert, vpwp_chnge = clip_covar(
                nzm, ngrdcol, clip_vpwp, wp2, vp2, vpwp_pert,
            )
        else:
            upwp_pert, upwp_chnge = clip_covar(
                nzm, ngrdcol, clip_upwp, wp2, wp2, upwp_pert,
            )
            vpwp_pert, vpwp_chnge = clip_covar(
                nzm, ngrdcol, clip_vpwp, wp2, wp2, vpwp_pert,
            )

    if edsclr_dim > 0:
        Kmh_zt = zm2zt(nzm, nzt, ngrdcol, gr, Kmh_zm, zero)

        lhs_diff = diffusion_zt_lhs(
            nzm, nzt, ngrdcol, gr, Kmh_zm, Kmh_zt, nu_zero,
            invrs_rho_ds_zt, rho_ds_zm,
        )

        if l_lmm_stepping:
            edsclrm_old = edsclrm
        else:
            edsclrm_old = jnp.zeros_like(edsclrm)

        l_imp_sfc_momentum_flux = False

        for edsclr in range(edsclr_dim):
            rhs_scalar, stats = windm_edsclrm_rhs(
                nzm, nzt, ngrdcol, gr,
                windm_edsclrm_scalar, dt,
                lhs_diff, edsclrm[:, :, edsclr],
                edsclrm_forcing[:, :, edsclr],
                rho_ds_zm, invrs_rho_ds_zt,
                l_imp_sfc_momentum_flux,
                wpedsclrp[:, gr.k_lb_zm, edsclr],
                stats,
            )
            rhs = rhs.at[:, :, edsclr].set(rhs_scalar)

        for edsclr in range(edsclr_dim):
            xpwp = calc_xpwp(
                nzm, nzt, ngrdcol, gr,
                Km_zm_p_nu10, edsclrm[:, :, edsclr],
            )
            wpedsclrp = wpedsclrp.at[:, 1:nzm - 1, edsclr].set(
                -one_half * xpwp[:, 1:nzm - 1]
            )

        wpedsclrp = wpedsclrp.at[:, gr.k_ub_zm, :edsclr_dim].set(zero)

        lhs = windm_edsclrm_lhs(
            nzm, nzt, ngrdcol, gr, dt,
            lhs_ma_zt, lhs_diff,
            wind_speed, u_star_sqd,
            rho_ds_zm, invrs_rho_ds_zt,
            l_implemented, l_imp_sfc_momentum_flux,
        )

        l_need_rcond = False
        solution_scalar, err_info, stats = windm_edsclrm_solve(
            nzt, ngrdcol, gr, edsclr_dim,
            tridiag_solve_method,
            l_implemented,
            stats, l_need_rcond,
            lhs, rhs[:, :, :edsclr_dim], err_info,
        )
        solution = solution.at[:, :, :edsclr_dim].set(solution_scalar)

        edsclrm = edsclrm.at[:, :, :edsclr_dim].set(solution[:, :, :edsclr_dim])

        if l_lmm_stepping:
            edsclrm = edsclrm.at[:, :, :edsclr_dim].set(
                one_half * (edsclrm_old[:, :, :edsclr_dim] + edsclrm[:, :, :edsclr_dim])
            )

        for edsclr in range(edsclr_dim):
            xpwp = calc_xpwp(
                nzm, nzt, ngrdcol, gr,
                Kmh_zm, edsclrm[:, :, edsclr],
            )
            wpedsclrp = wpedsclrp.at[:, 1:nzm - 1, edsclr].set(
                -one_half * xpwp[:, 1:nzm - 1]
            )

    if edsclr_dim > 1 and l_do_expldiff_rtm_thlm:
        thlm700 = pvertinterp(nzt, ngrdcol, gr, p_in_Pa, 70000.0, thlm)
        thlm1000 = pvertinterp(nzt, ngrdcol, gr, p_in_Pa, 100000.0, thlm)
        apply_explicit_diffusion = (thlm700 - thlm1000) < 20.0

        if stats.l_sample:
            thlm_ed = jnp.where(
                apply_explicit_diffusion[:, None],
                (edsclrm[:, :, edsclr_dim - 2] - thlm) / dt,
                zero,
            )
            rtm_ed = jnp.where(
                apply_explicit_diffusion[:, None],
                (edsclrm[:, :, edsclr_dim - 1] - rtm) / dt,
                zero,
            )
        else:
            thlm_ed = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
            rtm_ed = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

        thlm = jnp.where(
            apply_explicit_diffusion[:, None],
            edsclrm[:, :, edsclr_dim - 2],
            thlm,
        )
        rtm = jnp.where(
            apply_explicit_diffusion[:, None],
            edsclrm[:, :, edsclr_dim - 1],
            rtm,
        )
    else:
        thlm_ed = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        rtm_ed = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

    if stats.l_sample:
        stats = stats.update("thlm_ed", thlm_ed)
        stats = stats.update("rtm_ed", rtm_ed)

    if edsclr_dim > 0 and fill_holes_type != 0:
        for edsclr in range(edsclr_dim):
            edsclrm_filled = fill_holes_vertical(
                nzt, ngrdcol, zero_threshold,
                gr.k_lb_zt, gr.k_ub_zt,
                gr.dzt, rho_ds_zt, gr.grid_dir_indx,
                fill_holes_type,
                edsclrm[:, :, edsclr],
            )
            edsclrm = edsclrm.at[:, :, edsclr].set(edsclrm_filled)

    return (
        upwp_cl_num, vpwp_cl_num,
        um, vm, thlm, rtm, edsclrm, upwp, vpwp, wpedsclrp,
        um_pert, vm_pert, upwp_pert, vpwp_pert, err_info, stats,
    )


@partial(
    jax.jit,
    static_argnames=(
        "nzt",
        "ngrdcol",
        "nrhs",
        "tridiag_solve_method",
        "l_implemented",
        "l_need_rcond",
    ),
)
def windm_edsclrm_solve(
    nzt, ngrdcol, gr, nrhs,
    tridiag_solve_method,
    l_implemented,
    stats, l_need_rcond,
    lhs, rhs, err_info,
):
    """Solve the horizontal wind or eddy-scalar tridiagonal system."""
    lhs_solve = lhs
    rhs_solve = rhs

    if l_force_descending_solves and gr.grid_dir_indx > 0:
        lhs_solve = lhs_solve[::-1, :, ::-1]
        rhs_solve = rhs_solve[:, ::-1, :]

    err_info, solution, rcond = tridiag_solve(
        "windm_edsclrm", tridiag_solve_method, ngrdcol, nzt,
        lhs_solve, rhs_solve, err_info,
        use_rcond=l_need_rcond,
        l_implemented=l_implemented,
    )

    if l_need_rcond and stats.l_sample:
        stats = stats.update("windm_matrix_condt_num", one / rcond)

    if l_force_descending_solves and gr.grid_dir_indx > 0:
        solution = solution[:, ::-1, :]

    return solution, err_info, stats


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "solve_type",
        "l_imp_sfc_momentum_flux",
    ),
)
def windm_edsclrm_implicit_stats(
    nzm, nzt, ngrdcol, solve_type, gr,
    xm, invrs_dzt,
    lhs_diff, lhs_ma_zt,
    invrs_rho_ds_zt, u_star_sqd,
    rho_ds_zm, wind_speed,
    l_imp_sfc_momentum_flux,
    stats,
):
    """Compute implicit contributions to um and vm."""
    del nzm
    if not stats.l_sample:
        return stats

    if solve_type == windm_edsclrm_um:
        name_ma = "um_ma"
        name_ta = "um_ta"
    elif solve_type == windm_edsclrm_vm:
        name_ma = "vm_ma"
        name_ta = "vm_ta"
    else:
        return stats

    imp_sfc_flux = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

    if l_imp_sfc_momentum_flux and stats.var_on_stats_list(name_ta):
        imp_sfc_flux = imp_sfc_flux.at[:, gr.k_lb_zt].set(
            -gr.grid_dir
            * invrs_rho_ds_zt[:, gr.k_lb_zt]
            * invrs_dzt[:, gr.k_lb_zt]
            * rho_ds_zm[:, gr.k_lb_zm]
            * (u_star_sqd / wind_speed[:, gr.k_lb_zt])
        )

    stats_tmp = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if gr.grid_dir_indx > 0:
        stats_tmp = stats_tmp.at[:, 0].set(
            -lhs_ma_zt[1, :, 0] * xm[:, 0]
            -lhs_ma_zt[0, :, 0] * xm[:, 1]
        )
        stats_tmp = stats_tmp.at[:, 1:-1].set(
            -lhs_ma_zt[2, :, 1:-1] * xm[:, :-2]
            -lhs_ma_zt[1, :, 1:-1] * xm[:, 1:-1]
            -lhs_ma_zt[0, :, 1:-1] * xm[:, 2:]
        )
        stats_tmp = stats_tmp.at[:, -1].set(
            -lhs_ma_zt[2, :, -1] * xm[:, -2]
            -lhs_ma_zt[1, :, -1] * xm[:, -1]
        )
    else:
        stats_tmp = stats_tmp.at[:, -1].set(
            -lhs_ma_zt[1, :, -1] * xm[:, -1]
            -lhs_ma_zt[2, :, -1] * xm[:, -2]
        )
        stats_tmp = stats_tmp.at[:, 1:-1].set(
            -lhs_ma_zt[0, :, 1:-1] * xm[:, 2:]
            -lhs_ma_zt[1, :, 1:-1] * xm[:, 1:-1]
            -lhs_ma_zt[2, :, 1:-1] * xm[:, :-2]
        )
        stats_tmp = stats_tmp.at[:, 0].set(
            -lhs_ma_zt[0, :, 0] * xm[:, 1]
            -lhs_ma_zt[1, :, 0] * xm[:, 0]
        )
    stats = stats.update(name_ma, stats_tmp)

    stats_tmp = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if gr.grid_dir_indx > 0:
        stats_tmp = stats_tmp.at[:, 0].set(
            (-one_half * lhs_diff[1, :, 0] + imp_sfc_flux[:, 0]) * xm[:, 0]
            -one_half * lhs_diff[0, :, 0] * xm[:, 1]
        )
        stats_tmp = stats_tmp.at[:, 1:-1].set(
            -one_half * lhs_diff[2, :, 1:-1] * xm[:, :-2]
            + (-one_half * lhs_diff[1, :, 1:-1] + imp_sfc_flux[:, 1:-1]) * xm[:, 1:-1]
            -one_half * lhs_diff[0, :, 1:-1] * xm[:, 2:]
        )
        stats_tmp = stats_tmp.at[:, -1].set(
            -one_half * lhs_diff[2, :, -1] * xm[:, -2]
            + (-one_half * lhs_diff[1, :, -1] + imp_sfc_flux[:, -1]) * xm[:, -1]
        )
    else:
        stats_tmp = stats_tmp.at[:, -1].set(
            (-one_half * lhs_diff[1, :, -1] + imp_sfc_flux[:, -1]) * xm[:, -1]
            -one_half * lhs_diff[2, :, -1] * xm[:, -2]
        )
        stats_tmp = stats_tmp.at[:, 1:-1].set(
            -one_half * lhs_diff[0, :, 1:-1] * xm[:, 2:]
            + (-one_half * lhs_diff[1, :, 1:-1] + imp_sfc_flux[:, 1:-1]) * xm[:, 1:-1]
            -one_half * lhs_diff[2, :, 1:-1] * xm[:, :-2]
        )
        stats_tmp = stats_tmp.at[:, 0].set(
            -one_half * lhs_diff[0, :, 0] * xm[:, 1]
            + (-one_half * lhs_diff[1, :, 0] + imp_sfc_flux[:, 0]) * xm[:, 0]
        )
    return stats.finalize_budget(name_ta, stats_tmp)


@partial(
    jax.jit,
    static_argnames=("nzt", "ngrdcol", "solve_type", "l_implemented"),
)
def compute_uv_tndcy(
    nzt, ngrdcol, solve_type,
    fcor, perp_wind_m, perp_wind_g,
    xm_forcing, l_implemented,
    stats,
):
    """Compute the explicit tendency for the um and vm wind components."""
    if not l_implemented:
        if solve_type == windm_edsclrm_um:
            name_gf = "um_gf"
            name_cf = "um_cf"
            name_f = "um_f"
            xm_gf = -fcor[:, None] * perp_wind_g
            xm_cf = fcor[:, None] * perp_wind_m
        elif solve_type == windm_edsclrm_vm:
            name_gf = "vm_gf"
            name_cf = "vm_cf"
            name_f = "vm_f"
            xm_gf = fcor[:, None] * perp_wind_g
            xm_cf = -fcor[:, None] * perp_wind_m
        else:
            name_gf = ""
            name_cf = ""
            name_f = ""
            xm_gf = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
            xm_cf = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

        xm_tndcy = xm_gf + xm_cf + xm_forcing

        if stats.l_sample:
            stats = stats.update(name_gf, xm_gf)
            stats = stats.update(name_cf, xm_cf)
            stats = stats.update(name_f, xm_forcing)
    else:
        xm_tndcy = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

    return xm_tndcy, stats


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "l_implemented",
        "l_imp_sfc_momentum_flux",
    ),
)
def windm_edsclrm_lhs(
    nzm, nzt, ngrdcol, gr, dt,
    lhs_ma_zt, lhs_diff,
    wind_speed, u_star_sqd,
    rho_ds_zm, invrs_rho_ds_zt,
    l_implemented, l_imp_sfc_momentum_flux,
):
    """Calculate the implicit portion of the wind or eddy-scalar equation."""
    del nzm, ngrdcol
    invrs_dt = one / dt

    lhs = jnp.zeros_like(lhs_diff)
    lhs = lhs.at[0].set(one_half * lhs_diff[0])
    lhs = lhs.at[1].set(one_half * lhs_diff[1] + invrs_dt)
    lhs = lhs.at[2].set(one_half * lhs_diff[2])

    if not l_implemented:
        if gr.grid_dir_indx > 0:
            lhs = lhs.at[:, :, :nzt - 1].add(lhs_ma_zt[:, :, :nzt - 1])
        else:
            lhs = lhs.at[:, :, 1:nzt].add(lhs_ma_zt[:, :, 1:nzt])

    if l_imp_sfc_momentum_flux:
        lhs = lhs.at[1, :, gr.k_lb_zt].add(
            gr.grid_dir
            * invrs_rho_ds_zt[:, gr.k_lb_zt]
            * gr.invrs_dzt[:, gr.k_lb_zt]
            * rho_ds_zm[:, gr.k_lb_zm]
            * (u_star_sqd / wind_speed[:, gr.k_lb_zt])
        )

    return lhs


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "solve_type",
        "l_imp_sfc_momentum_flux",
    ),
)
def windm_edsclrm_rhs(
    nzm, nzt, ngrdcol, gr, solve_type, dt,
    lhs_diff, xm, xm_tndcy,
    rho_ds_zm, invrs_rho_ds_zt,
    l_imp_sfc_momentum_flux, xpwp_sfc,
    stats,
):
    """Calculate explicit RHS contributions for wind or eddy-scalar equations."""
    del nzm
    invrs_dt = one / dt

    if solve_type == windm_edsclrm_um:
        name_ta = "um_ta"
    elif solve_type == windm_edsclrm_vm:
        name_ta = "vm_ta"
    else:
        name_ta = ""

    rhs = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    rhs = rhs.at[:, 0].set(
        one_half
        * (-lhs_diff[1, :, 0] * xm[:, 0] - lhs_diff[0, :, 0] * xm[:, 1])
        + xm_tndcy[:, 0]
        + invrs_dt * xm[:, 0]
    )

    if gr.grid_dir_indx > 0:
        rhs = rhs.at[:, 1:-1].set(
            one_half
            * (
                -lhs_diff[2, :, 1:-1] * xm[:, :-2]
                -lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                -lhs_diff[0, :, 1:-1] * xm[:, 2:]
            )
            + xm_tndcy[:, 1:-1]
            + invrs_dt * xm[:, 1:-1]
        )
    else:
        rhs = rhs.at[:, 1:-1].set(
            one_half
            * (
                -lhs_diff[0, :, 1:-1] * xm[:, 2:]
                -lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                -lhs_diff[2, :, 1:-1] * xm[:, :-2]
            )
            + xm_tndcy[:, 1:-1]
            + invrs_dt * xm[:, 1:-1]
        )

    rhs = rhs.at[:, -1].set(
        one_half
        * (-lhs_diff[2, :, -1] * xm[:, -2] - lhs_diff[1, :, -1] * xm[:, -1])
        + xm_tndcy[:, -1]
        + invrs_dt * xm[:, -1]
    )

    if stats.l_sample and stats.var_on_stats_list(name_ta):
        stats_tmp = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp = stats_tmp.at[:, 0].set(
            one_half
            * (lhs_diff[1, :, 0] * xm[:, 0] + lhs_diff[0, :, 0] * xm[:, 1])
        )

        if gr.grid_dir_indx > 0:
            stats_tmp = stats_tmp.at[:, 1:-1].set(
                one_half
                * (
                    lhs_diff[2, :, 1:-1] * xm[:, :-2]
                    + lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                    + lhs_diff[0, :, 1:-1] * xm[:, 2:]
                )
            )
        else:
            stats_tmp = stats_tmp.at[:, 1:-1].set(
                one_half
                * (
                    lhs_diff[0, :, 1:-1] * xm[:, 2:]
                    + lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                    + lhs_diff[2, :, 1:-1] * xm[:, :-2]
                )
            )

        stats_tmp = stats_tmp.at[:, -1].set(
            one_half
            * (lhs_diff[2, :, -1] * xm[:, -2] + lhs_diff[1, :, -1] * xm[:, -1])
        )
        stats = stats.begin_budget(name_ta, stats_tmp)

    if not l_imp_sfc_momentum_flux:
        sfc_term = (
            gr.grid_dir
            * invrs_rho_ds_zt[:, gr.k_lb_zt]
            * gr.invrs_dzt[:, gr.k_lb_zt]
            * rho_ds_zm[:, gr.k_lb_zm]
            * xpwp_sfc
        )
        rhs = rhs.at[:, gr.k_lb_zt].add(sfc_term)

        if stats.l_sample and stats.var_on_stats_list(name_ta):
            stats_tmp = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
            stats_tmp = stats_tmp.at[:, gr.k_lb_zt].set(sfc_term)
            stats = stats.update_budget(name_ta, stats_tmp)

    return rhs, stats


__all__ = [
    "advance_windm_edsclrm",
    "windm_edsclrm_solve",
    "windm_edsclrm_implicit_stats",
    "compute_uv_tndcy",
    "windm_edsclrm_lhs",
    "windm_edsclrm_rhs",
]
