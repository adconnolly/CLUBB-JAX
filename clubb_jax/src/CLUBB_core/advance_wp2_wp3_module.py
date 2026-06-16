"""JAX-side entry point for ``src/CLUBB_core/advance_wp2_wp3_module.F90``.

Description:
  Advance w'^2 and w'^3 one timestep.

References:
  https://arxiv.org/pdf/1711.03675v1.pdf#nameddest=url:wp2_wp3_eqns

  Eqn. 12 & 18 on p. 3545--3546 of
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

from clubb_jax.src.CLUBB_core.Skx_module import Skx_func
from clubb_jax.src.CLUBB_core.advance_helper_module import calc_wp3_on_wp2
from clubb_jax.src.CLUBB_core.clip_explicit import clip_skewness, clip_variance
from clubb_jax.src.CLUBB_core.diffusion import diffusion_zt_lhs, diffusion_zm_lhs
from clubb_jax.src.CLUBB_core.error_code import clubb_at_least_debug_level
from clubb_jax.src.CLUBB_core.fill_holes import (
    fill_holes_vertical,
    fill_holes_wp2_from_horz_tke,
)
from clubb_jax.src.CLUBB_core.grid_class import (
    T_ABOVE,
    T_BELOW,
    ddzt,
    zm2zt,
    zm2zt2zm,
    zt2zm,
)
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zt_lhs, term_ma_zm_lhs
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import band_solve
from clubb_jax.src.CLUBB_core.clubb_constants import (
    clip_wp2,
    eps,
    five,
    four,
    gamma_over_implicit_ts,
    grav,
    iC1,
    iC1b,
    iC1c,
    iC4,
    iC8,
    iC8b,
    iC11,
    iC11b,
    iC11c,
    iC12,
    iC_uu_buoy,
    iC_uu_shr,
    iC_wp2_pr_dfsn,
    iC_wp3_pr_dfsn,
    iC_wp3_pr_tp,
    iC_wp3_pr_turb,
    iSkw_max_mag,
    ia3_coef_min,
    ic_K1,
    ic_K8,
    iiPDF_ADG1,
    iiPDF_new,
    iiPDF_new_hybrid,
    l_explicit_turbulent_adv_wp3,
    l_force_descending_solves,
    max_mag_correlation_flux,
    one,
    one_half,
    one_third,
    penta_bicgstab,
    three,
    three_halves,
    two,
    two_thirds,
    w_tol,
    w_tol_sqd,
    wp2_max,
    zero,
    zero_threshold,
)
from clubb_jax.src.derived_types import (
    ErrInfo,
    Grid,
    NuVertResDep,
    implicit_coefs_terms,
)


# Set logical to true for Crank-Nicholson diffusion scheme or to false for
# completely implicit diffusion scheme.
l_crank_nich_diff = False

ndiags2 = 2
ndiags3 = 3
ndiags5 = 5


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "iiPDF_type",
        "penta_solve_method",
        "fill_holes_type",
        "l_min_wp2_from_corr_wx",
        "l_upwind_xm_ma",
        "l_tke_aniso",
        "l_standard_term_ta",
        "l_partial_upwind_wp3",
        "l_damp_wp2_using_em",
        "l_use_C11_Richardson",
        "l_damp_wp3_Skw_squared",
        "l_lmm_stepping",
        "l_use_tke_in_wp3_pr_turb_term",
        "l_use_tke_in_wp2_wp3_K_dfsn",
        "l_use_wp3_lim_with_smth_Heaviside",
        "l_wp2_fill_holes_tke",
        "l_ho_nontrad_coriolis",
        "l_implemented",
    ),
)
def advance_wp2_wp3(
    nzm: int, nzt: int, ngrdcol: int, gr: Grid, dt,
    sfc_elevation, fcor_y, sigma_sqd_w, wm_zm,
    wm_zt,
    wpup2, wpvp2, wp2up2, wp2vp2, wp4,
    wpthvp, wp2thvp, wp2up, um, vm, upwp, vpwp,
    em, Kh_zm, Kh_zt, invrs_tau_C4_zm,
    invrs_tau_wp3_zt, invrs_tau_C1_zm,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm,
    invrs_rho_ds_zt, thv_ds_zm,
    thv_ds_zt, mixt_frac, Cx_fnc_Richardson,
    lhs_splat_wp2, lhs_splat_wp3,
    pdf_implicit_coefs_terms: implicit_coefs_terms,
    wprtp, wpthlp, rtp2, thlp2,
    clubb_params, nu_vert_res_dep: NuVertResDep,
    iiPDF_type: int,
    penta_solve_method: int,
    fill_holes_type: int,
    l_min_wp2_from_corr_wx: bool,
    l_upwind_xm_ma: bool,
    l_tke_aniso: bool,
    l_standard_term_ta: bool,
    l_partial_upwind_wp3: bool,
    l_damp_wp2_using_em: bool,
    l_use_C11_Richardson: bool,
    l_damp_wp3_Skw_squared: bool,
    l_lmm_stepping: bool,
    l_use_tke_in_wp3_pr_turb_term: bool,
    l_use_tke_in_wp2_wp3_K_dfsn: bool,
    l_use_wp3_lim_with_smth_Heaviside: bool,
    l_wp2_fill_holes_tke: bool,
    l_ho_nontrad_coriolis: bool,
    l_implemented: bool,
    stats: JaxStats,
    up2, vp2, wp2, wp3, err_info: ErrInfo,
):
    """Advance vertical velocity variance and skewness one model timestep."""
    del mixt_frac

    wp2_zt = zm2zt(nzm, nzt, ngrdcol, gr, wp2, w_tol_sqd)
    wp3_zm = zt2zm(nzm, nzt, ngrdcol, gr, wp3)
    Skw_zt = Skx_func(nzt, ngrdcol, wp2_zt, wp3, w_tol, clubb_params)
    Skw_zm = Skx_func(nzm, ngrdcol, wp2, wp3_zm, w_tol, clubb_params)
    wp3_on_wp2, wp3_on_wp2_zt = calc_wp3_on_wp2(
        nzm, nzt, ngrdcol, gr, wp2, wp3,
    )
    del wp3_on_wp2_zt

    # Compute the a3 coefficient (formula 25 in `Equations for CLUBB')
    a3_coef = -two * (one - sigma_sqd_w) ** 2 + three
    a3_coef = jnp.maximum(a3_coef, clubb_params[:, ia3_coef_min][:, None])
    a3_coef_zt = zm2zt(nzm, nzt, ngrdcol, gr, a3_coef)

    if stats.l_sample:
        stats = stats.update("a3_coef_zt", a3_coef_zt)
        stats = stats.update("a3_coef", a3_coef)

    if l_crank_nich_diff and l_use_tke_in_wp2_wp3_K_dfsn:
        err_info = err_info.set_fatal()
        return up2, vp2, wp2, wp3, err_info, stats

    # Vince Larson added code to make C11 function of Skw. 13 Mar 2005
    if l_use_C11_Richardson:
        C11_Skw_fnc = zm2zt(
            nzm, nzt, ngrdcol, gr, Cx_fnc_Richardson, zero_threshold,
        )
    else:
        C11 = clubb_params[:, iC11]
        C11b = clubb_params[:, iC11b]
        C11c = clubb_params[:, iC11c]
        C11c_safe = jnp.where(jnp.abs(C11c) > zero, C11c, one)
        C11_varying = jnp.abs(C11 - C11b) > jnp.abs(C11 + C11b) * eps / two
        C11_Skw_fnc = jnp.where(
            C11_varying[:, None],
            C11b[:, None]
            + (C11 - C11b)[:, None]
            * jnp.exp(-one_half * (Skw_zt / C11c_safe[:, None]) ** 2),
            C11b[:, None],
        )

    C1 = clubb_params[:, iC1]
    C1b = clubb_params[:, iC1b]
    C1c = clubb_params[:, iC1c]
    C1c_safe = jnp.where(jnp.abs(C1c) > zero, C1c, one)
    C1_varying = jnp.abs(C1 - C1b) > jnp.abs(C1 + C1b) * eps / two
    C1_Skw_fnc = jnp.where(
        C1_varying[:, None],
        C1b[:, None]
        + (C1 - C1b)[:, None]
        * jnp.exp(-one_half * (Skw_zm / C1c_safe[:, None]) ** 2),
        C1b[:, None],
    )

    if l_damp_wp2_using_em:
        C1_Skw_fnc = one_third * C1_Skw_fnc

    # Set C16_fnc based on Richardson_num
    C16_fnc = Cx_fnc_Richardson[:, :nzt]

    if clubb_at_least_debug_level(0):
        C11_bad = jnp.any((C11_Skw_fnc > one) | (C11_Skw_fnc < zero), axis=1)
        C16_bad = jnp.any((C16_fnc > one) | (C16_fnc < zero), axis=1)
        err_info = err_info.set_fatal(mask=C11_bad | C16_bad)

    if stats.l_sample:
        stats = stats.update("C11_Skw_fnc", C11_Skw_fnc)
        stats = stats.update("C1_Skw_fnc", C1_Skw_fnc)

    # Define the Coefficent of Eddy Diffusivity for the wp2 and wp3.
    Kw1 = clubb_params[:, ic_K1][:, None] * Kh_zt
    Kw8 = clubb_params[:, ic_K8][:, None] * Kh_zm

    coef_wp4_implicit = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    a1_coef = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    a1_coef_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

    if not l_explicit_turbulent_adv_wp3:
        if iiPDF_type == iiPDF_new or iiPDF_type == iiPDF_new_hybrid:
            coef_wp4_implicit_zt = pdf_implicit_coefs_terms.coef_wp4_implicit
            coef_wp4_implicit = zt2zm(
                nzm, nzt, ngrdcol, gr, coef_wp4_implicit_zt, zero_threshold,
            )
            coef_wp4_implicit = coef_wp4_implicit.at[:, gr.k_lb_zm].set(zero)
            coef_wp4_implicit = coef_wp4_implicit.at[:, gr.k_ub_zm].set(zero)
            if stats.l_sample:
                stats = stats.update("coef_wp4_implicit", coef_wp4_implicit)
        elif iiPDF_type == iiPDF_ADG1:
            a1_coef = one / (one - sigma_sqd_w)
            a1_coef_zt = zm2zt(
                nzm, nzt, ngrdcol, gr, a1_coef, zero_threshold,
            )

    # Not using pressure term, set to 0
    rhs_pr3_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)

    # Initiaize some terms to zero
    wp3_term_ta_lhs_result = jnp.zeros((ndiags5, ngrdcol, nzt), dtype=jnp.float64)
    wp3_pr3_lhs = jnp.zeros((ndiags5, ngrdcol, nzt), dtype=jnp.float64)

    Kw1_zm = zt2zm(nzm, nzt, ngrdcol, gr, Kw1, zero)
    Kw8_zt = zm2zt(nzm, nzt, ngrdcol, gr, Kw8, zero)

    # Experimental bouyancy term
    # Experimental term from CLUBB TRAC ticket #411
    if not l_use_tke_in_wp3_pr_turb_term:
        dum_dz = ddzt(nzm, nzt, ngrdcol, gr, um)
        dvm_dz = ddzt(nzm, nzt, ngrdcol, gr, vm)
    else:
        dum_dz = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        dvm_dz = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)

    em_smth = zm2zt2zm(nzm, nzt, ngrdcol, gr, em)
    wp2_smth = zm2zt2zm(nzm, nzt, ngrdcol, gr, wp2)

    rhs_pr_turb_wp3 = wp3_term_pr_turb_rhs(
        nzm, nzt, ngrdcol, gr, clubb_params[:, iC_wp3_pr_turb],
        Kh_zt, wpthvp,
        dum_dz, dvm_dz,
        upwp, vpwp,
        thv_ds_zt,
        rho_ds_zm, invrs_rho_ds_zt,
        em_smth, wp2_smth,
        l_use_tke_in_wp3_pr_turb_term,
    )

    rhs_pr_dfsn_wp3 = wp3_term_pr_dfsn_rhs(
        nzm, nzt, ngrdcol, gr,
        clubb_params[:, iC_wp3_pr_dfsn],
        rho_ds_zm, invrs_rho_ds_zt,
        wp2up2, wp2vp2, wp4,
        up2, vp2, wp2,
    )

    rhs_pr_dfsn_wp2 = wp2_term_pr_dfsn_rhs(
        nzm, nzt, ngrdcol, gr, clubb_params[:, iC_wp2_pr_dfsn],
        rho_ds_zt, invrs_rho_ds_zm,
        wpup2, wpvp2, wp3,
    )

    # This part handles the wp2 equation terms.
    lhs_diff_zm = diffusion_zm_lhs(
        nzm, nzt, ngrdcol, gr, Kw1, Kw1_zm, nu_vert_res_dep.nu1,
        invrs_rho_ds_zm, rho_ds_zt,
    )

    # This part handles the wp3 equation terms.
    lhs_diff_zt = diffusion_zt_lhs(
        nzm, nzt, ngrdcol, gr, Kw8, Kw8_zt, nu_vert_res_dep.nu8,
        invrs_rho_ds_zt, rho_ds_zm,
    )

    lhs_diff_zm_crank = jnp.zeros_like(lhs_diff_zm)
    lhs_diff_zt_crank = jnp.zeros_like(lhs_diff_zt)
    if l_crank_nich_diff:
        lhs_diff_zm_crank = lhs_diff_zm_crank.at[:, :, 1:-1].set(
            one_half * lhs_diff_zm[:, :, 1:-1]
        )
        lhs_diff_zt_crank = lhs_diff_zt_crank.at[:, :, 1:-1].set(
            one_half
            * lhs_diff_zt[:, :, 1:-1]
            * clubb_params[:, iC12][None, :, None]
        )

    if l_tke_aniso:
        rhs_pr1_wp2 = wp2_term_pr1_rhs(
            nzm, ngrdcol, gr, clubb_params[:, iC4],
            up2, vp2, invrs_tau_C4_zm,
        )

        lhs_pr1_wp2 = wp2_term_pr1_lhs(
            nzm, ngrdcol, gr,
            clubb_params[:, iC4], invrs_tau_C4_zm,
        )
    else:
        rhs_pr1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        lhs_pr1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)

    C_wp3_pr_tp = jnp.ones((ngrdcol,), dtype=jnp.float64)
    lhs_adv_tp_wp3 = wp3_term_tp_lhs(
        nzm, nzt, ngrdcol, gr, C_wp3_pr_tp,
        wp2, rho_ds_zm, invrs_rho_ds_zt,
    )

    C_wp3_pr_tp = -clubb_params[:, iC_wp3_pr_tp]
    lhs_pr_tp_wp3 = wp3_term_tp_lhs(
        nzm, nzt, ngrdcol, gr, C_wp3_pr_tp,
        wp2, rho_ds_zm, invrs_rho_ds_zt,
    )

    lhs_tp_wp3 = lhs_adv_tp_wp3 + lhs_pr_tp_wp3

    lhs_pr1_wp3 = wp3_term_pr1_lhs(
        nzt, ngrdcol, gr,
        clubb_params[:, iC8], clubb_params[:, iC8b],
        invrs_tau_wp3_zt, Skw_zt,
        l_damp_wp3_Skw_squared,
    )

    lhs_dp1_wp2 = wp2_term_dp1_lhs(
        nzm, ngrdcol, gr,
        C1_Skw_fnc, invrs_tau_C1_zm,
    )

    rhs_bp_pr2_wp2 = wp2_terms_bp_pr2_rhs(
        nzm, ngrdcol, gr,
        clubb_params[:, iC_uu_buoy],
        thv_ds_zm, wpthvp,
    )

    rhs_pr3_wp2 = wp2_term_pr3_rhs(
        nzm, nzt, ngrdcol, gr,
        clubb_params[:, iC_uu_shr],
        clubb_params[:, iC_uu_buoy],
        thv_ds_zm, wpthvp, upwp,
        um, vpwp, vm,
    )

    rhs_dp1_wp2 = wp2_term_dp1_rhs(
        nzm, ngrdcol, gr, C1_Skw_fnc,
        invrs_tau_C1_zm, w_tol_sqd, up2, vp2,
        l_damp_wp2_using_em,
    )

    rhs_bp1_pr2_wp3 = wp3_terms_bp1_pr2_rhs(
        nzt, ngrdcol, gr, C11_Skw_fnc,
        thv_ds_zt, wp2thvp,
    )

    rhs_pr1_wp3 = wp3_term_pr1_rhs(
        nzt, ngrdcol, gr,
        clubb_params[:, iC8], clubb_params[:, iC8b],
        invrs_tau_wp3_zt, Skw_zt, wp3,
        l_damp_wp3_Skw_squared,
    )

    lhs_ta_wp3 = jnp.zeros((ndiags2, ngrdcol, nzt), dtype=jnp.float64)
    rhs_ta_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if l_explicit_turbulent_adv_wp3:
        rhs_ta_wp3 = wp3_term_ta_explicit_rhs(
            nzm, nzt, ngrdcol, gr,
            wp4, rho_ds_zm, invrs_rho_ds_zt,
        )
    else:
        if iiPDF_type == iiPDF_ADG1:
            wp3_term_ta_lhs_result = wp3_term_ta_ADG1_lhs(
                nzm, nzt, ngrdcol, gr,
                wp2, a1_coef, a1_coef_zt,
                a3_coef, a3_coef_zt,
                wp3_on_wp2, rho_ds_zm,
                rho_ds_zt, invrs_rho_ds_zt,
                l_standard_term_ta,
                l_partial_upwind_wp3,
            )
        elif iiPDF_type == iiPDF_new or iiPDF_type == iiPDF_new_hybrid:
            lhs_ta_wp3 = wp3_term_ta_new_pdf_lhs(
                nzm, nzt, ngrdcol, gr, coef_wp4_implicit,
                wp2, rho_ds_zm, invrs_rho_ds_zt,
            )
            wp3_term_ta_lhs_result = wp3_term_ta_lhs_result.at[1, :, :].set(
                lhs_ta_wp3[0, :, :]
            )
            wp3_term_ta_lhs_result = wp3_term_ta_lhs_result.at[3, :, :].set(
                lhs_ta_wp3[1, :, :]
            )

    rhs, stats = wp23_rhs(
        nzm, nzt, ngrdcol, gr, dt, fcor_y,
        wp3_term_ta_lhs_result,
        lhs_diff_zm, lhs_diff_zt, lhs_diff_zm_crank, lhs_diff_zt_crank,
        lhs_tp_wp3, lhs_adv_tp_wp3, lhs_pr_tp_wp3,
        lhs_ta_wp3, lhs_dp1_wp2, rhs_dp1_wp2, lhs_pr1_wp2,
        rhs_pr1_wp2, lhs_pr1_wp3, rhs_pr1_wp3, rhs_bp_pr2_wp2,
        rhs_pr_dfsn_wp2, rhs_bp1_pr2_wp3, rhs_pr3_wp2, rhs_pr3_wp3,
        rhs_ta_wp3, rhs_pr_turb_wp3, rhs_pr_dfsn_wp3,
        wp2, wp3, wpup2, wpvp2,
        wpthvp, wp2thvp, wp2up, up2, vp2, upwp,
        C11_Skw_fnc, thv_ds_zm, thv_ds_zt,
        lhs_splat_wp2, lhs_splat_wp3,
        clubb_params,
        iiPDF_type,
        l_tke_aniso,
        l_use_tke_in_wp2_wp3_K_dfsn,
        l_ho_nontrad_coriolis,
        stats,
    )

    lhs_ma_zm = term_ma_zm_lhs(
        nzm, nzt, ngrdcol, wm_zm,
        gr.invrs_dzm, gr.weights_zm2zt,
    )

    lhs_ma_zt = term_ma_zt_lhs(
        nzm, nzt, ngrdcol, wm_zt, gr.weights_zt2zm,
        gr.invrs_dzt, gr.invrs_dzm,
        l_upwind_xm_ma, gr.grid_dir,
    )

    lhs_diff_zt = lhs_diff_zt * clubb_params[:, iC12][None, :, None]

    if l_crank_nich_diff:
        lhs_diff_zm = lhs_diff_zm.at[:, :, 1:-1].set(
            one_half * lhs_diff_zm[:, :, 1:-1]
        )
        lhs_diff_zt = lhs_diff_zt.at[:, :, 1:-1].set(
            one_half * lhs_diff_zt[:, :, 1:-1]
        )

    lhs_ta_wp2 = wp2_term_ta_lhs(
        nzm, nzt, ngrdcol, gr,
        rho_ds_zt, invrs_rho_ds_zm,
    )

    lhs_ac_pr2_wp2 = wp2_terms_ac_pr2_lhs(
        nzm, nzt, ngrdcol, gr,
        clubb_params[:, iC_uu_shr], wm_zt,
    )

    lhs_ac_pr2_wp3 = wp3_terms_ac_pr2_lhs(
        nzm, nzt, ngrdcol, gr, C11_Skw_fnc, wm_zm,
    )

    lhs = wp23_lhs(
        nzm, nzt, ngrdcol, dt,
        wp3_term_ta_lhs_result,
        lhs_diff_zm, lhs_diff_zt, lhs_ma_zm,
        lhs_ma_zt, lhs_ta_wp2,
        lhs_tp_wp3,
        lhs_ac_pr2_wp2, lhs_ac_pr2_wp3, lhs_dp1_wp2,
        lhs_pr1_wp3, lhs_pr1_wp2, lhs_splat_wp2, lhs_splat_wp3,
        l_tke_aniso,
    )

    wp2_old = wp2
    wp3_old = wp3

    up2, vp2, wp2, wp3, wp3_zm, wp2_zt, err_info, stats = wp23_solve(
        nzm, nzt, ngrdcol, gr, dt, lhs, rhs,
        lhs_ma_zm, lhs_dp1_wp2, lhs_diff_zm,
        lhs_ta_wp2, lhs_pr1_wp2, lhs_pr1_wp3,
        lhs_diff_zt, lhs_adv_tp_wp3, lhs_pr_tp_wp3,
        wp3_pr3_lhs, lhs_ma_zt,
        wp3_term_ta_lhs_result,
        wm_zm, wm_zt,
        sfc_elevation, C11_Skw_fnc,
        rho_ds_zm,
        upwp, vpwp, wprtp, wpthlp, rtp2, thlp2,
        clubb_params,
        penta_solve_method,
        fill_holes_type,
        l_min_wp2_from_corr_wx,
        l_tke_aniso,
        l_use_tke_in_wp2_wp3_K_dfsn,
        l_use_wp3_lim_with_smth_Heaviside,
        l_wp2_fill_holes_tke,
        l_implemented,
        stats,
        up2, vp2, wp2, wp3, wp3_zm, wp2_zt, err_info,
    )

    if l_lmm_stepping:
        wp2 = one_half * (wp2_old + wp2)
        wp3 = one_half * (wp3_old + wp3)

    if stats.l_sample:
        stats = stats.update("wp2_zt", wp2_zt)
        stats = stats.update("wp3_zm", wp3_zm)

    return up2, vp2, wp2, wp3, err_info, stats


def wp23_solve(
    nzm, nzt, ngrdcol, gr, dt, lhs, rhs,
    lhs_ma_zm, lhs_dp1_wp2, lhs_diff_zm,
    lhs_ta_wp2, lhs_pr1_wp2, lhs_pr1_wp3,
    lhs_diff_zt, lhs_adv_tp_wp3, lhs_pr_tp_wp3,
    wp3_pr3_lhs, lhs_ma_zt,
    wp3_term_ta_lhs_result,
    wm_zm, wm_zt,
    sfc_elevation, C11_Skw_fnc,
    rho_ds_zm,
    upwp, vpwp, wprtp, wpthlp, rtp2, thlp2,
    clubb_params,
    penta_solve_method,
    fill_holes_type,
    l_min_wp2_from_corr_wx,
    l_tke_aniso,
    l_use_tke_in_wp2_wp3_K_dfsn,
    l_use_wp3_lim_with_smth_Heaviside,
    l_wp2_fill_holes_tke,
    l_implemented,
    stats,
    up2, vp2, wp2, wp3, wp3_zm, wp2_zt, err_info,
):
    """Decompose, and back substitute the matrix for wp2/wp3."""
    rhs_save = rhs

    old_solut = jnp.zeros((ngrdcol, 2 * nzm - 1), dtype=jnp.float64)
    if penta_solve_method == penta_bicgstab:
        old_solut = old_solut.at[:, 0::2].set(wp2)
        old_solut = old_solut.at[:, 1::2].set(wp3)

    lhs_solve = lhs
    rhs_solve = rhs
    old_solut_solve = old_solut
    if l_force_descending_solves and gr.grid_dir_indx > 0:
        lhs_solve = lhs_solve[::-1, :, ::-1]
        rhs_solve = rhs_solve[:, ::-1]
        old_solut_solve = old_solut_solve[:, ::-1]

    l_need_rcond = bool(
        stats.l_sample and stats.var_on_stats_list("wp23_matrix_condt_num")
    )

    err_info, solut, rcond = band_solve(
        "wp2_wp3", penta_solve_method,
        ngrdcol, 2, 2, 2 * nzm - 1, 1,
        l_implemented,
        lhs_solve, rhs_solve, err_info,
        old_soln=old_solut_solve,
        use_rcond=l_need_rcond,
    )

    if l_need_rcond and stats.l_sample:
        stats = stats.update("wp23_matrix_condt_num", one / rcond)

    if l_force_descending_solves and gr.grid_dir_indx > 0:
        solut = solut[:, ::-1]

    # Copy result into output arrays and clip
    wp2 = solut[:, 0::2]
    wp3 = solut[:, 1::2]

    if stats.l_sample:
        C_uu_shr_zeros = jnp.zeros((ngrdcol,), dtype=jnp.float64)
        C_uu_shr_plus_one = clubb_params[:, iC_uu_shr] + one
        C11_Skw_fnc_zeros = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        C11_Skw_fnc_plus_one = C11_Skw_fnc + one

        lhs_wp2_ac_term = wp2_terms_ac_pr2_lhs(
            nzm, nzt, ngrdcol, gr, C_uu_shr_zeros, wm_zt,
        )
        lhs_wp2_pr2_term = wp2_terms_ac_pr2_lhs(
            nzm, nzt, ngrdcol, gr, C_uu_shr_plus_one, wm_zt,
        )
        lhs_wp3_ac_term = wp3_terms_ac_pr2_lhs(
            nzm, nzt, ngrdcol, gr, C11_Skw_fnc_zeros, wm_zm,
        )
        lhs_wp3_pr2_term = wp3_terms_ac_pr2_lhs(
            nzm, nzt, ngrdcol, gr, C11_Skw_fnc_plus_one, wm_zm,
        )

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
            -gamma_over_implicit_ts * lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1]
        )
        stats = stats.finalize_budget("wp2_dp1", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        if gr.grid_dir_indx > 0:
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                -lhs_diff_zm[2, :, 1:-1] * wp2[:, :-2]
                -lhs_diff_zm[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_diff_zm[0, :, 1:-1] * wp2[:, 2:]
            )
        else:
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                -lhs_diff_zm[0, :, 1:-1] * wp2[:, 2:]
                -lhs_diff_zm[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_diff_zm[2, :, 1:-1] * wp2[:, :-2]
            )
        if l_crank_nich_diff or l_use_tke_in_wp2_wp3_K_dfsn:
            stats = stats.finalize_budget("wp2_dp2", stats_tmp_zm)
        else:
            stats = stats.update("wp2_dp2", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
            -lhs_ta_wp2[1, :, 1:-1] * wp3[:, :-1]
            -lhs_ta_wp2[0, :, 1:-1] * wp3[:, 1:]
        )
        stats = stats.update("wp2_ta", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        if gr.grid_dir_indx > 0:
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                -lhs_ma_zm[2, :, 1:-1] * wp2[:, :-2]
                -lhs_ma_zm[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_ma_zm[0, :, 1:-1] * wp2[:, 2:]
            )
        else:
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                -lhs_ma_zm[0, :, 1:-1] * wp2[:, 2:]
                -lhs_ma_zm[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_ma_zm[2, :, 1:-1] * wp2[:, :-2]
            )
        stats = stats.update("wp2_ma", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
            -lhs_wp2_ac_term[:, 1:-1] * wp2[:, 1:-1]
        )
        stats = stats.update("wp2_ac", stats_tmp_zm)

        if l_tke_aniso:
            stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                -gamma_over_implicit_ts * lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1]
            )
            stats = stats.finalize_budget("wp2_pr1", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
            -lhs_wp2_pr2_term[:, 1:-1] * wp2[:, 1:-1]
        )
        stats = stats.finalize_budget("wp2_pr2", stats_tmp_zm)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -gamma_over_implicit_ts * lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1]
        )
        stats = stats.finalize_budget("wp3_pr1", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        if gr.grid_dir_indx > 0:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -lhs_diff_zt[2, :, 1:-1] * wp3[:, :-2]
                -lhs_diff_zt[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_diff_zt[0, :, 1:-1] * wp3[:, 2:]
            )
        else:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -lhs_diff_zt[0, :, 1:-1] * wp3[:, 2:]
                -lhs_diff_zt[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_diff_zt[2, :, 1:-1] * wp3[:, :-2]
            )
        if l_crank_nich_diff or l_use_tke_in_wp2_wp3_K_dfsn:
            stats = stats.finalize_budget("wp3_dp1", stats_tmp_zt)
        else:
            stats = stats.update("wp3_dp1", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -gamma_over_implicit_ts * wp3_term_ta_lhs_result[4, :, 1:-1] * wp3[:, :-2]
            -gamma_over_implicit_ts * wp3_term_ta_lhs_result[3, :, 1:-1] * wp2[:, 1:-2]
            -gamma_over_implicit_ts * wp3_term_ta_lhs_result[2, :, 1:-1] * wp3[:, 1:-1]
            -gamma_over_implicit_ts * wp3_term_ta_lhs_result[1, :, 1:-1] * wp2[:, 2:-1]
            -gamma_over_implicit_ts * wp3_term_ta_lhs_result[0, :, 1:-1] * wp3[:, 2:]
        )
        stats = stats.finalize_budget("wp3_ta", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -gamma_over_implicit_ts * lhs_adv_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]
            -gamma_over_implicit_ts * lhs_adv_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
        )
        stats = stats.finalize_budget("wp3_tp", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -gamma_over_implicit_ts * lhs_pr_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]
            -gamma_over_implicit_ts * lhs_pr_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
        )
        stats = stats.finalize_budget("wp3_pr_tp", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -wp3_pr3_lhs[4, :, 1:-1] * wp3[:, :-2]
            -wp3_pr3_lhs[3, :, 1:-1] * wp2[:, 1:-2]
            -wp3_pr3_lhs[2, :, 1:-1] * wp3[:, 1:-1]
            -wp3_pr3_lhs[1, :, 1:-1] * wp2[:, 2:-1]
            -wp3_pr3_lhs[0, :, 1:-1] * wp3[:, 2:]
        )
        stats = stats.finalize_budget("wp3_pr3", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        if gr.grid_dir_indx > 0:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -lhs_ma_zt[2, :, 1:-1] * wp3[:, :-2]
                -lhs_ma_zt[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_ma_zt[0, :, 1:-1] * wp3[:, 2:]
            )
        else:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -lhs_ma_zt[0, :, 1:-1] * wp3[:, 2:]
                -lhs_ma_zt[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_ma_zt[2, :, 1:-1] * wp3[:, :-2]
            )
        stats = stats.update("wp3_ma", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -lhs_wp3_ac_term[:, 1:-1] * wp3[:, 1:-1]
        )
        stats = stats.update("wp3_ac", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -lhs_wp3_pr2_term[:, 1:-1] * wp3[:, 1:-1]
        )
        stats = stats.finalize_budget("wp3_pr2", stats_tmp_zt)

    if stats.l_sample:
        stats = stats.begin_budget("wp2_pd", wp2 / dt)
        stats = stats.begin_budget("up2_pd", up2 / dt)
        stats = stats.begin_budget("vp2_pd", vp2 / dt)

    if fill_holes_type != 0:
        wp2 = fill_holes_vertical(
            nzm, ngrdcol, w_tol_sqd,
            gr.k_lb_zm + gr.grid_dir_indx,
            gr.k_ub_zm - gr.grid_dir_indx,
            gr.dzm, rho_ds_zm, gr.grid_dir_indx,
            fill_holes_type,
            wp2,
        )

        if l_wp2_fill_holes_tke:
            wp2, up2, vp2 = fill_holes_wp2_from_horz_tke(
                nzm, ngrdcol, w_tol_sqd, 1, nzm - 2,
                wp2, up2, vp2,
            )

    if stats.l_sample:
        stats = stats.finalize_budget("wp2_pd", wp2 / dt)
        stats = stats.finalize_budget("up2_pd", up2 / dt, l_count_sample=False)
        stats = stats.finalize_budget("vp2_pd", vp2 / dt, l_count_sample=False)

    if l_min_wp2_from_corr_wx:
        corr_max_sqd = max_mag_correlation_flux ** 2
        wp2_min_array = jnp.maximum(
            w_tol_sqd,
            wprtp ** 2 / (rtp2 * corr_max_sqd),
        )
        wp2_min_array = jnp.maximum(
            wp2_min_array,
            wpthlp ** 2 / (thlp2 * corr_max_sqd),
        )
        wp2_min_array = jnp.maximum(
            wp2_min_array,
            upwp ** 2 / (up2 * corr_max_sqd),
        )
        wp2_min_array = jnp.maximum(
            wp2_min_array,
            vpwp ** 2 / (vp2 * corr_max_sqd),
        )
        wp2_min_array = jnp.minimum(one, wp2_min_array)
    else:
        wp2_min_array = jnp.full((ngrdcol, nzm), w_tol_sqd, dtype=jnp.float64)

    wp2, stats = clip_variance(
        nzm, ngrdcol, gr, clip_wp2, dt, wp2_min_array,
        stats,
        wp2,
        wp2_max,
    )

    wp2_zt = zm2zt(nzm, nzt, ngrdcol, gr, wp2, w_tol_sqd)

    wp3, stats = clip_skewness(
        nzt, ngrdcol, gr, dt, sfc_elevation,
        clubb_params[:, iSkw_max_mag], wp2_zt,
        l_use_wp3_lim_with_smth_Heaviside,
        stats,
        wp3,
    )

    wp3_zm = zt2zm(nzm, nzt, ngrdcol, gr, wp3)

    del rhs_save
    return up2, vp2, wp2, wp3, wp3_zm, wp2_zt, err_info, stats


def wp23_lhs(
    nzm, nzt, ngrdcol, dt,
    wp3_term_ta_lhs_result,
    lhs_diff_zm, lhs_diff_zt, lhs_ma_zm,
    lhs_ma_zt, lhs_ta_wp2,
    lhs_tp_wp3,
    lhs_ac_pr2_wp2, lhs_ac_pr2_wp3, lhs_dp1_wp2,
    lhs_pr1_wp3, lhs_pr1_wp2, lhs_splat_wp2, lhs_splat_wp3,
    l_tke_aniso,
):
    """Compute LHS band diagonal matrix for w'^2 and w'^3."""
    del nzt
    invrs_dt = one / dt
    lhs = jnp.zeros((ndiags5, ngrdcol, 2 * nzm - 1), dtype=jnp.float64)

    lhs = lhs.at[2, :, 0].set(one)
    lhs = lhs.at[2, :, 1].set(one)

    lhs = lhs.at[0, :, 2:-1:2].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1])
    lhs = lhs.at[1, :, 2:-1:2].set(lhs_ta_wp2[0, :, 1:-1])
    lhs = lhs.at[2, :, 2:-1:2].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1]
        + lhs_ac_pr2_wp2[:, 1:-1]
        + gamma_over_implicit_ts * lhs_dp1_wp2[:, 1:-1]
        + invrs_dt
    )
    lhs = lhs.at[3, :, 2:-1:2].set(lhs_ta_wp2[1, :, 1:-1])
    lhs = lhs.at[4, :, 2:-1:2].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1])

    lhs = lhs.at[0, :, 3:-2:2].set(lhs_ma_zt[0, :, 1:-1] + lhs_diff_zt[0, :, 1:-1])
    lhs = lhs.at[1, :, 3:-2:2].set(gamma_over_implicit_ts * lhs_tp_wp3[0, :, 1:-1])
    lhs = lhs.at[2, :, 3:-2:2].set(
        lhs_ma_zt[1, :, 1:-1] + lhs_diff_zt[1, :, 1:-1]
        + lhs_ac_pr2_wp3[:, 1:-1]
        + gamma_over_implicit_ts * lhs_pr1_wp3[:, 1:-1]
        + lhs_splat_wp3[:, 1:-1]
        + invrs_dt
    )
    lhs = lhs.at[3, :, 3:-2:2].set(gamma_over_implicit_ts * lhs_tp_wp3[1, :, 1:-1])
    lhs = lhs.at[4, :, 3:-2:2].set(lhs_ma_zt[2, :, 1:-1] + lhs_diff_zt[2, :, 1:-1])

    lhs = lhs.at[2, :, 2 * nzm - 3].set(one)
    lhs = lhs.at[2, :, 2 * nzm - 2].set(one)

    if l_tke_aniso:
        lhs = lhs.at[2, :, 2:-1:2].add(
            gamma_over_implicit_ts * lhs_pr1_wp2[:, 1:-1]
        )

    lhs = lhs.at[2, :, 2:-1:2].add(lhs_splat_wp2[:, 1:-1])

    if not l_explicit_turbulent_adv_wp3:
        lhs = lhs.at[:, :, 3:-2:2].add(
            gamma_over_implicit_ts * wp3_term_ta_lhs_result[:, :, 1:-1]
        )

    return lhs


def wp23_rhs(
    nzm, nzt, ngrdcol, gr, dt, fcor_y,
    wp3_term_ta_lhs_result,
    lhs_diff_zm, lhs_diff_zt, lhs_diff_zm_crank, lhs_diff_zt_crank,
    lhs_tp_wp3, lhs_adv_tp_wp3, lhs_pr_tp_wp3,
    lhs_ta_wp3, lhs_dp1_wp2, rhs_dp1_wp2, lhs_pr1_wp2,
    rhs_pr1_wp2, lhs_pr1_wp3, rhs_pr1_wp3, rhs_bp_pr2_wp2,
    rhs_pr_dfsn_wp2, rhs_bp1_pr2_wp3, rhs_pr3_wp2, rhs_pr3_wp3,
    rhs_ta_wp3, rhs_pr_turb_wp3, rhs_pr_dfsn_wp3,
    wp2, wp3, wpup2, wpvp2,
    wpthvp, wp2thvp, wp2up, up2, vp2, upwp,
    C11_Skw_fnc, thv_ds_zm, thv_ds_zt,
    lhs_splat_wp2, lhs_splat_wp3,
    clubb_params,
    iiPDF_type,
    l_tke_aniso,
    l_use_tke_in_wp2_wp3_K_dfsn,
    l_ho_nontrad_coriolis,
    stats,
):
    """Compute RHS vector for w'^2 and w'^3."""
    invrs_dt = one / dt
    rhs = jnp.zeros((ngrdcol, 2 * nzm - 1), dtype=jnp.float64)

    rhs = rhs.at[:, 3:-2:2].set(rhs_pr_turb_wp3[:, 1:-1] + rhs_pr_dfsn_wp3[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr_dfsn_wp2[:, 1:-1])

    if l_crank_nich_diff:
        if gr.grid_dir_indx > 0:
            rhs = rhs.at[:, 2:-1:2].add(
                -lhs_diff_zm_crank[2, :, 1:-1] * wp2[:, :-2]
                -lhs_diff_zm_crank[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_diff_zm_crank[0, :, 1:-1] * wp2[:, 2:]
            )
            rhs = rhs.at[:, 3:-2:2].add(
                -lhs_diff_zt_crank[2, :, 1:-1] * wp3[:, :-2]
                -lhs_diff_zt_crank[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_diff_zt_crank[0, :, 1:-1] * wp3[:, 2:]
            )
        else:
            rhs = rhs.at[:, 2:-1:2].add(
                -lhs_diff_zm_crank[0, :, 1:-1] * wp2[:, 2:]
                -lhs_diff_zm_crank[1, :, 1:-1] * wp2[:, 1:-1]
                -lhs_diff_zm_crank[2, :, 1:-1] * wp2[:, :-2]
            )
            rhs = rhs.at[:, 3:-2:2].add(
                -lhs_diff_zt_crank[0, :, 1:-1] * wp3[:, 2:]
                -lhs_diff_zt_crank[1, :, 1:-1] * wp3[:, 1:-1]
                -lhs_diff_zt_crank[2, :, 1:-1] * wp3[:, :-2]
            )

    if l_use_tke_in_wp2_wp3_K_dfsn:
        if gr.grid_dir_indx > 0:
            rhs = rhs.at[:, 2:-1:2].add(
                -lhs_diff_zm[2, :, 1:-1] * (up2[:, :-2] + vp2[:, :-2])
                -lhs_diff_zm[1, :, 1:-1] * (up2[:, 1:-1] + vp2[:, 1:-1])
                -lhs_diff_zm[0, :, 1:-1] * (up2[:, 2:] + vp2[:, 2:])
            )
            rhs = rhs.at[:, 3:-2:2].add(
                -lhs_diff_zt[2, :, 1:-1] * (wpup2[:, :-2] + wpvp2[:, :-2])
                -lhs_diff_zt[1, :, 1:-1] * (wpup2[:, 1:-1] + wpvp2[:, 1:-1])
                -lhs_diff_zt[2, :, 1:-1] * (wpup2[:, 2:] + wpvp2[:, 2:])
            )
        else:
            rhs = rhs.at[:, 2:-1:2].add(
                -lhs_diff_zm[0, :, 1:-1] * (up2[:, 2:] + vp2[:, 2:])
                -lhs_diff_zm[1, :, 1:-1] * (up2[:, 1:-1] + vp2[:, 1:-1])
                -lhs_diff_zm[2, :, 1:-1] * (up2[:, :-2] + vp2[:, :-2])
            )
            rhs = rhs.at[:, 3:-2:2].add(
                -lhs_diff_zt[0, :, 1:-1] * (wpup2[:, 2:] + wpvp2[:, 2:])
                -lhs_diff_zt[1, :, 1:-1] * (wpup2[:, 1:-1] + wpvp2[:, 1:-1])
                -lhs_diff_zt[0, :, 1:-1] * (wpup2[:, :-2] + wpvp2[:, :-2])
            )

    if l_tke_aniso:
        rhs = rhs.at[:, 2:-1:2].add(rhs_pr1_wp2[:, 1:-1])
        rhs = rhs.at[:, 2:-1:2].add(
            (one - gamma_over_implicit_ts)
            * (-lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1])
        )

    if l_ho_nontrad_coriolis:
        rhs = rhs.at[:, 2:-1:2].add(two * fcor_y[:, None] * upwp[:, 1:-1])
        rhs = rhs.at[:, 3:-2:2].add(three * fcor_y[:, None] * wp2up[:, 1:-1])

    rhs = rhs.at[:, 3:-2:2].add(invrs_dt * wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(
        (one - gamma_over_implicit_ts)
        * (
            -lhs_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
            -lhs_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]
        )
    )
    rhs = rhs.at[:, 3:-2:2].add(rhs_bp1_pr2_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(rhs_pr1_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(
        (one - gamma_over_implicit_ts)
        * (-lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1])
    )

    rhs = rhs.at[:, 2:-1:2].add(invrs_dt * wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_bp_pr2_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr3_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_dp1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(
        (one - gamma_over_implicit_ts)
        * (-lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1])
    )

    if l_explicit_turbulent_adv_wp3:
        rhs = rhs.at[:, 3:-2:2].add(rhs_ta_wp3[:, 1:-1])
    else:
        if iiPDF_type == iiPDF_ADG1:
            rhs = rhs.at[:, 3:-2:2].add(
                (one - gamma_over_implicit_ts)
                * (
                    -wp3_term_ta_lhs_result[0, :, 1:-1] * wp3[:, 2:]
                    -wp3_term_ta_lhs_result[1, :, 1:-1] * wp2[:, 2:-1]
                    -wp3_term_ta_lhs_result[2, :, 1:-1] * wp3[:, 1:-1]
                    -wp3_term_ta_lhs_result[3, :, 1:-1] * wp2[:, 1:-2]
                    -wp3_term_ta_lhs_result[4, :, 1:-1] * wp3[:, :-2]
                )
            )
        elif iiPDF_type == iiPDF_new or iiPDF_type == iiPDF_new_hybrid:
            rhs = rhs.at[:, 3:-2:2].add(
                (one - gamma_over_implicit_ts)
                * (
                    -lhs_ta_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                    -lhs_ta_wp3[1, :, 1:-1] * wp2[:, 1:-2]
                )
            )

    if gr.grid_dir_indx > 0:
        rhs = rhs.at[:, 0].set(wp2[:, gr.k_lb_zm])
        rhs = rhs.at[:, 1].set(zero)
        rhs = rhs.at[:, 2 * nzt - 1].set(zero)
        rhs = rhs.at[:, 2 * nzm - 2].set(w_tol_sqd)
    else:
        rhs = rhs.at[:, 2 * nzm - 2].set(wp2[:, gr.k_lb_zm])
        rhs = rhs.at[:, 2 * nzt - 1].set(zero)
        rhs = rhs.at[:, 1].set(zero)
        rhs = rhs.at[:, 0].set(w_tol_sqd)

    if stats.l_sample:
        C_uu_buoy_zeros = jnp.zeros((ngrdcol,), dtype=jnp.float64)
        C_uu_buoy_plus_one = clubb_params[:, iC_uu_buoy] + one
        C11_Skw_fnc_zeros = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        C11_Skw_fnc_plus_one = C11_Skw_fnc + one

        rhs_bp_wp2 = wp2_terms_bp_pr2_rhs(
            nzm, ngrdcol, gr, C_uu_buoy_zeros, thv_ds_zm, wpthvp,
        )
        rhs_pr2_wp2 = wp2_terms_bp_pr2_rhs(
            nzm, ngrdcol, gr, C_uu_buoy_plus_one, thv_ds_zm, wpthvp,
        )
        rhs_bp1_wp3 = wp3_terms_bp1_pr2_rhs(
            nzt, ngrdcol, gr, C11_Skw_fnc_zeros, thv_ds_zt, wp2thvp,
        )
        rhs_pr2_wp3 = wp3_terms_bp1_pr2_rhs(
            nzt, ngrdcol, gr, C11_Skw_fnc_plus_one, thv_ds_zt, wp2thvp,
        )

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(rhs_bp_wp2[:, 1:-1])
        stats = stats.update("wp2_bp", stats_tmp_zm)

        if l_ho_nontrad_coriolis:
            stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(two * fcor_y[:, None] * upwp[:, 1:-1])
            stats = stats.update("wp2_nct", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(rhs_pr_dfsn_wp2[:, 1:-1])
        stats = stats.update("wp2_pr_dfsn", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(-lhs_splat_wp2[:, 1:-1] * wp2[:, 1:-1])
        stats = stats.update("wp2_splat", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(rhs_pr3_wp2[:, 1:-1])
        stats = stats.update("wp2_pr3", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(-rhs_pr2_wp2[:, 1:-1])
        stats = stats.begin_budget("wp2_pr2", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(-rhs_dp1_wp2[:, 1:-1])
        stats = stats.begin_budget("wp2_dp1", stats_tmp_zm)

        stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
        stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
            (one - gamma_over_implicit_ts)
            * (-lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1])
        )
        stats = stats.update_budget("wp2_dp1", stats_tmp_zm)

        if l_crank_nich_diff or l_use_tke_in_wp2_wp3_K_dfsn:
            stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
            if l_crank_nich_diff:
                if gr.grid_dir_indx > 0:
                    stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].add(
                        lhs_diff_zm_crank[2, :, 1:-1] * wp2[:, :-2]
                        + lhs_diff_zm_crank[1, :, 1:-1] * wp2[:, 1:-1]
                        + lhs_diff_zm_crank[0, :, 1:-1] * wp2[:, 2:]
                    )
                else:
                    stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].add(
                        lhs_diff_zm_crank[0, :, 1:-1] * wp2[:, 2:]
                        + lhs_diff_zm_crank[1, :, 1:-1] * wp2[:, 1:-1]
                        + lhs_diff_zm_crank[2, :, 1:-1] * wp2[:, :-2]
                    )
            if l_use_tke_in_wp2_wp3_K_dfsn:
                if gr.grid_dir_indx > 0:
                    stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].add(
                        lhs_diff_zm[2, :, 1:-1] * (up2[:, :-2] + vp2[:, :-2])
                        + lhs_diff_zm[1, :, 1:-1] * (up2[:, 1:-1] + vp2[:, 1:-1])
                        + lhs_diff_zm[0, :, 1:-1] * (up2[:, 2:] + vp2[:, 2:])
                    )
                else:
                    stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].add(
                        lhs_diff_zm[0, :, 1:-1] * (up2[:, 2:] + vp2[:, 2:])
                        + lhs_diff_zm[1, :, 1:-1] * (up2[:, 1:-1] + vp2[:, 1:-1])
                        + lhs_diff_zm[2, :, 1:-1] * (up2[:, :-2] + vp2[:, :-2])
                    )
            stats = stats.begin_budget("wp2_dp2", stats_tmp_zm)

        if l_tke_aniso:
            stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(-rhs_pr1_wp2[:, 1:-1])
            stats = stats.begin_budget("wp2_pr1", stats_tmp_zm)
            stats_tmp_zm = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
            stats_tmp_zm = stats_tmp_zm.at[:, 1:-1].set(
                (one - gamma_over_implicit_ts)
                * (-lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1])
            )
            stats = stats.update_budget("wp2_pr1", stats_tmp_zm)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(rhs_bp1_wp3[:, 1:-1])
        stats = stats.update("wp3_bp1", stats_tmp_zt)

        if l_ho_nontrad_coriolis:
            stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(three * fcor_y[:, None] * wp2up[:, 1:-1])
            stats = stats.update("wp3_nct", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(rhs_pr_turb_wp3[:, 1:-1])
        stats = stats.update("wp3_pr_turb", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(rhs_pr_dfsn_wp3[:, 1:-1])
        stats = stats.update("wp3_pr_dfsn", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(-lhs_splat_wp3[:, 1:-1] * wp3[:, 1:-1])
        stats = stats.update("wp3_splat", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(-rhs_pr2_wp3[:, 1:-1])
        stats = stats.begin_budget("wp3_pr2", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(-rhs_pr1_wp3[:, 1:-1])
        stats = stats.begin_budget("wp3_pr1", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            (one - gamma_over_implicit_ts)
            * (-lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1])
        )
        stats = stats.update_budget("wp3_pr1", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        if l_explicit_turbulent_adv_wp3:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(-rhs_ta_wp3[:, 1:-1])
        elif iiPDF_type == iiPDF_ADG1:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -(one - gamma_over_implicit_ts)
                * (
                    -wp3_term_ta_lhs_result[0, :, 1:-1] * wp3[:, 2:]
                    -wp3_term_ta_lhs_result[1, :, 1:-1] * wp2[:, 2:-1]
                    -wp3_term_ta_lhs_result[2, :, 1:-1] * wp3[:, 1:-1]
                    -wp3_term_ta_lhs_result[3, :, 1:-1] * wp2[:, 1:-2]
                    -wp3_term_ta_lhs_result[4, :, 1:-1] * wp3[:, :-2]
                )
            )
        elif iiPDF_type == iiPDF_new or iiPDF_type == iiPDF_new_hybrid:
            stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
                -(one - gamma_over_implicit_ts)
                * (
                    -lhs_ta_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                    -lhs_ta_wp3[1, :, 1:-1] * wp2[:, 1:-2]
                )
            )
        stats = stats.begin_budget("wp3_ta", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -(one - gamma_over_implicit_ts)
            * (
                -lhs_adv_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                -lhs_adv_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]
            )
        )
        stats = stats.begin_budget("wp3_tp", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(
            -(one - gamma_over_implicit_ts)
            * (
                -lhs_pr_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                -lhs_pr_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]
            )
        )
        stats = stats.begin_budget("wp3_pr_tp", stats_tmp_zt)

        stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
        stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].set(rhs_pr3_wp3[:, 1:-1])
        stats = stats.begin_budget("wp3_pr3", stats_tmp_zt)

        if l_crank_nich_diff or l_use_tke_in_wp2_wp3_K_dfsn:
            stats_tmp_zt = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
            if l_crank_nich_diff:
                if gr.grid_dir_indx > 0:
                    stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].add(
                        lhs_diff_zt[2, :, 1:-1] * wp3[:, :-2]
                        + lhs_diff_zt[1, :, 1:-1] * wp3[:, 1:-1]
                        + lhs_diff_zt[0, :, 1:-1] * wp3[:, 2:]
                    )
                else:
                    stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].add(
                        lhs_diff_zt[0, :, 1:-1] * wp3[:, 2:]
                        + lhs_diff_zt[1, :, 1:-1] * wp3[:, 1:-1]
                        + lhs_diff_zt[2, :, 1:-1] * wp3[:, :-2]
                    )
            if l_use_tke_in_wp2_wp3_K_dfsn:
                if gr.grid_dir_indx > 0:
                    stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].add(
                        lhs_diff_zt[2, :, 1:-1] * (wpup2[:, :-2] + wpvp2[:, :-2])
                        + lhs_diff_zt[1, :, 1:-1] * (wpup2[:, 1:-1] + wpvp2[:, 1:-1])
                        + lhs_diff_zt[0, :, 1:-1] * (wpup2[:, 2:] + wpvp2[:, 2:])
                    )
                else:
                    stats_tmp_zt = stats_tmp_zt.at[:, 1:-1].add(
                        lhs_diff_zt[0, :, 1:-1] * (wpup2[:, 2:] + wpvp2[:, 2:])
                        + lhs_diff_zt[1, :, 1:-1] * (wpup2[:, 1:-1] + wpvp2[:, 1:-1])
                        + lhs_diff_zt[2, :, 1:-1] * (wpup2[:, :-2] + wpvp2[:, :-2])
                    )
            stats = stats.begin_budget("wp3_dp1", stats_tmp_zt)

    return rhs, stats


def wp2_term_ta_lhs(nzm, nzt, ngrdcol, gr, rho_ds_zt, invrs_rho_ds_zm):
    """Turbulent advection term for w'^2: implicit portion of the code."""
    del nzt
    lhs_ta_wp2 = jnp.zeros((ndiags2, ngrdcol, nzm), dtype=jnp.float64)
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    lhs_ta_wp2 = lhs_ta_wp2.at[0, :, 1:-1].set(fac * rho_ds_zt[:, 1:])
    lhs_ta_wp2 = lhs_ta_wp2.at[1, :, 1:-1].set(-fac * rho_ds_zt[:, :-1])
    return lhs_ta_wp2


def wp2_terms_ac_pr2_lhs(nzm, nzt, ngrdcol, gr, C_uu_shr, wm_zt):
    """Accumulation of w'^2 and w'^2 pressure term 2: implicit portion."""
    del nzt
    lhs_ac_pr2_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    lhs_ac_pr2_wp2 = lhs_ac_pr2_wp2.at[:, 1:-1].set(
        (one - C_uu_shr[:, None]) * two * gr.invrs_dzm[:, 1:-1]
        * (wm_zt[:, 1:] - wm_zt[:, :-1])
    )
    return lhs_ac_pr2_wp2


def wp2_term_dp1_lhs(nzm, ngrdcol, gr, C1_Skw_fnc, invrs_tau1m):
    """Dissipation term 1 for w'^2: implicit portion of the code."""
    del gr
    lhs_dp1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    lhs_dp1_wp2 = lhs_dp1_wp2.at[:, 1:-1].set(
        C1_Skw_fnc[:, 1:-1] * invrs_tau1m[:, 1:-1]
    )
    return lhs_dp1_wp2


def wp2_term_pr1_lhs(nzm, ngrdcol, gr, C4, invrs_tau_C4_zm):
    """Pressure term 1 for w'^2: implicit portion of the code."""
    del gr
    lhs_pr1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    lhs_pr1_wp2 = lhs_pr1_wp2.at[:, 1:-1].set(
        (two * C4[:, None] * invrs_tau_C4_zm[:, 1:-1]) / three
    )
    return lhs_pr1_wp2


def wp2_terms_bp_pr2_rhs(nzm, ngrdcol, gr, C_uu_buoy, thv_ds_zm, wpthvp):
    """Buoyancy production of w'^2 and w'^2 pressure term 2."""
    del gr
    rhs_bp_pr2_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    rhs_bp_pr2_wp2 = rhs_bp_pr2_wp2.at[:, 1:-1].set(
        (one - C_uu_buoy[:, None]) * two
        * (grav / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
    )
    return rhs_bp_pr2_wp2


def wp2_term_dp1_rhs(
    nzm, ngrdcol, gr, C1_Skw_fnc,
    invrs_tau1m, threshold, up2, vp2,
    l_damp_wp2_using_em,
):
    """Dissipation term 1 for w'^2: explicit portion of the code."""
    del gr
    rhs_dp1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    if l_damp_wp2_using_em:
        rhs_dp1_wp2 = rhs_dp1_wp2.at[:, 1:-1].set(
            -(C1_Skw_fnc[:, 1:-1] * invrs_tau1m[:, 1:-1])
            * (up2[:, 1:-1] + vp2[:, 1:-1])
        )
    else:
        rhs_dp1_wp2 = rhs_dp1_wp2.at[:, 1:-1].set(
            (C1_Skw_fnc[:, 1:-1] * invrs_tau1m[:, 1:-1]) * threshold
        )
    return rhs_dp1_wp2


def wp2_term_pr3_rhs(
    nzm, nzt, ngrdcol, gr,
    C_uu_shr,
    C_uu_buoy,
    thv_ds_zm, wpthvp, upwp,
    um, vpwp, vm,
):
    """Pressure term 3 for w'^2: explicit portion of the code."""
    del nzt
    rhs_pr3_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    rhs_pr3_wp2 = rhs_pr3_wp2.at[:, 1:-1].set(
        two_thirds
        * (
            C_uu_buoy[:, None]
            * (grav / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
            + C_uu_shr[:, None]
            * (
                -upwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
                -vpwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
            )
        )
    )
    rhs_pr3_wp2 = rhs_pr3_wp2.at[:, 1:-1].set(
        jnp.maximum(rhs_pr3_wp2[:, 1:-1], zero_threshold)
    )
    return rhs_pr3_wp2


def wp2_term_pr1_rhs(nzm, ngrdcol, gr, C4, up2, vp2, invrs_tau_C4_zm):
    """Pressure term 1 for w'^2: explicit portion of the code."""
    del gr
    rhs_pr1_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    rhs_pr1_wp2 = rhs_pr1_wp2.at[:, 1:-1].set(
        (C4[:, None] * (up2[:, 1:-1] + vp2[:, 1:-1])
         * invrs_tau_C4_zm[:, 1:-1]) / three
    )
    return rhs_pr1_wp2


def wp2_term_pr_dfsn_rhs(
    nzm, nzt, ngrdcol, gr, C_wp2_pr_dfsn,
    rho_ds_zt, invrs_rho_ds_zm,
    wpup2, wpvp2, wp3,
):
    """Pressure-diffusion RHS for w'^2."""
    del nzt
    wpuip2 = wpup2 + wpvp2 + wp3
    rhs_pr_dfsn_wp2 = jnp.zeros((ngrdcol, nzm), dtype=jnp.float64)
    rhs_pr_dfsn_wp2 = rhs_pr_dfsn_wp2.at[:, 1:-1].set(
        C_wp2_pr_dfsn[:, None] * invrs_rho_ds_zm[:, 1:-1]
        * gr.invrs_dzm[:, 1:-1]
        * (
            rho_ds_zt[:, 1:] * wpuip2[:, 1:]
            - rho_ds_zt[:, :-1] * wpuip2[:, :-1]
        )
    )
    if gr.grid_dir_indx > 0:
        rhs_pr_dfsn_wp2 = rhs_pr_dfsn_wp2.at[:, gr.k_lb_zm].set(
            rhs_pr_dfsn_wp2[:, gr.k_lb_zm + gr.grid_dir_indx]
        )
    else:
        rhs_pr_dfsn_wp2 = rhs_pr_dfsn_wp2.at[:, gr.k_lb_zm].set(
            rhs_pr_dfsn_wp2[:, gr.k_lb_zm + gr.grid_dir_indx]
        )
    rhs_pr_dfsn_wp2 = rhs_pr_dfsn_wp2.at[:, gr.k_ub_zm].set(zero)
    return rhs_pr_dfsn_wp2


def wp3_term_ta_new_pdf_lhs(
    nzm, nzt, ngrdcol, gr, coef_wp4_implicit,
    wp2, rho_ds_zm, invrs_rho_ds_zt,
):
    """Turbulent advection of <w'^3>: implicit portion for the new PDF."""
    del nzm
    lhs_ta_wp3 = jnp.zeros((ndiags2, ngrdcol, nzt), dtype=jnp.float64)
    lhs_ta_wp3 = lhs_ta_wp3.at[0, :, 1:-1].set(
        invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
        * rho_ds_zm[:, 2:nzt] * coef_wp4_implicit[:, 2:nzt] * wp2[:, 2:nzt]
    )
    lhs_ta_wp3 = lhs_ta_wp3.at[1, :, 1:-1].set(
        -invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
        * rho_ds_zm[:, 1:nzt - 1] * coef_wp4_implicit[:, 1:nzt - 1] * wp2[:, 1:nzt - 1]
    )
    return lhs_ta_wp3


def wp3_term_ta_ADG1_lhs(
    nzm, nzt, ngrdcol, gr,
    wp2, a1_coef, a1_coef_zt,
    a3_coef, a3_coef_zt,
    wp3_on_wp2, rho_ds_zm,
    rho_ds_zt, invrs_rho_ds_zt,
    l_standard_term_ta,
    l_partial_upwind_wp3,
):
    """Turbulent advection of w'^3: implicit portion for the ADG1 PDF."""
    del nzm
    lhs_ta_wp3 = jnp.zeros((ndiags5, ngrdcol, nzt), dtype=jnp.float64)
    inv = invrs_rho_ds_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]

    if l_standard_term_ta:
        if not l_partial_upwind_wp3:
            lhs_ta_wp3 = lhs_ta_wp3.at[0, :, 1:-1].set(
                inv * idzt
                * rho_ds_zm[:, 2:nzt] * a1_coef[:, 2:nzt] * wp3_on_wp2[:, 2:nzt]
                * gr.weights_zt2zm[:, 2:nzt, T_ABOVE]
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[1, :, 1:-1].set(
                inv * idzt
                * rho_ds_zm[:, 2:nzt] * a3_coef[:, 2:nzt] * wp2[:, 2:nzt]
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[2, :, 1:-1].set(
                inv * idzt
                * (
                    rho_ds_zm[:, 2:nzt] * a1_coef[:, 2:nzt] * wp3_on_wp2[:, 2:nzt]
                    * gr.weights_zt2zm[:, 2:nzt, T_BELOW]
                    - rho_ds_zm[:, 1:nzt - 1] * a1_coef[:, 1:nzt - 1]
                    * wp3_on_wp2[:, 1:nzt - 1]
                    * gr.weights_zt2zm[:, 1:nzt - 1, T_ABOVE]
                )
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[3, :, 1:-1].set(
                -inv * idzt
                * rho_ds_zm[:, 1:nzt - 1] * a3_coef[:, 1:nzt - 1]
                * wp2[:, 1:nzt - 1]
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[4, :, 1:-1].set(
                -inv * idzt
                * rho_ds_zm[:, 1:nzt - 1] * a1_coef[:, 1:nzt - 1]
                * wp3_on_wp2[:, 1:nzt - 1]
                * gr.weights_zt2zm[:, 1:nzt - 1, T_BELOW]
            )
        else:
            grid_dir = gr.grid_dir
            lhs_ta_wp3 = lhs_ta_wp3.at[0, :, 1:-1].set(
                inv * idzt * rho_ds_zt[:, 2:nzt] * grid_dir
                * jnp.minimum(
                    grid_dir * a1_coef[:, 2:nzt] * wp3_on_wp2[:, 2:nzt],
                    zero,
                )
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[1, :, 1:-1].set(
                inv * idzt
                * rho_ds_zm[:, 2:nzt] * a3_coef[:, 2:nzt] * wp2[:, 2:nzt]
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[2, :, 1:-1].set(
                inv * idzt * rho_ds_zt[:, 1:-1] * grid_dir
                * (
                    jnp.maximum(
                        grid_dir * a1_coef[:, 2:nzt] * wp3_on_wp2[:, 2:nzt],
                        zero,
                    )
                    - jnp.minimum(
                        grid_dir * a1_coef[:, 1:nzt - 1]
                        * wp3_on_wp2[:, 1:nzt - 1],
                        zero,
                    )
                )
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[3, :, 1:-1].set(
                -inv * idzt
                * rho_ds_zm[:, 1:nzt - 1] * a3_coef[:, 1:nzt - 1]
                * wp2[:, 1:nzt - 1]
            )
            lhs_ta_wp3 = lhs_ta_wp3.at[4, :, 1:-1].set(
                -inv * idzt * rho_ds_zt[:, :nzt - 2] * grid_dir
                * jnp.maximum(
                    grid_dir * a1_coef[:, 1:nzt - 1]
                    * wp3_on_wp2[:, 1:nzt - 1],
                    zero,
                )
            )
    else:
        lhs_ta_wp3 = lhs_ta_wp3.at[0, :, 1:-1].set(
            inv * a1_coef_zt[:, 1:-1] * idzt
            * rho_ds_zm[:, 2:nzt] * wp3_on_wp2[:, 2:nzt]
            * gr.weights_zt2zm[:, 2:nzt, T_ABOVE]
        )
        lhs_ta_wp3 = lhs_ta_wp3.at[1, :, 1:-1].set(
            inv * a3_coef_zt[:, 1:-1] * idzt
            * rho_ds_zm[:, 2:nzt] * wp2[:, 2:nzt]
        )
        lhs_ta_wp3 = lhs_ta_wp3.at[2, :, 1:-1].set(
            inv * a1_coef_zt[:, 1:-1] * idzt
            * (
                rho_ds_zm[:, 2:nzt] * wp3_on_wp2[:, 2:nzt]
                * gr.weights_zt2zm[:, 2:nzt, T_BELOW]
                - rho_ds_zm[:, 1:nzt - 1] * wp3_on_wp2[:, 1:nzt - 1]
                * gr.weights_zt2zm[:, 1:nzt - 1, T_ABOVE]
            )
        )
        lhs_ta_wp3 = lhs_ta_wp3.at[3, :, 1:-1].set(
            -inv * a3_coef_zt[:, 1:-1] * idzt
            * rho_ds_zm[:, 1:nzt - 1] * wp2[:, 1:nzt - 1]
        )
        lhs_ta_wp3 = lhs_ta_wp3.at[4, :, 1:-1].set(
            -inv * a1_coef_zt[:, 1:-1] * idzt
            * rho_ds_zm[:, 1:nzt - 1] * wp3_on_wp2[:, 1:nzt - 1]
            * gr.weights_zt2zm[:, 1:nzt - 1, T_BELOW]
        )

    return lhs_ta_wp3


def wp3_term_tp_lhs(nzm, nzt, ngrdcol, gr, coef_wp3_tp, wp2, rho_ds_zm, invrs_rho_ds_zt):
    """Turbulent production of w'^3: implicit portion of the code."""
    del nzm
    lhs_tp_wp3 = jnp.zeros((ndiags2, ngrdcol, nzt), dtype=jnp.float64)
    lhs_tp_wp3 = lhs_tp_wp3.at[0, :, 1:-1].set(
        -coef_wp3_tp[:, None] * three * invrs_rho_ds_zt[:, 1:-1]
        * gr.invrs_dzt[:, 1:-1]
        * rho_ds_zm[:, 2:nzt] * wp2[:, 2:nzt]
        + coef_wp3_tp[:, None] * three_halves * gr.invrs_dzt[:, 1:-1]
        * wp2[:, 2:nzt]
    )
    lhs_tp_wp3 = lhs_tp_wp3.at[1, :, 1:-1].set(
        coef_wp3_tp[:, None] * three * invrs_rho_ds_zt[:, 1:-1]
        * gr.invrs_dzt[:, 1:-1]
        * rho_ds_zm[:, 1:nzt - 1] * wp2[:, 1:nzt - 1]
        - coef_wp3_tp[:, None] * three_halves * gr.invrs_dzt[:, 1:-1]
        * wp2[:, 1:nzt - 1]
    )
    return lhs_tp_wp3


def wp3_terms_ac_pr2_lhs(nzm, nzt, ngrdcol, gr, C11_Skw_fnc, wm_zm):
    """Accumulation of w'^3 and w'^3 pressure term 2: implicit portion."""
    del nzm
    lhs_ac_pr2_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    lhs_ac_pr2_wp3 = lhs_ac_pr2_wp3.at[:, 1:-1].set(
        (one - C11_Skw_fnc[:, 1:-1])
        * three * gr.invrs_dzt[:, 1:-1]
        * (wm_zm[:, 2:nzt] - wm_zm[:, 1:nzt - 1])
    )
    return lhs_ac_pr2_wp3


def wp3_term_pr1_lhs(
    nzt, ngrdcol, gr,
    C8, C8b,
    invrs_tau_wp3_zt, Skw_zt,
    l_damp_wp3_Skw_squared,
):
    """Pressure term 1 for w'^3: implicit portion of the code."""
    del gr
    lhs_pr1_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if l_damp_wp3_Skw_squared:
        lhs_pr1_wp3 = lhs_pr1_wp3.at[:, 1:-1].set(
            (C8[:, None] * invrs_tau_wp3_zt[:, 1:-1])
            * (three * C8b[:, None] * Skw_zt[:, 1:-1] ** 2 + one)
        )
    else:
        lhs_pr1_wp3 = lhs_pr1_wp3.at[:, 1:-1].set(
            (C8[:, None] * invrs_tau_wp3_zt[:, 1:-1])
            * (five * C8b[:, None] * Skw_zt[:, 1:-1] ** 4 + one)
        )
    return lhs_pr1_wp3


def wp3_term_ta_explicit_rhs(nzm, nzt, ngrdcol, gr, wp4, rho_ds_zm, invrs_rho_ds_zt):
    """Turbulent advection of <w'^3>: explicit portion of the code."""
    del nzm
    rhs_ta_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    rhs_ta_wp3 = rhs_ta_wp3.at[:, 1:-1].set(
        -invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
        * (
            rho_ds_zm[:, 2:nzt] * wp4[:, 2:nzt]
            - rho_ds_zm[:, 1:nzt - 1] * wp4[:, 1:nzt - 1]
        )
    )
    return rhs_ta_wp3


def wp3_terms_bp1_pr2_rhs(nzt, ngrdcol, gr, C11_Skw_fnc, thv_ds_zt, wp2thvp):
    """Buoyancy production of w'^3 and w'^3 pressure term 2."""
    del gr
    rhs_bp1_pr2_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    rhs_bp1_pr2_wp3 = rhs_bp1_pr2_wp3.at[:, 1:-1].set(
        (one - C11_Skw_fnc[:, 1:-1])
        * three * (grav / thv_ds_zt[:, 1:-1]) * wp2thvp[:, 1:-1]
    )
    return rhs_bp1_pr2_wp3


def wp3_term_pr_turb_rhs(
    nzm, nzt, ngrdcol, gr, C_wp3_pr_turb,
    Kh_zt, wpthvp, dum_dz, dvm_dz,
    upwp, vpwp,
    thv_ds_zt,
    rho_ds_zm, invrs_rho_ds_zt,
    em, wp2,
    l_use_tke_in_wp3_pr_turb_term,
):
    """Pressure-turbulence correlation RHS for w'^3."""
    del nzm
    rhs_pr_turb_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if not l_use_tke_in_wp3_pr_turb_term:
        rhs_pr_turb_wp3 = rhs_pr_turb_wp3.at[:, 1:-1].set(
            -C_wp3_pr_turb[:, None] * Kh_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
            * (
                grav / thv_ds_zt[:, 1:-1]
                * (wpthvp[:, 2:nzt] - wpthvp[:, 1:nzt - 1])
                - (
                    upwp[:, 2:nzt] * dum_dz[:, 2:nzt]
                    - upwp[:, 1:nzt - 1] * dum_dz[:, 1:nzt - 1]
                )
                - (
                    vpwp[:, 2:nzt] * dvm_dz[:, 2:nzt]
                    - vpwp[:, 1:nzt - 1] * dvm_dz[:, 1:nzt - 1]
                )
            )
        )
    else:
        rhs_pr_turb_wp3 = rhs_pr_turb_wp3.at[:, 1:-1].set(
            -C_wp3_pr_turb[:, None] * invrs_rho_ds_zt[:, 1:-1]
            * gr.invrs_dzt[:, 1:-1]
            * (
                rho_ds_zm[:, 2:nzt] * wp2[:, 2:nzt] * em[:, 2:nzt]
                - rho_ds_zm[:, 1:nzt - 1] * wp2[:, 1:nzt - 1] * em[:, 1:nzt - 1]
            )
        )
    return rhs_pr_turb_wp3


def wp3_term_pr_dfsn_rhs(
    nzm, nzt, ngrdcol, gr, C_wp3_pr_dfsn,
    rho_ds_zm, invrs_rho_ds_zt,
    wp2up2, wp2vp2, wp4,
    up2, vp2, wp2,
):
    """Pressure-diffusion RHS for w'^3."""
    del nzm
    wp2uip2 = wp2up2 + wp2vp2 + wp4
    wp2_uip2 = wp2 * up2 + wp2 * vp2 + wp2 * wp2
    rhs_pr_dfsn_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    rhs_pr_dfsn_wp3 = rhs_pr_dfsn_wp3.at[:, 1:-1].set(
        C_wp3_pr_dfsn[:, None] * invrs_rho_ds_zt[:, 1:-1]
        * gr.invrs_dzt[:, 1:-1]
        * (
            rho_ds_zm[:, 2:nzt] * (wp2uip2[:, 2:nzt] - wp2_uip2[:, 2:nzt])
            - rho_ds_zm[:, 1:nzt - 1]
            * (wp2uip2[:, 1:nzt - 1] - wp2_uip2[:, 1:nzt - 1])
        )
    )
    return rhs_pr_dfsn_wp3


def wp3_term_pr1_rhs(
    nzt, ngrdcol, gr,
    C8, C8b,
    invrs_tau_wp3_zt, Skw_zt, wp3,
    l_damp_wp3_Skw_squared,
):
    """Pressure term 1 for w'^3: explicit portion of the code."""
    del gr
    rhs_pr1_wp3 = jnp.zeros((ngrdcol, nzt), dtype=jnp.float64)
    if l_damp_wp3_Skw_squared:
        rhs_pr1_wp3 = rhs_pr1_wp3.at[:, 1:-1].set(
            (C8[:, None] * invrs_tau_wp3_zt[:, 1:-1])
            * (two * C8b[:, None] * Skw_zt[:, 1:-1] ** 2) * wp3[:, 1:-1]
        )
    else:
        rhs_pr1_wp3 = rhs_pr1_wp3.at[:, 1:-1].set(
            (C8[:, None] * invrs_tau_wp3_zt[:, 1:-1])
            * (four * C8b[:, None] * Skw_zt[:, 1:-1] ** 4) * wp3[:, 1:-1]
        )
    return rhs_pr1_wp3


__all__ = [
    "advance_wp2_wp3",
    "wp23_solve",
    "wp23_lhs",
    "wp23_rhs",
    "wp2_term_ta_lhs",
    "wp2_terms_ac_pr2_lhs",
    "wp2_term_dp1_lhs",
    "wp2_term_pr1_lhs",
    "wp2_terms_bp_pr2_rhs",
    "wp2_term_dp1_rhs",
    "wp2_term_pr3_rhs",
    "wp2_term_pr1_rhs",
    "wp2_term_pr_dfsn_rhs",
    "wp3_term_ta_new_pdf_lhs",
    "wp3_term_ta_ADG1_lhs",
    "wp3_term_tp_lhs",
    "wp3_terms_ac_pr2_lhs",
    "wp3_term_pr1_lhs",
    "wp3_term_ta_explicit_rhs",
    "wp3_terms_bp1_pr2_rhs",
    "wp3_term_pr_turb_rhs",
    "wp3_term_pr_dfsn_rhs",
    "wp3_term_pr1_rhs",
]
