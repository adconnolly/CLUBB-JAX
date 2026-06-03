"""Python port of src/CLUBB_core/advance_clubb_core_module.F90.

Translates the Fortran advance_clubb_core subroutine into Python/NumPy,
calling individual Fortran subroutines via the F2PY API for the complex
advance/closure routines, and using NumPy for the inline array math.
"""

import dataclasses
import os
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset, _is_tracer_arg, _safe_sqrt


def _capture_core_kwargs(kw):
    """Env-gated one-shot capture of advance_clubb_core's kwargs (REFACTOR B4 stage 1).

    With CLUBB_CAPTURE_KWARGS=<path> set, pickle the first step's full kwarg set so the
    differentiable pure-functional core can be built and validated bit-exactly against this
    numpy orchestration (and gradded) without a live run. Drops the non-picklable stats_writer
    and forces l_sample=False so the replay skips the diagnostic path. Dormant (one `os.environ.get`)
    when the env var is unset."""
    import pickle
    path = os.environ["CLUBB_CAPTURE_KWARGS"]
    if os.path.exists(path):          # one-shot: capture only the first call
        return
    kw = dict(kw)
    kw["stats_writer"] = None
    kw["l_sample"] = False
    try:
        with open(path, "wb") as f:
            pickle.dump(kw, f)
        print(f"[CLUBB_CAPTURE_KWARGS] wrote {len(kw)} kwargs -> {path}")
    except Exception as e:
        print(f"[CLUBB_CAPTURE_KWARGS] pickle failed: {e}")

# clubb_api (f90-wrapped Fortran) is imported LAZILY inside the pre-advance PDF block only — the
# default (post-advance) path is pure JAX, so the driver imports clean of clubb_python and runs
# standalone for the faithful cases. Only ipdf_pre_advance/pre_post placement (non-default) needs it.
from clubb_jax.src.CLUBB_core.grid_class import zm2zt, zt2zm, ddzt, zm2zt2zm, zm2zt_jax, zt2zm_jax, zt2zm2zt
from clubb_jax.src.CLUBB_core.diffusion import (
    diffusion_zt_lhs_jax, diffusion_zm_lhs_jax,
    term_ma_zm_lhs_jax, xpyp_term_ta_pdf_lhs_jax,
    xpyp_term_ta_pdf_rhs_jax,
    xpyp_term_ta_pdf_lhs_upwind_jax, xpyp_term_ta_pdf_rhs_upwind_jax,
    term_dp1_lhs_jax, xp2_xpyp_lhs_jax,
    term_dp1_rhs_jax, xp2_xpyp_rhs_jax,
)
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_lu_solve_jax
from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import (
    advance_xm_wpxp_jax,
    clip_covar_jax,
    compute_shared_xm_wpxp_lhs_terms,
    wpxp_term_pr1_lhs_jax,
    wpxp_terms_bp_pr3_rhs_jax,
)
from clubb_jax.src.CLUBB_core.sponge_layer_damping import sponge_damp_xm
from clubb_jax.src.CLUBB_core.mono_flux_limiter import (
    monotonic_turbulent_flux_limit, monotonic_turbulent_flux_limit_jax, calc_turb_adv_range,
    MFL_UM, MFL_VM, MFL_RTM, MFL_THLM,
)
from clubb_jax.src.CLUBB_core.constants_clubb import (
    w_tol as _W_TOL, w_tol_sqd as _W_TOL_SQD,
    rt_tol as _RT_TOL, thl_tol as _THL_TOL,
    rt_tol_mfl as _RT_TOL_MFL, thl_tol_mfl as _THL_TOL_MFL,
)
from clubb_jax.src.CLUBB_core.advance_wp2_wp3_module import advance_wp2_wp3_jax
from clubb_jax.src.CLUBB_core.advance_windm_edsclrm_module import advance_windm_edsclrm_jax
from clubb_jax.src.CLUBB_core.advance_xp3_module import compute_xp3_jax, skx_func_jax, advance_xp3_jax
from clubb_jax.src.CLUBB_core.T_in_K_module import calculate_thvm_jax
from clubb_jax.src.CLUBB_core.advance_helper_module import (
    calc_ri_zm_jax,
    compute_cx_fnc_richardson_jax,
    calc_stability_correction_jax,
    smooth_max_jax,
)
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq_jax, sat_mixrat_ice_jax
from clubb_jax.src.CLUBB_core.sfc_varnce_module import calc_sfc_varnce_jax
from clubb_jax.src.CLUBB_core.sigma_sqd_w_module import compute_sigma_sqd_w_jax
from clubb_jax.src.CLUBB_core.advance_helper_module import calc_brunt_vaisala_freq_sqd_jax
from clubb_jax.src.CLUBB_core.mixing_length import (
    diagnose_lscale_from_tau_jax,
    calc_lscale_directly_jax,
)
from clubb_jax.src.CLUBB_core.advance_helper_module import wp23_term_splat_lhs_jax
from clubb_jax.src.CLUBB_core.fill_holes import fill_holes_vertical_jax
from clubb_jax.src.CLUBB_core.clip_explicit import (
    clip_variance_jax,
    clip_skewness_jax,
    clip_covars_denom_jax,
    clip_rcm_jax,
    fill_holes_wp2_from_horz_tke_jax,
)
from clubb_jax.src.CLUBB_core.numerical_check import parameterization_check_jax
from clubb_jax.src.CLUBB_core.adg1_adg2_3d_luhar_pdf import (
    ADG1_pdf_driver_jax, calc_comp_corrs_binormal_jax, calc_wp2xp_pdf_jax,
    calc_wpxp2_pdf_jax, calc_wp2xp2_pdf_jax, calc_wp4_pdf_jax,
    calc_wpxpyp_pdf_jax,
)
from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp,
    em_min,
    ep,
    ep1,
    ep2,
    eps,
    grav,
    Lv,
    max_mag_correlation,
    max_num_stdevs,
    chi_tol,
    T_freeze_K,
    sqrt_2,
    sqrt_2pi,
    Rd,
    l_smooth_min_max,
    min_max_smth_mag,
    one,
    rt_tol,
    thl_tol,
    three_halves,
    two,
    unused_var,
    w_tol,
    w_tol_sqd,
    zero,
    zero_threshold,
    # Parameter indices
    ia3_coef_min,
    ia_const,
    ibeta,
    ibv_efold,
    ic_K,
    ic_K1,
    ic_K8,
    ic_K2,
    ic_K6,
    iSkw_max_mag,
    ic_K9,
    ic_K10,
    ic_K10h,
    iC4,
    iC14,
    iC_uu_shr,
    iC_uu_buoy,
    iC_wp2_splat,
    iC2rt,
    iC6rt,
    iC6thl,
    igamma_coef,
    igamma_coefb,
    igamma_coefc,
    ilambda0_stability_coef,
    imu,
    itaumax,
    iup2_sfc_coef,
    ixp3_coef_base,
    ixp3_coef_slope,
    iC_invrs_tau_bkgnd,
    iC_invrs_tau_sfc,
    iC_invrs_tau_shear,
    iC_invrs_tau_N2,
    iC_invrs_tau_N2_wp2,
    iC_invrs_tau_N2_xp2,
    iC_invrs_tau_N2_wpxp,
    iC_invrs_tau_N2_clear_wp3,
    iC_invrs_tau_wpxp_Ri,
    iC_invrs_tau_wpxp_N2_thresh,
    ialtitude_threshold,
    iwpxp_Ri_exp,
    iz_displace,
    # Model flag constants
    iiPDF_ADG1,
    ipdf_post_advance_fields,
    ipdf_pre_advance_fields,
    ipdf_pre_post_advance_fields,
    smth_type,
    tau_const,
    ufmin,
    below_grnd_val,
)

CLUBB_FATAL_ERROR = 99


@dataclasses.dataclass
class PdfParamsJAX:
    """Drop-in replacement for Fortran pdf_params, populated from JAX ADG1 driver."""
    mixt_frac: np.ndarray
    w_1: np.ndarray
    w_2: np.ndarray
    varnce_w_1: np.ndarray
    varnce_w_2: np.ndarray
    rt_1: np.ndarray
    rt_2: np.ndarray
    varnce_rt_1: np.ndarray
    varnce_rt_2: np.ndarray
    alpha_rt: np.ndarray
    thl_1: np.ndarray
    thl_2: np.ndarray
    varnce_thl_1: np.ndarray
    varnce_thl_2: np.ndarray
    alpha_thl: np.ndarray
    corr_rt_thl_1: np.ndarray
    corr_rt_thl_2: np.ndarray


# Iter34 stats: JAX vs Fortran for complete pdf_closure_driver outputs


# Iter43 stats: clip_covar_jax vs Fortran for rtpthlp (Cauchy-Schwarz)


# Accumulates per-timestep max |JAX - Fortran| for diffusion + solver (Iteration 4)

# Call counter for term_ma_zm_lhs (Iteration 5; Fortran oracle removed Iter53)

# Call counter for xpyp_term_ta_pdf_lhs_jax (Iteration 6; Fortran oracle removed Iter51)

# Accumulates max |JAX - numpy_reference| for term_dp1_lhs (Iteration 7)

# Accumulates max |JAX - numpy_reference| for xp2_xpyp_lhs assembly (Iteration 8)

# Accumulates max |JAX - numpy_reference| for xp2_xpyp_rhs (Iteration 9)

# advance_xp2_xpyp call count (Fortran oracle removed Iter47)

# Accumulates max |JAX - Fortran| for full advance_xm_wpxp solve (Iteration 11)

# Accumulates max |JAX - Fortran| for full advance_wp2_wp3 solve (Iteration 12)


_prev_adg1_j25 = None  # carries Block U ADG1 result across timesteps for Block P override


def reset_clubb_core_state():
    """Reset the cross-timestep module state so a fresh run starts clean (reentrancy/composability).
    Must be called at case init — otherwise a second case in the same process inherits the first
    case's `_prev_adg1_j25` (wrong grid shape → broadcast error). The Fortran reuses stack locals
    between calls, which is why this state lives at module scope rather than in `state`."""
    global _prev_adg1_j25
    _prev_adg1_j25 = None


_RC_TOL = 1.0e-6       # Tolerance for cloud water mixing ratio [kg/kg]
_CLOUD_FRAC_MIN = 0.005


def _vertical_avg(rho_ds, field, dz):
    """Density-weighted vertical average. All args shape (nz,)."""
    denom = _xp.sum(rho_ds * dz)
    if denom == 0.0:
        return 0.0
    return _xp.sum(rho_ds * dz * field) / denom


def _vertical_integral(rho_ds, field, dz):
    """Vertical integral. All args shape (nz,)."""
    return _xp.sum(field * rho_ds * dz)


def _calculate_spurious_source(integral_after, integral_before,
                               flux_top, flux_sfc, integral_forcing, dt):
    return ((integral_after - integral_before) / dt
            + flux_top - flux_sfc - integral_forcing)


def _stats_accumulate_py(sw, *, nzm, nzt, ngrdcol, dt, gr,
                          l_implemented, l_host_applies_sfc_fluxes,
                          l_stability_correct_tau_zm,
                          um, vm, upwp, vpwp, up2, vp2,
                          thlm, rtm, thlm_before, rtm_before,
                          thlm_forcing, rtm_forcing,
                          wpthlp_sfc, wprtp_sfc, wprtp, wpthlp,
                          wp2, wp3, rtp2, rtp3, thlp2, thlp3, rtpthlp,
                          p_in_Pa, exner, rho, rho_zm, rho_ds_zm, rho_ds_zt,
                          thv_ds_zm, thv_ds_zt, wm_zt, wm_zm, rcm, cloud_frac,
                          ice_supersat_frac,
                          thvm, ug, vg, ddzt_umvm_sqd, stability_correction,
                          Kh_zt, rsat, Kh_zm, em, wp3_on_wp2, wp3_on_wp2_zt,
                          sclrm, sclrp2, sclrprtp, sclrpthlp, sclrm_forcing,
                          wpsclrp, wpedsclrp, edsclrm, edsclrm_forcing,
                          saturation_formula, sclr_dim=0, edsclr_dim=0):
    """Python equivalent of Fortran stats_accumulate in stats_clubb_utilities.F90.

    Calls sw.update(name, val) for every variable written by the Fortran routine.
    Only called when sw.l_sample is True.
    """
    if sw is None:
        return

    # Compute grid layer thicknesses
    grid_dir = float(gr.grid_dir)
    # dzt[i,k] = (zm[i,k+1] - zm[i,k]) * grid_dir  (shape: ngrdcol, nzt)
    dzt = (_asarray(gr.zm)[:, 1:] - _asarray(gr.zm)[:, :-1]) * grid_dir
    # dzm: shape (ngrdcol, nzm)
    dzm = _asarray(gr.dzm) * grid_dir

    # ---- Computed diagnostics ----

    # T_in_K: absolute temperature  T = thlm * exner + (Lv/Cp) * rcm
    if sw.var_on_stats_list("T_in_K") or sw.var_on_stats_list("rsati"):
        T_in_K_acc = thlm * exner + (Lv / Cp) * rcm
        sw.update("T_in_K", T_in_K_acc)
        if sw.var_on_stats_list("rsati"):
            _rsati_acc = _asarray(
                sat_mixrat_ice_jax(jnp.asarray(p_in_Pa), jnp.asarray(T_in_K_acc)),
                dtype=np.float64,
            )
            sw.update("rsati", _rsati_acc)

    # rcm_in_cloud
    if sw.var_on_stats_list("rcm_in_cloud"):
        rcm_in_cloud = _xp.where(cloud_frac > _CLOUD_FRAC_MIN, rcm / cloud_frac, rcm)
        sw.update("rcm_in_cloud", rcm_in_cloud)

    # shear (zm-level)
    if sw.var_on_stats_list("shear"):
        shear = np.zeros((ngrdcol, nzm), dtype=np.float64)
        # Interior zm levels k=1..nzm-2
        um_diff = um[:, 1:] - um[:, :-1]  # (ngrdcol, nzm-2)
        vm_diff = vm[:, 1:] - vm[:, :-1]
        invrs_dzm_int = _asarray(gr.invrs_dzm)[:, 1:-1]
        shear[:, 1:-1] = (- upwp[:, 1:-1] * um_diff * invrs_dzm_int
                          - vpwp[:, 1:-1] * vm_diff * invrs_dzm_int)
        sw.update("shear", shear)

    # zt-level profile variables (unconditional)
    sw.update("thlm", thlm)
    sw.update("thvm", thvm)
    sw.update("rtm", rtm)
    sw.update("rcm", rcm)
    # Cloud diagnostics (Fortran writes these; their absence makes the diagnostic
    # comparison show a spurious 0 for every cloud case and hides real cloud-PDF
    # divergences during debugging). cloud_frac/ice_supersat_frac are the physics
    # state values, so this also lets the compare verify them.
    sw.update("cloud_frac", cloud_frac)
    sw.update("ice_supersat_frac", ice_supersat_frac)
    sw.update("um", um)
    sw.update("vm", vm)
    sw.update("wm_zt", wm_zt)
    sw.update("ug", ug)
    sw.update("vg", vg)
    sw.update("p_in_Pa", p_in_Pa)
    sw.update("exner", exner)
    sw.update("rho_ds_zt", rho_ds_zt)
    sw.update("thv_ds_zt", thv_ds_zt)
    sw.update("wp3", wp3)
    sw.update("Kh_zt", Kh_zt)
    sw.update("rho", rho)
    sw.update("rsat", rsat)
    sw.update("thlp3", thlp3)
    sw.update("rtp3", rtp3)
    sw.update("wp3_on_wp2_zt", wp3_on_wp2_zt)

    # Scalar species (zt)
    for s in range(sclr_dim):
        sw.update(f"sclr{s+1}m", sclrm[:, :, s])
        sw.update(f"sclr{s+1}m_f", sclrm_forcing[:, :, s])
    for e in range(edsclr_dim):
        sw.update(f"edsclr{e+1}m", edsclrm[:, :, e])
        sw.update(f"edsclr{e+1}m_f", edsclrm_forcing[:, :, e])

    # zm-level profile variables (unconditional)
    sw.update("wm_zm", wm_zm)
    sw.update("ddzt_umvm_sqd", ddzt_umvm_sqd)
    sw.update("wp2", wp2)
    sw.update("rtp2", rtp2)
    sw.update("thlp2", thlp2)
    sw.update("rtpthlp", rtpthlp)
    sw.update("wprtp", wprtp)
    sw.update("wpthlp", wpthlp)
    if l_stability_correct_tau_zm:
        sw.update("stability_correction", stability_correction)
    sw.update("Kh_zm", Kh_zm)
    sw.update("upwp", upwp)
    sw.update("vpwp", vpwp)
    sw.update("vp2", vp2)
    sw.update("up2", up2)
    sw.update("rho_zm", rho_zm)
    sw.update("rho_ds_zm", rho_ds_zm)
    sw.update("thv_ds_zm", thv_ds_zm)
    sw.update("em", em)
    sw.update("wp3_on_wp2", wp3_on_wp2)
    # CFL number: wp3_on_wp2 * dt / dzm
    if sw.var_on_stats_list("wp3_on_wp2_cfl_num"):
        sw.update("wp3_on_wp2_cfl_num", wp3_on_wp2 * dt / dzm)

    # Scalar species (zm)
    for s in range(sclr_dim):
        sw.update(f"sclr{s+1}p2", sclrp2[:, :, s])
        sw.update(f"sclr{s+1}prtp", sclrprtp[:, :, s])
        sw.update(f"sclr{s+1}pthlp", sclrpthlp[:, :, s])
        sw.update(f"wpsclr{s+1}p", wpsclrp[:, :, s])
    for e in range(edsclr_dim):
        sw.update(f"wpedsclr{e+1}p", wpedsclrp[:, :, e])

    # Surface / scalar variables
    zt_arr = _asarray(gr.zt)  # (ngrdcol, nzt)

    # cc: max cloud fraction over zt column
    for i in range(ngrdcol):
        sw.update_col("cc", float(_xp.max(cloud_frac[i, :])), icol=i)

    # z_cloud_base
    if sw.var_on_stats_list("z_cloud_base"):
        for i in range(ngrdcol):
            k = 0
            while rcm[i, k] < _RC_TOL and k < nzt - 1:
                k += 1
            if k == 0:
                z_cb = float(zt_arr[i, 0])
            elif k < nzt - 1:
                # linear interpolation: where rcm crosses rc_tol
                rcm_k, rcm_km1 = float(rcm[i, k]), float(rcm[i, k - 1])
                zt_k, zt_km1 = float(zt_arr[i, k]), float(zt_arr[i, k - 1])
                if abs(rcm_k - rcm_km1) > 1.0e-30:
                    z_cb = ((_RC_TOL - rcm_km1) / (rcm_k - rcm_km1)
                            * (zt_k - zt_km1) + zt_km1)
                else:
                    z_cb = zt_k
            else:
                z_cb = -10.0
            sw.update_col("z_cloud_base", z_cb, icol=i)

    # lwp, vwp (optional)
    if sw.var_on_stats_list("lwp"):
        for i in range(ngrdcol):
            sw.update_col("lwp",
                           _vertical_integral(rho_ds_zt[i], rcm[i], dzt[i]),
                           icol=i)
    if sw.var_on_stats_list("vwp"):
        for i in range(ngrdcol):
            sw.update_col("vwp",
                           _vertical_integral(rho_ds_zt[i], rtm[i] - rcm[i], dzt[i]),
                           icol=i)

    # Density-weighted vertical averages
    for i in range(ngrdcol):
        sw.update_col("thlm_vert_avg",
                       _vertical_avg(rho_ds_zt[i], thlm[i], dzt[i]), icol=i)
        sw.update_col("rtm_vert_avg",
                       _vertical_avg(rho_ds_zt[i], rtm[i], dzt[i]), icol=i)
        sw.update_col("um_vert_avg",
                       _vertical_avg(rho_ds_zt[i], um[i], dzt[i]), icol=i)
        sw.update_col("vm_vert_avg",
                       _vertical_avg(rho_ds_zt[i], vm[i], dzt[i]), icol=i)
        sw.update_col("wp2_vert_avg",
                       _vertical_avg(rho_ds_zm[i], wp2[i], dzm[i]), icol=i)
        sw.update_col("up2_vert_avg",
                       _vertical_avg(rho_ds_zm[i], up2[i], dzm[i]), icol=i)
        sw.update_col("vp2_vert_avg",
                       _vertical_avg(rho_ds_zm[i], vp2[i], dzm[i]), icol=i)
        sw.update_col("rtp2_vert_avg",
                       _vertical_avg(rho_ds_zm[i], rtp2[i], dzm[i]), icol=i)
        sw.update_col("thlp2_vert_avg",
                       _vertical_avg(rho_ds_zm[i], thlp2[i], dzm[i]), icol=i)

    # Normalized total variation
    if sw.var_on_stats_list("tot_vartn_normlzd_rtm"):
        for i in range(ngrdcol):
            span = abs(rtm[i, -1] - rtm[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(rtm[i, 1:] - rtm[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_rtm", val, icol=i)

    if sw.var_on_stats_list("tot_vartn_normlzd_thlm"):
        for i in range(ngrdcol):
            span = abs(thlm[i, -1] - thlm[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(thlm[i, 1:] - thlm[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_thlm", val, icol=i)

    if sw.var_on_stats_list("tot_vartn_normlzd_wprtp"):
        for i in range(ngrdcol):
            span = abs(wprtp[i, -1] - wprtp[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(wprtp[i, 1:] - wprtp[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_wprtp", val, icol=i)

    # Spurious source (rtm and thlm conservation check)
    k_ub = int(gr.k_ub_zm)  # upper boundary zm index (Python 0-based)
    k_lb = int(gr.k_lb_zm)  # lower boundary zm index (Python 0-based)
    for i in range(ngrdcol):
        if (l_implemented or
                (np.all(_xp.abs(wm_zt[i]) < eps) and np.all(_xp.abs(wm_zm[i]) < eps))):
            rtm_flux_top = float(rho_ds_zm[i, k_ub] * wprtp[i, k_ub])
            if not l_host_applies_sfc_fluxes:
                rtm_flux_sfc = float(rho_ds_zm[i, k_lb] * wprtp_sfc[i])
            else:
                rtm_flux_sfc = 0.0
            rtm_int_before = _vertical_integral(rho_ds_zt[i], rtm_before[i], dzt[i])
            rtm_int_after = _vertical_integral(rho_ds_zt[i], rtm[i], dzt[i])
            rtm_int_forcing = _vertical_integral(rho_ds_zt[i], rtm_forcing[i], dzt[i])
            rtm_spur = _calculate_spurious_source(
                rtm_int_after, rtm_int_before, rtm_flux_top, rtm_flux_sfc,
                rtm_int_forcing, dt)

            thlm_flux_top = float(rho_ds_zm[i, k_ub] * wpthlp[i, k_ub])
            if not l_host_applies_sfc_fluxes:
                thlm_flux_sfc = float(rho_ds_zm[i, k_lb] * wpthlp_sfc[i])
            else:
                thlm_flux_sfc = 0.0
            thlm_int_before = _vertical_integral(rho_ds_zt[i], thlm_before[i], dzt[i])
            thlm_int_after = _vertical_integral(rho_ds_zt[i], thlm[i], dzt[i])
            thlm_int_forcing = _vertical_integral(rho_ds_zt[i], thlm_forcing[i], dzt[i])
            thlm_spur = _calculate_spurious_source(
                thlm_int_after, thlm_int_before, thlm_flux_top, thlm_flux_sfc,
                thlm_int_forcing, dt)
        else:
            rtm_spur = -9999.0
            thlm_spur = -9999.0
        sw.update_col("rtm_spur_src", rtm_spur, icol=i)
        sw.update_col("thlm_spur_src", thlm_spur, icol=i)


def _apply_sponge_field(key, xm, xm_ref, gr, dt_advance, sponge_cfg):
    """Apply sponge-layer damping to a mean field toward its reference profile.

    Faithful to the sponge block at the end of advance_xm_wpxp
    (advance_xm_wpxp_module.F90:1053-1123). A no-op unless `sponge_cfg` contains
    `key` (i.e. that field's l_sponge_damping is set). tau/depth are precomputed
    once at init (sponge_layer_damping.initialize_tau_sponge_damp). The reference
    profile xm_ref is the initial sounding profile (clubb_driver.F90:5298-5316).
    """
    if not sponge_cfg or key not in sponge_cfg:
        return xm
    prof = sponge_cfg[key]
    tau, depth = prof['tau'], prof['depth']
    zt_a = _asarray(gr.zt, dtype=np.float64)
    zm_a = _asarray(gr.zm, dtype=np.float64)
    ref_a = _asarray(xm_ref, dtype=np.float64)
    dt = float(dt_advance)
    # Vectorized + tracer-transparent (REFACTOR B5): sponge_damp_xm is now pure broadcast arithmetic
    # (tau (nz,) broadcasts over the (ngrdcol, nz) field), so this is bit-identical to the per-column loop
    # while keeping the prognostic xm on the autodiff graph (the old `np.array(xm)`+in-place loop severed it).
    return sponge_damp_xm(_asarray(xm, dtype=np.float64), ref_a, zt_a, zm_a[:, -1:], tau, depth, dt)


def advance_clubb_core(
    *,
    gr,
    nzm,
    nzt,
    ngrdcol,
    dt_main,
    flags,
    sclr_dim,
    edsclr_dim,
    hydromet_dim,
    clubb_params,
    fcor,
    fcor_y,
    host_dx,
    host_dy,
    wm_zm,
    wm_zt,
    rho_ds_zt,
    rtm,
    thlm,
    rho,
    rfrzm,
    sfc_elevation,
    upwp_sfc,
    vpwp_sfc,
    wpthlp,
    wprtp_sfc,
    upwp,
    vpwp,
    upwp_sfc_pert,
    vpwp_sfc_pert,
    wpsclrp,
    wpedsclrp_sfc,
    p_sfc,
    thv_ds_zm,
    thv_ds_zt,
    wp2,
    wp3,
    thlp2,
    rtp2,
    rtpthlp,
    um,
    vm,
    p_in_Pa,
    exner,
    rcm,
    ice_supersat_frac,
    up2,
    vp2,
    wprtp,
    wpthlp_sfc,
    wp2thvp,
    wp2up,
    rtpthvp,
    thlpthvp,
    wpthvp,
    wphydrometp,
    wp2hmp,
    rtphmp_zt,
    thlphmp_zt,
    lmin,
    mixt_frac_max_mag,
    T0,
    ts_nudge,
    rtm_min,
    rtm_nudge_max_altitude,
    um_forcing,
    vm_forcing,
    thlm_forcing,
    rtm_forcing,
    wprtp_forcing,
    wpthlp_forcing,
    rtp2_forcing,
    thlp2_forcing,
    rtpthlp_forcing,
    err_info,
    sclr_tol,
    thlm_ref,
    rtm_ref,
    um_ref,
    vm_ref,
    ug,
    vg,
    sclrm_forcing,
    edsclrm_forcing,
    sclrp2,
    sclrprtp,
    sclrpthlp,
    sclr_idx,
    pdf_params,
    pdf_params_zm,
    pdf_implicit_coefs_terms,
    nu_vert_res_dep,
    sclrm,
    sclrpthvp,
    up3,
    vp3,
    um_pert,
    vm_pert,
    uprcp,
    vprcp,
    rc_coef_zm,
    wp4,
    wpup2,
    wpvp2,
    wp2up2,
    wp2vp2,
    wp2rtp,
    wp2thlp,
    upwp_pert,
    vpwp_pert,
    sclrp3,
    cloud_frac,
    thlp3,
    rtp3,
    edsclrm,
    wpsclrp_sfc,
    l_mix_rat_hm,
    rho_ds_zm,
    invrs_rho_ds_zm,
    invrs_rho_ds_zt,
    rho_zm,
    l_sample=False,
    l_gamma_Skw=True,
    l_advance_xp3=False,
    l_use_invrs_tau_N2_iso=False,
    order_xm_wpxp=1,
    order_xp2_xpyp=2,
    order_wp2_wp3=3,
    order_windm=4,
    debug_level=0,
    stats_writer=None,
    wprtp2_carry=None,
    wpthlp2_carry=None,
    wprtpthlp_carry=None,
    sponge_cfg=None,
):
    """Advance CLUBB one timestep with an explicit argument surface."""
    if os.environ.get("CLUBB_CAPTURE_KWARGS"):
        _capture_core_kwargs(dict(locals()))   # B4 stage 1: capture the full kwarg fixture (dormant otherwise)
    shzt = (ngrdcol, nzt)
    shzm = (ngrdcol, nzm)
    shzts = (ngrdcol, nzt, max(sclr_dim, 1))
    shzms = (ngrdcol, nzm, max(sclr_dim, 1))
    Kh_zm = np.zeros(shzm)
    Kh_zt = np.zeros(shzt)
    Lscale = np.zeros(shzt)
    invrs_tau_zm = np.zeros(shzm)
    _cloud_frac_zm = np.zeros(shzm)
    _ice_supersat_frac_zm = np.zeros(shzm)
    _rc_coef = np.zeros(shzt)
    _rcm_supersat_adj = np.zeros(shzt)
    _rcm_zm = np.zeros(shzm)
    _rcp2 = np.zeros(shzm)
    _rcp2_zt = np.zeros(shzt)
    _rtm_zm = np.zeros(shzm)
    _rtprcp = np.zeros(shzm)
    _sclrprcp = np.zeros(shzms)
    _sigma_sqd_w = np.zeros(shzm)
    _skw_velocity = np.zeros(shzm)
    _thlm_zm = np.zeros(shzm)
    _wp2rcp = np.zeros(shzt)
    _wp2sclrp = np.zeros(shzts)
    # For ipdf_post_advance_fields, these carry the previous timestep's post-advance
    # pdf_closure output — Fortran stack reuse means locals persist between calls.
    _wprtp2    = _asarray(wprtp2_carry,    dtype=np.float64) if wprtp2_carry    is not None else np.zeros(shzt)
    _wprtpthlp = _asarray(wprtpthlp_carry, dtype=np.float64) if wprtpthlp_carry is not None else np.zeros(shzt)
    _wpsclrp2 = np.zeros(shzts)
    _wpsclrprtp = np.zeros(shzts)
    _wpsclrpthlp = np.zeros(shzts)
    _wpthlp2   = _asarray(wpthlp2_carry,   dtype=np.float64) if wpthlp2_carry   is not None else np.zeros(shzt)
    cloud_cover = np.zeros(shzt)
    cloudy_downdraft_frac = np.zeros(shzt)
    cloudy_updraft_frac = np.zeros(shzt)
    rcm_in_layer = np.zeros(shzt)
    thlprcp = np.zeros(shzm)
    w_down_in_cloud = np.zeros(shzt)
    w_up_in_cloud = np.zeros(shzt)
    wprcp_out = np.zeros(shzm)
    rsat = np.zeros(shzt)
    dt = dt_main
    l_gamma_skw = l_gamma_Skw
    l_advance_xp3_flag = l_advance_xp3
    l_use_invrs_tau_n2_iso = l_use_invrs_tau_N2_iso
    order_xm_wpxp_val = order_xm_wpxp
    order_xp2_xpyp_val = order_xp2_xpyp
    order_wp2_wp3_val = order_wp2_wp3
    order_windm_val = order_windm

    l_implemented = False  # standalone mode
    global _prev_adg1_j25  # cross-timestep ADG1 state for Block P override (iter40)

    # ================================================================== #
    # Block A: Setup
    # ================================================================== #
    dt_advance = two * dt if flags.l_lmm_stepping else dt

    # Iter56: set_lscale_max replaced with pure Python
    if l_implemented:
        Lscale_max = 0.25 * _xp.minimum(_asarray(host_dx), _asarray(host_dy))
    else:
        Lscale_max = np.full(ngrdcol, 1.0e5, dtype=np.float64)

    # ================================================================== #
    # Block B: Stats — spurious source pre-integration
    # ================================================================== #
    thlm_before = thlm.copy()
    rtm_before = rtm.copy()

    if debug_level >= 2:
        err_info = parameterization_check_jax(
            err_info=err_info,
            nzm=nzm, nzt=nzt, ngrdcol=ngrdcol, sclr_dim=sclr_dim, edsclr_dim=edsclr_dim,
            thlm_forcing=thlm_forcing, rtm_forcing=rtm_forcing,
            um_forcing=um_forcing, vm_forcing=vm_forcing,
            wm_zm=wm_zm, wm_zt=wm_zt, p_in_Pa=p_in_Pa,
            rho_zm=rho_zm, rho=rho, exner=exner,
            rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
            invrs_rho_ds_zm=invrs_rho_ds_zm, invrs_rho_ds_zt=invrs_rho_ds_zt,
            thv_ds_zm=thv_ds_zm, thv_ds_zt=thv_ds_zt,
            wpthlp_sfc=wpthlp_sfc, wprtp_sfc=wprtp_sfc,
            upwp_sfc=upwp_sfc, vpwp_sfc=vpwp_sfc, p_sfc=p_sfc,
            um=um, upwp=upwp, vm=vm, vpwp=vpwp,
            up2=up2, vp2=vp2, rtm=rtm, wprtp=wprtp,
            thlm=thlm, wpthlp=wpthlp, wp2=wp2, wp3=wp3,
            rtp2=rtp2, thlp2=thlp2, rtpthlp=rtpthlp,
            prefix="beginning of ", wpsclrp_sfc=wpsclrp_sfc, wpedsclrp_sfc=wpedsclrp_sfc,
            sclrm=sclrm, wpsclrp=wpsclrp, sclrp2=sclrp2,
            sclrprtp=sclrprtp, sclrpthlp=sclrpthlp,
            sclrm_forcing=sclrm_forcing, edsclrm=edsclrm,
            edsclrm_forcing=edsclrm_forcing,
        )
        err_code = err_info.err_code
        if err_code is not None and np.any(_asarray(err_code) == CLUBB_FATAL_ERROR):
            return

    # ================================================================== #
    # Block D: Stats — begin budget tracking
    # ================================================================== #
    if l_sample and stats_writer is not None:
        stats_writer.update("rfrzm", rfrzm)
        stats_writer.begin_budget("wp2_bt", wp2 / dt)
        stats_writer.begin_budget("vp2_bt", vp2 / dt)
        stats_writer.begin_budget("up2_bt", up2 / dt)
        stats_writer.begin_budget("wprtp_bt", wprtp / dt)
        stats_writer.begin_budget("wpthlp_bt", wpthlp / dt)
        if flags.l_predict_upwp_vpwp:
            stats_writer.begin_budget("upwp_bt", upwp / dt)
            stats_writer.begin_budget("vpwp_bt", vpwp / dt)
        stats_writer.begin_budget("rtp2_bt", rtp2 / dt)
        stats_writer.begin_budget("thlp2_bt", thlp2 / dt)
        stats_writer.begin_budget("rtpthlp_bt", rtpthlp / dt)
        stats_writer.begin_budget("rtm_bt", rtm / dt)
        stats_writer.begin_budget("thlm_bt", thlm / dt)
        stats_writer.begin_budget("um_bt", um / dt)
        stats_writer.begin_budget("vm_bt", vm / dt)
        stats_writer.begin_budget("wp3_bt", wp3 / dt)

    # ================================================================== #
    # Block E: Set surface boundary conditions
    # ================================================================== #
    k_lb = gr.k_lb_zm  # 0-based lower boundary for momentum levels

    if not flags.l_host_applies_sfc_fluxes:
        wpthlp = _iset(wpthlp, np.s_[:, k_lb], wpthlp_sfc)
        wprtp = _iset(wprtp, np.s_[:, k_lb], wprtp_sfc)
        upwp = _iset(upwp, np.s_[:, k_lb], upwp_sfc)
        vpwp = _iset(vpwp, np.s_[:, k_lb], vpwp_sfc)

        if flags.l_linearize_pbl_winds:
            upwp_pert = _iset(upwp_pert, np.s_[:, k_lb], upwp_sfc_pert)
            vpwp_pert = _iset(vpwp_pert, np.s_[:, k_lb], vpwp_sfc_pert)

        if sclr_dim > 0:
            for sclr in range(sclr_dim):
                wpsclrp = _iset(wpsclrp, np.s_[:, k_lb, sclr], wpsclrp_sfc[:, sclr])

        if edsclr_dim > 0:
            wpedsclrp = np.zeros((ngrdcol, nzm, edsclr_dim))
            for edsclr in range(edsclr_dim):
                wpedsclrp = _iset(wpedsclrp, np.s_[:, k_lb, edsclr], wpedsclrp_sfc[:, edsclr])
        else:
            wpedsclrp = np.zeros((ngrdcol, nzm, max(edsclr_dim, 1)))
    else:
        wpthlp = _iset(wpthlp, np.s_[:, k_lb], 0.0)
        wprtp = _iset(wprtp, np.s_[:, k_lb], 0.0)
        upwp = _iset(upwp, np.s_[:, k_lb], 0.0)
        vpwp = _iset(vpwp, np.s_[:, k_lb], 0.0)

        if sclr_dim > 0:
            for sclr in range(sclr_dim):
                wpsclrp = _iset(wpsclrp, np.s_[:, k_lb, sclr], 0.0)

        if edsclr_dim > 0:
            wpedsclrp = np.zeros((ngrdcol, nzm, edsclr_dim))
        else:
            wpedsclrp = np.zeros((ngrdcol, nzm, max(edsclr_dim, 1)))

    # ================================================================== #
    # Block F: Set mu
    # ================================================================== #
    # Standalone mode: mu from tunable parameters (no CLUBBND_CAM)
    # Parameter indices are 1-based into clubb_params(:, nparams)
    mu = clubb_params[:, imu - 1].copy()

    # ================================================================== #
    # Block G: Pre-advance PDF closure (conditional)
    # ================================================================== #
    if (flags.ipdf_call_placement == ipdf_pre_advance_fields
            or flags.ipdf_call_placement == ipdf_pre_post_advance_fields):

        l_samp_stats = True  # always sample for pre or pre_post

        from clubb_python import clubb_api  # lazy: only pre-advance PDF placement needs Fortran
        pdf_result = clubb_api.pdf_closure_driver(
            gr=gr, nzm=nzm, nzt=nzt, ngrdcol=ngrdcol,
            dt=dt, hydromet_dim=hydromet_dim, sclr_dim=sclr_dim,
            sclr_tol=sclr_tol,
            wprtp=wprtp, thlm=thlm, wpthlp=wpthlp,
            rtp2=rtp2, rtp3=rtp3,
            thlp2=thlp2, thlp3=thlp3,
            rtpthlp=rtpthlp, wp2=wp2, wp3=wp3,
            wm_zm=wm_zm, wm_zt=wm_zt,
            um=um, up2=up2, upwp=upwp, up3=up3,
            vm=vm, vp2=vp2, vpwp=vpwp, vp3=vp3,
            p_in_pa=p_in_Pa, exner=exner,
            thv_ds_zm=thv_ds_zm, thv_ds_zt=thv_ds_zt,
            rtm_ref=rtm_ref,
            wphydrometp=wphydrometp,
            wp2hmp=wp2hmp,
            rtphmp_zt=rtphmp_zt,
            thlphmp_zt=thlphmp_zt,
            sclrm=sclrm, wpsclrp=wpsclrp,
            sclrp2=sclrp2, sclrprtp=sclrprtp,
            sclrpthlp=sclrpthlp, sclrp3=sclrp3,
            p_sfc=p_sfc,
            l_samp_stats_in_pdf_call=l_samp_stats,
            mixt_frac_max_mag=mixt_frac_max_mag,
            ts_nudge=ts_nudge,
            rtm_min=rtm_min,
            rtm_nudge_max_altitude=rtm_nudge_max_altitude,
            clubb_params=clubb_params,
            iiPDF_type=flags.iiPDF_type,
            saturation_formula=flags.saturation_formula,
            l_predict_upwp_vpwp=flags.l_predict_upwp_vpwp,
            l_rtm_nudge=flags.l_rtm_nudge,
            l_trapezoidal_rule_zt=flags.l_trapezoidal_rule_zt,
            l_trapezoidal_rule_zm=flags.l_trapezoidal_rule_zm,
            l_call_pdf_closure_twice=flags.l_call_pdf_closure_twice,
            l_use_cloud_cover=flags.l_use_cloud_cover,
            l_rcm_supersat_adj=flags.l_rcm_supersat_adj,
            l_mix_rat_hm=l_mix_rat_hm,
            pdf_params=pdf_params,
            pdf_params_zm=pdf_params_zm,
            pdf_implicit_coefs_terms=pdf_implicit_coefs_terms,
            err_info=err_info,
            rtm=rtm,
        )
        (rtm, pdf_implicit_coefs_terms, pdf_params, pdf_params_zm, err_info,
         rcm, cloud_frac, ice_supersat_frac, wprcp_out, _sigma_sqd_w, wpthvp, wp2thvp,
         wp2up, rtpthvp, thlpthvp, _rc_coef, rcm_in_layer, cloud_cover,
         _rcp2_zt, thlprcp, rc_coef_zm, sclrpthvp, wpup2, wpvp2, wp2up2,
         wp2vp2, wp4, wp2rtp, _wprtp2, wp2thlp, _wpthlp2, _wprtpthlp, _wp2rcp,
         _rtprcp, _rcp2, uprcp, vprcp, w_up_in_cloud, w_down_in_cloud,
         cloudy_updraft_frac, cloudy_downdraft_frac, _skw_velocity,
         _cloud_frac_zm, _ice_supersat_frac_zm, _rtm_zm, _thlm_zm, _rcm_zm,
         _rcm_supersat_adj, _wp2sclrp, _wpsclrp2, _sclrprcp,
         _wpsclrprtp, _wpsclrpthlp) = pdf_result

        # Check for fatal error
        err_code = err_info.err_code
        if err_code is not None and np.any(_asarray(err_code) == CLUBB_FATAL_ERROR):
            return

    # ================================================================== #
    # Block H: Interpolations — wp2_zt, wp3_zm, Skw, sigma_sqd_w, a3_coef
    # ================================================================== #
    wp2 = wp2
    wp3 = wp3

    wp2_zt = _xp.maximum(
        _asarray(zm2zt(wp2, gr)),
        w_tol_sqd,
    )
    wp3_zm = _asarray(zt2zm(wp3, gr))

    # Iter52: JAX-only skx_func (Fortran oracle removed)
    Skw_zt = _asarray(skx_func_jax(jnp.asarray(wp2_zt), jnp.asarray(wp3),
                                      w_tol, jnp.asarray(clubb_params)), dtype=np.float64)
    Skw_zm = _asarray(skx_func_jax(jnp.asarray(wp2), jnp.asarray(wp3_zm),
                                      w_tol, jnp.asarray(clubb_params)), dtype=np.float64)

    sigma_sqd_w = _sigma_sqd_w  # may be set by PDF closure above
    gamma_Skw_fnc = None  # set in pre- or post-advance PDF path below

    # ================================================================== #
    # Iter39: JAX pre-advance sigma_sqd_w (ARM: ipdf_pre_advance path)   #
    # Uses JAX Skw_zm (already overridden by iter38) + pre-advance state. #
    # ================================================================== #
    if flags.ipdf_call_placement in (ipdf_pre_advance_fields, ipdf_pre_post_advance_fields):
        _gc39 = clubb_params[:, igamma_coef - 1]
        _gb39 = clubb_params[:, igamma_coefb - 1]
        _gcf39 = clubb_params[:, igamma_coefc - 1]
        _gamma39 = np.empty((ngrdcol, nzm))
        if l_gamma_skw:
            for _k39 in range(nzm):
                for _i39 in range(ngrdcol):
                    if abs(_gc39[_i39] - _gb39[_i39]) > abs(_gc39[_i39] + _gb39[_i39]) * eps / 2:
                        _gamma39 = _iset(_gamma39, np.s_[_i39, _k39], _gb39[_i39] + (_gc39[_i39] - _gb39[_i39]) * _xp.exp(
                            -0.5 * (Skw_zm[_i39, _k39] / _gcf39[_i39]) ** 2))
                    else:
                        _gamma39 = _iset(_gamma39, np.s_[_i39, _k39], _gc39[_i39])
        else:
            _gamma39 = _xp.broadcast_to(
                clubb_params[:, igamma_coef - 1:igamma_coef], (ngrdcol, nzm)).copy()
        _ssw_jax39 = _asarray(compute_sigma_sqd_w_jax(
            jnp.asarray(_gamma39),
            jnp.asarray(wp2), jnp.asarray(thlp2), jnp.asarray(rtp2),
            jnp.asarray(up2), jnp.asarray(vp2),
            jnp.asarray(wpthlp), jnp.asarray(wprtp),
            jnp.asarray(upwp), jnp.asarray(vpwp),
            flags.l_predict_upwp_vpwp, gr,
        ))
        # Iter55: Fortran comparison removed (oracle removed Iter52)
        sigma_sqd_w = _ssw_jax39
        gamma_Skw_fnc = _gamma39  # save for stats write

    if flags.ipdf_call_placement == ipdf_post_advance_fields:
        # Calculate sigma_sqd_w here
        if l_gamma_skw:
            gamma_Skw_fnc = np.empty((ngrdcol, nzm))
            gc = clubb_params[:, igamma_coef - 1]
            gb = clubb_params[:, igamma_coefb - 1]
            gcf = clubb_params[:, igamma_coefc - 1]
            for k in range(nzm):
                for i in range(ngrdcol):
                    if abs(gc[i] - gb[i]) > abs(gc[i] + gb[i]) * eps / 2:
                        gamma_Skw_fnc = _iset(gamma_Skw_fnc, np.s_[i, k], gb[i] + (gc[i] - gb[i]) * _xp.exp(
                            -0.5 * (Skw_zm[i, k] / gcf[i]) ** 2))
                    else:
                        gamma_Skw_fnc = _iset(gamma_Skw_fnc, np.s_[i, k], gc[i])
        else:
            gamma_Skw_fnc = _xp.broadcast_to(
                clubb_params[:, igamma_coef - 1:igamma_coef], (ngrdcol, nzm)).copy()

        # Iter52: JAX-only compute_sigma_sqd_w (Fortran oracle removed)
        sigma_sqd_w = _asarray(compute_sigma_sqd_w_jax(
            jnp.asarray(gamma_Skw_fnc),
            jnp.asarray(wp2), jnp.asarray(thlp2), jnp.asarray(rtp2),
            jnp.asarray(up2), jnp.asarray(vp2),
            jnp.asarray(wpthlp), jnp.asarray(wprtp),
            jnp.asarray(upwp), jnp.asarray(vpwp),
            flags.l_predict_upwp_vpwp,
            gr,
        ), dtype=np.float64)

    if sigma_sqd_w is None:
        # If PDF closure was called pre-advance, sigma_sqd_w was set there
        # It should already be available from the pdf_closure unpack.
        sigma_sqd_w = np.zeros((ngrdcol, nzm))

    # a3 coefficient
    a3_coef = -two * (one - sigma_sqd_w) ** 2 + 3.0
    a3_min = clubb_params[:, ia3_coef_min - 1]
    for k in range(nzm):
        a3_coef = _iset(a3_coef, np.s_[:, k], _xp.maximum(a3_coef[:, k], a3_min))

    a3_coef_zt = _asarray(zm2zt(a3_coef, gr))

    if l_sample and stats_writer is not None:
        # wp2_zt and wp3_zm are written after advance_wp2_wp3 in Fortran;
        # they are written in the Iter68 stats block below with post-advance values.
        # For ipdf_post_advance_fields: Skw, sigma_sqd_w, gamma_Skw_fnc are written
        # INSIDE the post-advance pdf_closure (Fortran pdf_closure_module.F90:4446-4512)
        # using POST-advance state → skip here, write in Block U with post-advance values.
        if flags.ipdf_call_placement != ipdf_post_advance_fields:
            stats_writer.update("Skw_zt", Skw_zt)
            stats_writer.update("Skw_zm", Skw_zm)
            stats_writer.update("sigma_sqd_w", sigma_sqd_w)
            if gamma_Skw_fnc is not None:
                stats_writer.update("gamma_Skw_fnc", gamma_Skw_fnc)
        stats_writer.update("a3_coef", a3_coef)
        stats_writer.update("a3_coef_zt", a3_coef_zt)

    # Interpolate variances/covariances to thermodynamic levels
    thlp2_zt = _xp.maximum(
        _asarray(zm2zt(thlp2, gr)),
        thl_tol ** 2,
    )
    rtp2_zt = _xp.maximum(
        _asarray(zm2zt(rtp2, gr)),
        rt_tol ** 2,
    )
    rtpthlp_zt = _asarray(zm2zt(rtpthlp, gr))
    if l_sample and stats_writer is not None:
        stats_writer.update("rtpthlp_zt", rtpthlp_zt)

    # wp3_on_wp2 on zt levels
    wp3_on_wp2_zt = wp3 / _xp.maximum(wp2_zt, w_tol_sqd)
    wp3_on_wp2_zt = _xp.clip(wp3_on_wp2_zt, -1000.0, 1000.0)

    wp3_on_wp2 = _asarray(zt2zm(wp3_on_wp2_zt, gr))
    wp3_on_wp2_zt = _asarray(zm2zt(wp3_on_wp2, gr))

    # Iter25 shadow: compare ADG1_pdf_driver_jax against Fortran pdf_params
    if (flags.ipdf_call_placement in (ipdf_pre_advance_fields, ipdf_pre_post_advance_fields)
            and flags.iiPDF_type == iiPDF_ADG1):
        # Compute zt-level inputs (mirroring Fortran pdf_closure_driver internals)
        _up2_zt_j25 = jnp.maximum(zm2zt_jax(jnp.asarray(up2), gr), w_tol_sqd)
        _vp2_zt_j25 = jnp.maximum(zm2zt_jax(jnp.asarray(vp2), gr), w_tol_sqd)
        _sigma_sqd_w_zt_j25 = jnp.maximum(
            zm2zt_jax(jnp.asarray(sigma_sqd_w), gr), zero_threshold)
        _wprtp_zt_j25 = zm2zt_jax(jnp.asarray(wprtp), gr)
        _wpthlp_zt_j25 = zm2zt_jax(jnp.asarray(wpthlp), gr)
        _upwp_zt_j25 = zm2zt_jax(jnp.asarray(upwp), gr)
        _vpwp_zt_j25 = zm2zt_jax(jnp.asarray(vpwp), gr)
        _sqrt_wp2_zt_j25 = jnp.sqrt(jnp.asarray(wp2_zt))
        _beta_j25 = jnp.asarray(clubb_params[:, ibeta - 1])

        _adg1_j25 = ADG1_pdf_driver_jax(
            wm=jnp.asarray(wm_zt),
            rtm=jnp.asarray(rtm),
            thlm=jnp.asarray(thlm),
            um=jnp.asarray(um),
            vm=jnp.asarray(vm),
            wp2=jnp.asarray(wp2_zt),
            rtp2=jnp.asarray(rtp2_zt),
            thlp2=jnp.asarray(thlp2_zt),
            up2=_up2_zt_j25,
            vp2=_vp2_zt_j25,
            Skw=jnp.asarray(Skw_zt),
            wprtp=_wprtp_zt_j25,
            wpthlp=_wpthlp_zt_j25,
            upwp=_upwp_zt_j25,
            vpwp=_vpwp_zt_j25,
            sqrt_wp2=_sqrt_wp2_zt_j25,
            sigma_sqd_w=_sigma_sqd_w_zt_j25,
            beta=_beta_j25,
            mixt_frac_max_mag=mixt_frac_max_mag,
        )

        # Iter55: Fortran pdf_params comparison removed (oracle removed Iter50)
        if l_sample and stats_writer is not None:
            stats_writer.update("mixt_frac",    _asarray(_adg1_j25['mixt_frac'],    dtype=np.float64))
            stats_writer.update("w_1",          _asarray(_adg1_j25['w_1'],          dtype=np.float64))
            stats_writer.update("w_2",          _asarray(_adg1_j25['w_2'],          dtype=np.float64))
            stats_writer.update("varnce_w_1",   _asarray(_adg1_j25['varnce_w_1'],   dtype=np.float64))
            stats_writer.update("varnce_w_2",   _asarray(_adg1_j25['varnce_w_2'],   dtype=np.float64))
            stats_writer.update("rt_1",         _asarray(_adg1_j25['rt_1'],         dtype=np.float64))
            stats_writer.update("rt_2",         _asarray(_adg1_j25['rt_2'],         dtype=np.float64))
            stats_writer.update("varnce_rt_1",  _asarray(_adg1_j25['varnce_rt_1'],  dtype=np.float64))
            stats_writer.update("varnce_rt_2",  _asarray(_adg1_j25['varnce_rt_2'],  dtype=np.float64))
            stats_writer.update("thl_1",        _asarray(_adg1_j25['thl_1'],        dtype=np.float64))
            stats_writer.update("thl_2",        _asarray(_adg1_j25['thl_2'],        dtype=np.float64))
            stats_writer.update("varnce_thl_1", _asarray(_adg1_j25['varnce_thl_1'], dtype=np.float64))
            stats_writer.update("varnce_thl_2", _asarray(_adg1_j25['varnce_thl_2'], dtype=np.float64))

        # ============================================================== #
        # Block I_pre (Iter60): JAX rcm/cloud_frac from PDF parameters   #
        # Replaces Block G oracle values to eliminate binary vs shared    #
        # library floating-point differences at the chi/stdev_chi        #
        # threshold (max_num_stdevs = 5).  Only for ADG1 pre-advance.    #
        # ============================================================== #
        _mf    = _adg1_j25['mixt_frac']
        _rt1   = _adg1_j25['rt_1']
        _rt2   = _adg1_j25['rt_2']
        _thl1  = _adg1_j25['thl_1']
        _thl2  = _adg1_j25['thl_2']
        _vrt1  = _adg1_j25['varnce_rt_1']
        _vrt2  = _adg1_j25['varnce_rt_2']
        _vthl1 = _adg1_j25['varnce_thl_1']
        _vthl2 = _adg1_j25['varnce_thl_2']

        # Component correlation of rt and thl (same for both components)
        _corr_rtthl_1, _corr_rtthl_2 = calc_comp_corrs_binormal_jax(
            jnp.asarray(rtpthlp_zt), jnp.asarray(rtm), jnp.asarray(thlm),
            _rt1, _rt2, _thl1, _thl2,
            _vrt1, _vrt2, _vthl1, _vthl2,
            _mf,
        )

        # Liquid water temperature per component: tl_i = thl_i * exner
        _exner_j = jnp.asarray(exner)
        _p_j     = jnp.asarray(p_in_Pa)
        _tl1     = _thl1 * _exner_j
        _tl2     = _thl2 * _exner_j

        # Saturation mixing ratio per component
        _rsatl1 = sat_mixrat_liq_jax(_p_j, _tl1, flags.saturation_formula)
        _rsatl2 = sat_mixrat_liq_jax(_p_j, _tl2, flags.saturation_formula)

        # transform_pdf_chi_eta_component (Sommeria & Deardorff 1977, eq 3; SD eq 8)
        def _chi_transform_jax(tl, rsatl, rt, exner_in, varnce_rt, varnce_thl, corr_rt_thl):
            beta         = ep * Lv**2 / (Rd * Cp * tl**2)
            invrs        = 1.0 / (1.0 + beta * rsatl)
            chi          = (rt - rsatl) * invrs
            crt          = invrs
            cthl         = (1.0 + beta * rt) * invrs**2 * (Cp / Lv) * beta * rsatl * exner_in
            varnce_chi   = (crt**2 * varnce_rt
                            - 2.0 * corr_rt_thl * crt * cthl
                              * jnp.sqrt(varnce_rt * varnce_thl)
                            + cthl**2 * varnce_thl)
            stdev_chi    = _safe_sqrt(varnce_chi)
            return chi, stdev_chi

        _chi1, _schi1 = _chi_transform_jax(
            _tl1, _rsatl1, _rt1, _exner_j, _vrt1, _vthl1, _corr_rtthl_1)
        _chi2, _schi2 = _chi_transform_jax(
            _tl2, _rsatl2, _rt2, _exner_j, _vrt2, _vthl2, _corr_rtthl_2)

        # calc_liquid_cloud_frac_component (pdf_closure_module.F90, lines 2453-2479)
        def _liquid_cloud_frac_jax(mean_chi, stdev_chi_in):
            is_clear = (
                ((jnp.abs(mean_chi) <= eps) & (stdev_chi_in <= chi_tol))
                | (mean_chi < -max_num_stdevs * stdev_chi_in)
            )
            is_full  = mean_chi > max_num_stdevs * stdev_chi_in
            safe_s   = jnp.maximum(stdev_chi_in, 1.0e-100)
            zeta     = mean_chi / safe_s
            cf_mid   = 0.5 * (1.0 + jax.scipy.special.erf(zeta / sqrt_2))
            rc_mid   = (mean_chi * cf_mid
                        + stdev_chi_in * jnp.exp(-0.5 * zeta**2) / sqrt_2pi)
            cf = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, cf_mid))
            rc = jnp.where(is_clear, 0.0, jnp.where(is_full, mean_chi, rc_mid))
            return cf, rc

        _cf1, _rc1 = _liquid_cloud_frac_jax(_chi1, _schi1)
        _cf2, _rc2 = _liquid_cloud_frac_jax(_chi2, _schi2)

        # Combine components (pdf_closure_module.F90, lines 1020-1024)
        _cloud_frac_jax = _mf * _cf1 + (1.0 - _mf) * _cf2
        _rcm_jax        = jnp.maximum(0.0, _mf * _rc1 + (1.0 - _mf) * _rc2)

        # Override rcm and cloud_frac with JAX-computed values
        rcm        = _asarray(_rcm_jax,        dtype=np.float64)
        cloud_frac = _asarray(_cloud_frac_jax, dtype=np.float64)

    # ================================================================== #
    # Block I: Compute thvm
    # ================================================================== #
    # Iter52: JAX-only calculate_thvm (Fortran oracle removed)
    thvm = _asarray(calculate_thvm_jax(
        jnp.asarray(thlm), jnp.asarray(rtm), jnp.asarray(rcm),
        jnp.asarray(exner), jnp.asarray(thv_ds_zt),
    ), dtype=np.float64)

    # ================================================================== #
    # Block J: TKE computation
    # ================================================================== #
    if not flags.l_tke_aniso:
        em = three_halves * wp2
    else:
        em = 0.5 * (wp2 + vp2 + up2)

    sqrt_em_zt = _xp.maximum(
        _asarray(zm2zt(em, gr)),
        em_min,
    )
    sqrt_em_zt = _xp.sqrt(sqrt_em_zt)

    # ================================================================== #
    # Block K: Brunt-Vaisala, wind shear, Richardson number
    # ================================================================== #
    # Iter52: JAX-only calc_brunt_vaisala_freq_sqd (Fortran oracle removed)
    (brunt_vaisala_freq_sqd, brunt_vaisala_freq_sqd_mixed,
     brunt_vaisala_freq_sqd_smth, brunt_vaisala_freq_sqd_dry,
     brunt_vaisala_freq_sqd_moist) = [
        _asarray(x, dtype=np.float64)
        for x in calc_brunt_vaisala_freq_sqd_jax(
            jnp.asarray(thlm), jnp.asarray(exner), jnp.asarray(rtm),
            jnp.asarray(rcm), jnp.asarray(p_in_Pa), jnp.asarray(thvm),
            jnp.asarray(ice_supersat_frac),
            jnp.asarray(clubb_params[:, ibv_efold - 1]),
            float(T0),
            flags.l_use_thvm_in_bv_freq,
            flags.l_brunt_vaisala_freq_moist,
            flags.l_modify_limiters_for_cnvg_test,
            gr,
        )
    ]
    if l_sample and stats_writer is not None:
        stats_writer.update("bv_freq_sqd", brunt_vaisala_freq_sqd)
        stats_writer.update("bv_freq_sqd_mixed", brunt_vaisala_freq_sqd_mixed)
        stats_writer.update("bv_freq_sqd_smth", brunt_vaisala_freq_sqd_smth)
        stats_writer.update("bv_freq_sqd_dry", brunt_vaisala_freq_sqd_dry)
        stats_writer.update("bv_freq_sqd_moist", brunt_vaisala_freq_sqd_moist)

    ddzt_um = _asarray(ddzt(um, gr))
    ddzt_vm = _asarray(ddzt(vm, gr))
    ddzt_umvm_sqd = ddzt_um ** 2 + ddzt_vm ** 2

    if l_sample and stats_writer is not None:
        stats_writer.update("ddzt_umvm_sqd", ddzt_umvm_sqd)

    # Richardson number
    if flags.l_modify_limiters_for_cnvg_test:
        # Iter57: JAX calc_ri_zm (Fortran oracle removed; same formula validated for ARM path)
        Ri_zm = _asarray(calc_ri_zm_jax(
            jnp.asarray(brunt_vaisala_freq_sqd_smth),
            jnp.asarray(ddzt_umvm_sqd),
            0.0, 1.0e-12,
        ), dtype=np.float64)
        Ri_zm = _asarray(zm2zt2zm(Ri_zm, gr, zm_min=0.0))
    else:
        if l_smooth_min_max:
            # Iter57: smooth_max_jax replaces Fortran smooth_max
            brunt_vaisala_freq_clipped = _asarray(smooth_max_jax(
                jnp.asarray(1.0e-7),
                jnp.asarray(brunt_vaisala_freq_sqd_smth),
                1.0e-4 * min_max_smth_mag,
            ), dtype=np.float64)
            ddzt_umvm_sqd_clipped = _asarray(smooth_max_jax(
                jnp.asarray(ddzt_umvm_sqd),
                jnp.asarray(1.0e-7),
                1.0e-6 * min_max_smth_mag,
            ), dtype=np.float64)
            # Iter57: calc_ri_zm_jax with lim=0 (smooth_max already applied)
            Ri_zm = _asarray(calc_ri_zm_jax(
                jnp.asarray(brunt_vaisala_freq_clipped),
                jnp.asarray(ddzt_umvm_sqd_clipped),
                0.0, 0.0,
            ), dtype=np.float64)
        else:
            # Iter52: JAX-only calc_ri_zm (Fortran oracle removed)
            Ri_zm = _asarray(calc_ri_zm_jax(
                jnp.asarray(brunt_vaisala_freq_sqd_smth),
                jnp.asarray(ddzt_umvm_sqd),
                1.0e-7, 1.0e-7,
            ), dtype=np.float64)

    # ================================================================== #
    # Block L: Mixing length / dissipation time scale
    # ================================================================== #
    if not flags.l_diag_Lscale_from_tau:
        # Iter63: JAX calc_lscale_directly (l_avg_Lscale=False, ascending grid)
        _Lscale_j, _Lscale_up_j, _Lscale_down_j = calc_lscale_directly_jax(
            jnp.asarray(thvm), jnp.asarray(thlm), jnp.asarray(rtm),
            jnp.asarray(em), jnp.asarray(Lscale_max),
            jnp.asarray(p_in_Pa), jnp.asarray(exner), jnp.asarray(thv_ds_zt),
            jnp.asarray(clubb_params), lmin,
            flags.saturation_formula, l_implemented, gr,
        )
        Lscale      = _asarray(_Lscale_j,      dtype=np.float64)
        Lscale_up   = _asarray(_Lscale_up_j,   dtype=np.float64)
        Lscale_down = _asarray(_Lscale_down_j, dtype=np.float64)

        # tau from Lscale
        tau_zt = _xp.minimum(Lscale / sqrt_em_zt,
                            clubb_params[:, itaumax - 1:itaumax])

        Lscale_zm = _xp.maximum(
            _asarray(zt2zm(Lscale, gr)),
            zero_threshold,
        )

        tau_zm = _xp.minimum(
            Lscale_zm / _xp.sqrt(_xp.maximum(em_min, em)),
            clubb_params[:, itaumax - 1:itaumax],
        )

        invrs_tau_zm = one / tau_zm
        invrs_tau_wp2_zm = invrs_tau_zm.copy()
        invrs_tau_xp2_zm = invrs_tau_zm.copy()
        invrs_tau_wpxp_zm = invrs_tau_zm.copy()
        invrs_tau_wp3_zm = invrs_tau_zm.copy()
        tau_max_zm = _xp.broadcast_to(
            clubb_params[:, itaumax - 1:itaumax], (ngrdcol, nzm)).copy()

        invrs_tau_zt = one / tau_zt
        invrs_tau_wp3_zt = invrs_tau_zt.copy()
        tau_max_zt = _xp.broadcast_to(
            clubb_params[:, itaumax - 1:itaumax], (ngrdcol, nzt)).copy()

        # Placeholder variables not computed in this branch
        invrs_tau_no_N2_zm = np.zeros((ngrdcol, nzm))
        invrs_tau_bkgnd = np.zeros((ngrdcol, nzm))
        invrs_tau_shear = np.zeros((ngrdcol, nzm))
        invrs_tau_sfc = np.zeros((ngrdcol, nzm))
        invrs_tau_N2_iso = np.zeros((ngrdcol, nzm))
    else:
        # Iter52: JAX-only diagnose_lscale_from_tau (Fortran oracle removed)
        (_j_invrs_tau_zt, _j_invrs_tau_zm,
         _j_invrs_tau_sfc, _j_invrs_tau_no_N2, _j_invrs_tau_bkgnd,
         _j_invrs_tau_shear, _j_invrs_tau_N2_iso,
         _j_invrs_tau_wp2, _j_invrs_tau_xp2,
         _j_invrs_tau_wp3_zm, _j_invrs_tau_wp3_zt, _j_invrs_tau_wpxp,
         _j_tau_max_zm, _j_tau_max_zt, _j_tau_zm, _j_tau_zt,
         _j_Lscale, _j_Lscale_up, _j_Lscale_down,
         _j_brunt_freq_pos, _j_brunt_freq_out_cloud) = diagnose_lscale_from_tau_jax(
            upwp_sfc=jnp.asarray(upwp_sfc),
            vpwp_sfc=jnp.asarray(vpwp_sfc),
            ddzt_umvm_sqd=jnp.asarray(ddzt_umvm_sqd),
            ice_supersat_frac=jnp.asarray(ice_supersat_frac),
            em=jnp.asarray(em),
            sqrt_em_zt=jnp.asarray(sqrt_em_zt),
            ufmin=ufmin,
            tau_const=tau_const,
            sfc_elevation=jnp.asarray(sfc_elevation),
            Lscale_max=jnp.asarray(Lscale_max),
            clubb_params=jnp.asarray(clubb_params),
            Ri_zm=jnp.asarray(Ri_zm),
            brunt_vaisala_freq_sqd_smth=jnp.asarray(brunt_vaisala_freq_sqd_smth),
            l_e3sm_config=flags.l_e3sm_config,
            l_smooth_Heaviside_tau_wpxp=flags.l_smooth_Heaviside_tau_wpxp,
            gr=gr,
        )
        invrs_tau_zt     = _asarray(_j_invrs_tau_zt,     dtype=np.float64)
        invrs_tau_zm     = _asarray(_j_invrs_tau_zm,     dtype=np.float64)
        invrs_tau_sfc    = _asarray(_j_invrs_tau_sfc,    dtype=np.float64)
        invrs_tau_no_N2_zm = _asarray(_j_invrs_tau_no_N2, dtype=np.float64)
        invrs_tau_bkgnd  = _asarray(_j_invrs_tau_bkgnd,  dtype=np.float64)
        invrs_tau_shear  = _asarray(_j_invrs_tau_shear,  dtype=np.float64)
        invrs_tau_N2_iso = _asarray(_j_invrs_tau_N2_iso, dtype=np.float64)
        invrs_tau_wp2_zm = _asarray(_j_invrs_tau_wp2,    dtype=np.float64)
        invrs_tau_xp2_zm = _asarray(_j_invrs_tau_xp2,    dtype=np.float64)
        invrs_tau_wp3_zm = _asarray(_j_invrs_tau_wp3_zm, dtype=np.float64)
        invrs_tau_wp3_zt = _asarray(_j_invrs_tau_wp3_zt, dtype=np.float64)
        invrs_tau_wpxp_zm = _asarray(_j_invrs_tau_wpxp,  dtype=np.float64)
        tau_max_zm       = _asarray(_j_tau_max_zm,       dtype=np.float64)
        tau_max_zt       = _asarray(_j_tau_max_zt,       dtype=np.float64)
        tau_zm           = _asarray(_j_tau_zm,           dtype=np.float64)
        tau_zt           = _asarray(_j_tau_zt,           dtype=np.float64)
        Lscale           = _asarray(_j_Lscale,           dtype=np.float64)
        Lscale_up        = _asarray(_j_Lscale_up,        dtype=np.float64)
        Lscale_down      = _asarray(_j_Lscale_down,      dtype=np.float64)

    if l_sample and stats_writer is not None:
        stats_writer.update("Lscale", Lscale)
        stats_writer.update("Lscale_up", Lscale_up)
        stats_writer.update("Lscale_down", Lscale_down)
        stats_writer.update("tau_zm", tau_zm)
        stats_writer.update("tau_zt", tau_zt)
        if flags.l_diag_Lscale_from_tau:
            stats_writer.update("bv_freq_pos",       _asarray(_j_brunt_freq_pos,       dtype=np.float64))
            stats_writer.update("bv_freq_out_cloud",  _asarray(_j_brunt_freq_out_cloud, dtype=np.float64))

    # ================================================================== #
    # Block M: Eddy diffusivity
    # ================================================================== #
    # Kh_zt = c_K * Lscale * sqrt_em_zt
    c_K = clubb_params[:, ic_K - 1]
    Kh_zt = c_K[:, None] * Lscale * sqrt_em_zt

    Lscale_zm = _asarray(zt2zm(Lscale, gr))
    Kh_zm = (c_K[:, None]
             * _xp.maximum(Lscale_zm, zero_threshold)
             * _xp.sqrt(_xp.maximum(em, em_min)))

    # Iter52: JAX-only wp23_term_splat_lhs (Fortran oracle removed)
    _jax21_wp2, _jax21_wp3, _bv_sqd_splat21 = wp23_term_splat_lhs_jax(
        brunt_vaisala_freq_sqd_mixed=jnp.asarray(brunt_vaisala_freq_sqd_mixed),
        Lscale_zm=jnp.asarray(Lscale_zm),
        rho_ds_zm=jnp.asarray(rho_ds_zm),
        C_wp2_splat=jnp.asarray(clubb_params[:, iC_wp2_splat - 1]),
        below_grnd_val=below_grnd_val,
        gr=gr,
    )
    lhs_splat_wp2 = _asarray(_jax21_wp2, dtype=np.float64)
    lhs_splat_wp3 = _asarray(_jax21_wp3, dtype=np.float64)
    if l_sample and stats_writer is not None:
        stats_writer.update("bv_freq_sqd_splat", _asarray(_bv_sqd_splat21, dtype=np.float64))

    # Iter53: iter4/5 shadow comparisons removed (Fortran oracle validated; JAX-only)

    # ================================================================== #
    # Block N: Surface variances
    # ================================================================== #
    # Iter52: JAX-only calc_sfc_varnce (Fortran oracle removed)
    _wp2_pre17    = wp2.copy()
    _up2_pre17    = up2.copy()
    _vp2_pre17    = vp2.copy()
    _thlp2_pre17  = thlp2.copy()
    _rtp2_pre17   = rtp2.copy()
    _rtpthlp_pre17 = rtpthlp.copy()
    _zm_sfc17 = _asarray(gr.zm)[:, 0]
    (
        wp2, up2, vp2,
        thlp2, rtp2, rtpthlp,
    ) = [_asarray(x, dtype=np.float64) for x in calc_sfc_varnce_jax(
        jnp.asarray(upwp_sfc),
        jnp.asarray(vpwp_sfc),
        jnp.asarray(wpthlp),
        jnp.asarray(wprtp_sfc),
        jnp.asarray(lhs_splat_wp2),
        jnp.asarray(tau_zm),
        float(T0),
        jnp.asarray(clubb_params[:, iup2_sfc_coef - 1]),
        jnp.asarray(clubb_params[:, ia_const - 1]),
        jnp.asarray(_wp2_pre17),
        jnp.asarray(_up2_pre17),
        jnp.asarray(_vp2_pre17),
        jnp.asarray(_thlp2_pre17),
        jnp.asarray(_rtp2_pre17),
        jnp.asarray(_rtpthlp_pre17),
        jnp.asarray(_zm_sfc17),
        jnp.asarray(sfc_elevation),
    )]

    # Surface forcing (sf) budget stats — sfc_varnce_module.F90 pattern
    if l_sample and stats_writer is not None:
        _dt17 = float(dt_advance)
        stats_writer.update("wp2_sf",     (wp2     - _wp2_pre17)     / _dt17)
        stats_writer.update("up2_sf",     (up2     - _up2_pre17)     / _dt17)
        stats_writer.update("vp2_sf",     (vp2     - _vp2_pre17)     / _dt17)
        stats_writer.update("thlp2_sf",   (thlp2   - _thlp2_pre17)   / _dt17)
        stats_writer.update("rtp2_sf",    (rtp2    - _rtp2_pre17)    / _dt17)
        stats_writer.update("rtpthlp_sf", (rtpthlp - _rtpthlp_pre17) / _dt17)

    # ================================================================== #
    # Block O: Stats — pre-advance outputs (rvm, rel_humidity)
    # ================================================================== #
    if l_sample and stats_writer is not None:
        stats_writer.update("rvm", rtm - rcm)
        if stats_writer.var_on_stats_list("rel_humidity") or stats_writer.var_on_stats_list("rsat"):
            # Iter60: thlm2T_in_K and sat_mixrat_liq replaced with JAX
            T_in_K = thlm * exner + (Lv / Cp) * rcm
            rsat = _asarray(sat_mixrat_liq_jax(
                jnp.asarray(p_in_Pa),
                jnp.asarray(T_in_K),
                flags.saturation_formula,
            ), dtype=np.float64)
            rel_humidity = (rtm - rcm) / rsat
            stats_writer.update("rel_humidity", rel_humidity)

    # ================================================================== #
    # Block P: Extract PDF params for zm grid
    # ================================================================== #
    # Iter59: Restructured to eliminate Fortran pdf_params access for ARM ADG1 paths.
    # pdf_params (Fortran object) is zero-initialized; only used as fallback for non-ADG1/non-ARM.
    pdf_params = pdf_params

    if flags.l_call_pdf_closure_twice:
        pdf_params_zm = pdf_params_zm
        w_1_zm = pdf_params_zm.w_1.copy()
        w_2_zm = pdf_params_zm.w_2.copy()
        varnce_w_1_zm = pdf_params_zm.varnce_w_1.copy()
        varnce_w_2_zm = pdf_params_zm.varnce_w_2.copy()
        mixt_frac_zm = pdf_params_zm.mixt_frac.copy()
    elif (flags.ipdf_call_placement == ipdf_post_advance_fields
            and flags.iiPDF_type == iiPDF_ADG1):
        # ARM post-advance path: use previous timestep's ADG1 result (or zeros on ts1).
        # Fortran pdf_params is zero-initialized → zeros on ts1 is identical to Fortran.
        if _prev_adg1_j25 is not None:
            w_1_zm        = _asarray(zt2zm_jax(_prev_adg1_j25['w_1'],        gr), dtype=np.float64)
            w_2_zm        = _asarray(zt2zm_jax(_prev_adg1_j25['w_2'],        gr), dtype=np.float64)
            varnce_w_1_zm = _asarray(zt2zm_jax(_prev_adg1_j25['varnce_w_1'], gr), dtype=np.float64)
            varnce_w_2_zm = _asarray(zt2zm_jax(_prev_adg1_j25['varnce_w_2'], gr), dtype=np.float64)
            mixt_frac_zm  = _asarray(zt2zm_jax(_prev_adg1_j25['mixt_frac'],  gr), dtype=np.float64)
        else:
            # Iter59: timestep 1 — pdf_params zero-initialized by Fortran; replicate without Fortran
            w_1_zm        = np.zeros((ngrdcol, nzm), dtype=np.float64)
            w_2_zm        = np.zeros((ngrdcol, nzm), dtype=np.float64)
            varnce_w_1_zm = np.zeros((ngrdcol, nzm), dtype=np.float64)
            varnce_w_2_zm = np.zeros((ngrdcol, nzm), dtype=np.float64)
            mixt_frac_zm  = np.zeros((ngrdcol, nzm), dtype=np.float64)
    elif (flags.ipdf_call_placement in (ipdf_pre_advance_fields, ipdf_pre_post_advance_fields)
            and flags.iiPDF_type == iiPDF_ADG1
            and not flags.l_call_pdf_closure_twice):
        # ARM pre-advance path: use current timestep's ADG1 result
        w_1_zm        = _asarray(zt2zm_jax(_adg1_j25['w_1'],        gr), dtype=np.float64)
        w_2_zm        = _asarray(zt2zm_jax(_adg1_j25['w_2'],        gr), dtype=np.float64)
        varnce_w_1_zm = _asarray(zt2zm_jax(_adg1_j25['varnce_w_1'], gr), dtype=np.float64)
        varnce_w_2_zm = _asarray(zt2zm_jax(_adg1_j25['varnce_w_2'], gr), dtype=np.float64)
        mixt_frac_zm  = _asarray(zt2zm_jax(_adg1_j25['mixt_frac'],  gr), dtype=np.float64)
    else:
        # Non-ADG1 / non-ARM fallback: read from Fortran pdf_params object
        w_1_zm        = _asarray(zt2zm(pdf_params.w_1, gr))
        w_2_zm        = _asarray(zt2zm(pdf_params.w_2, gr))
        varnce_w_1_zm = _asarray(zt2zm(pdf_params.varnce_w_1, gr))
        varnce_w_2_zm = _asarray(zt2zm(pdf_params.varnce_w_2, gr))
        mixt_frac_zm  = _asarray(zt2zm(pdf_params.mixt_frac, gr))

    # ================================================================== #
    # Block Q: Stability correction, invrs_tau
    # ================================================================== #
    if flags.l_stability_correct_tau_zm:
        # Iter57: calc_stability_correction_jax replaces Fortran oracle
        stability_correction = _asarray(calc_stability_correction_jax(
            jnp.asarray(brunt_vaisala_freq_sqd),
            jnp.asarray(Lscale_zm),
            jnp.asarray(em),
            jnp.asarray(clubb_params),
        ), dtype=np.float64)
        if l_sample and stats_writer is not None:
            stats_writer.update("stability_correction", stability_correction)

        invrs_tau_N2_zm = invrs_tau_zm * stability_correction
        invrs_tau_C6_zm = invrs_tau_N2_zm.copy()
        invrs_tau_C1_zm = invrs_tau_N2_zm.copy()
    else:
        stability_correction = np.zeros((ngrdcol, nzm))
        invrs_tau_N2_zm = np.full((ngrdcol, nzm), unused_var)
        invrs_tau_C6_zm = invrs_tau_wpxp_zm.copy()
        invrs_tau_C1_zm = invrs_tau_wp2_zm.copy()

    # C14 always uses wp2 tau
    invrs_tau_C14_zm = invrs_tau_wp2_zm.copy()

    # C4 tau
    if (not flags.l_diag_Lscale_from_tau) and l_use_invrs_tau_n2_iso:
        raise RuntimeError(
            "Error! l_use_invrs_tau_N2_iso is not used when "
            "l_diag_Lscale_from_tau=false."
            "If you want to use Lscale code, go to file "
            "src/CLUBB_core/advance_clubb_core_module.F90 and "
            "change l_use_invrs_tau_N2_iso to false"
        )
    if not l_use_invrs_tau_n2_iso:
        invrs_tau_C4_zm = invrs_tau_wp2_zm.copy()
    else:
        invrs_tau_C4_zm = invrs_tau_N2_iso.copy()

    if l_sample and stats_writer is not None:
        stats_writer.update("invrs_tau_zm", invrs_tau_zm)
        stats_writer.update("invrs_tau_xp2_zm", invrs_tau_xp2_zm)
        stats_writer.update("invrs_tau_wp2_zm", invrs_tau_wp2_zm)
        stats_writer.update("invrs_tau_wpxp_zm", invrs_tau_wpxp_zm)
        stats_writer.update("Ri_zm", Ri_zm)
        stats_writer.update("invrs_tau_wp3_zm", invrs_tau_wp3_zm)
        if flags.l_diag_Lscale_from_tau:
            stats_writer.update("invrs_tau_no_N2_zm", invrs_tau_no_N2_zm)
            stats_writer.update("invrs_tau_bkgnd", invrs_tau_bkgnd)
            stats_writer.update("invrs_tau_sfc", invrs_tau_sfc)
            stats_writer.update("invrs_tau_shear", invrs_tau_shear)
    # ================================================================== #
    # Block R: Cx_fnc_Richardson
    # ================================================================== #
    if flags.l_use_C7_Richardson or flags.l_use_C11_Richardson:
        # Iter52: JAX-only compute_cx_fnc_richardson (Fortran oracle removed)
        Cx_fnc_Richardson = _asarray(compute_cx_fnc_richardson_jax(
            jnp.asarray(brunt_vaisala_freq_sqd),
            jnp.asarray(brunt_vaisala_freq_sqd_mixed),
            jnp.asarray(ddzt_umvm_sqd),
            jnp.asarray(clubb_params),
            l_use_shear_Richardson=flags.l_use_shear_Richardson,
            l_modify_limiters_for_cnvg_test=flags.l_modify_limiters_for_cnvg_test,
        ), dtype=np.float64)
    else:
        Cx_fnc_Richardson = np.zeros((ngrdcol, nzm))

    # ================================================================== #
    # Block S: Main advance loop (4 iterations)
    # ================================================================== #
    # Locally-needed PDF closure outputs for advance routines
    wprtp2 = _wprtp2
    wpthlp2 = _wpthlp2
    wprtpthlp = _wprtpthlp
    wp2sclrp = _wp2sclrp
    wpsclrp2 = _wpsclrp2
    wpsclrprtp = _wpsclrprtp
    wpsclrpthlp = _wpsclrpthlp

    wprtp_cl_num = 0
    wpthlp_cl_num = 0
    upwp_cl_num = 0
    vpwp_cl_num = 0
    wprtp_cl_max = 3
    wpthlp_cl_max = 3
    upwp_cl_max = 3
    vpwp_cl_max = 3

    for advance_iter in range(1, 5):

        if advance_iter == order_xm_wpxp_val:
            # ---- Save pre-call state for iter11 shadow comparison ----
            _wprtp_pre11 = wprtp.copy()
            _rtm_pre11   = rtm.copy()
            _wpthlp_pre11 = wpthlp.copy()
            _thlm_pre11   = thlm.copy()
            # Iter37: save pre-call um/vm/upwp/vpwp for JAX shadow comparison
            _um_pre37   = _asarray(um,   dtype=np.float64).copy()
            _vm_pre37   = _asarray(vm,   dtype=np.float64).copy()
            _upwp_pre37 = _asarray(upwp, dtype=np.float64).copy()
            _vpwp_pre37 = _asarray(vpwp, dtype=np.float64).copy()

            # Iter46: Fortran advance_xm_wpxp oracle removed; JAX-only below.

            # ============================================================ #
            # Iter11/46: JAX advance_xm_wpxp (wprtp/rtm, wpthlp/thlm)      #
            # Fortran oracle removed Iter46; verified machine epsilon Iter11 #
            # ARM: ADG1, l_diag_Lscale_from_tau=True, l_use_C7_Richardson   #
            # ============================================================ #
            _c_K6_11 = float(clubb_params[0, ic_K6 - 1])
            _Kw6_11 = _c_K6_11 * Kh_zt   # (ngrdcol, nzt)
            _nu6_11 = float(_asarray(nu_vert_res_dep.nu6, dtype=np.float64).flat[0])
            # For ARM l_diag_Lscale_from_tau=True: C6 is constant per column
            _C6rt_11 = jnp.broadcast_to(
                jnp.array(clubb_params[:, iC6rt - 1])[:, None], (ngrdcol, nzm))
            _C6thl_11 = jnp.broadcast_to(
                jnp.array(clubb_params[:, iC6thl - 1])[:, None], (ngrdcol, nzm))
            # For ARM l_use_C7_Richardson=True: C7 = Cx_fnc_Richardson
            _C7_11 = jnp.array(Cx_fnc_Richardson)   # (ngrdcol, nzm)

            # Iter68: advance_xm_wpxp_module.F90 stats (C6/C7 Skw_fnc, C6_term)
            if l_sample and stats_writer is not None:
                stats_writer.update("C7_Skw_fnc",   _asarray(_C7_11,   dtype=np.float64))
                stats_writer.update("C6rt_Skw_fnc", _asarray(_C6rt_11, dtype=np.float64))
                stats_writer.update("C6thl_Skw_fnc",_asarray(_C6thl_11,dtype=np.float64))
                _C6_term_11 = _C6rt_11 * jnp.asarray(invrs_tau_C6_zm)
                stats_writer.update("C6_term",      _asarray(_C6_term_11, dtype=np.float64))
                # coef_wp2rtp/thlp_implicit (ADG1): a1_zt * wp3_on_wp2_zt
                _a1_zm_11 = 1.0 / (1.0 - jnp.asarray(sigma_sqd_w))
                _a1_zt_11 = jnp.maximum(zm2zt_jax(_a1_zm_11, gr), zero_threshold)
                _coef_wp2rtp_11 = _a1_zt_11 * jnp.asarray(wp3_on_wp2_zt)
                stats_writer.update("coef_wp2rtp_implicit",  _asarray(_coef_wp2rtp_11, dtype=np.float64))
                stats_writer.update("coef_wp2thlp_implicit", _asarray(_coef_wp2rtp_11, dtype=np.float64))
                # coef_wprtp2/thlp2/rtpthlp_implicit (ADG1): (1/3)*beta*a1_zt*wp3_on_wp2_zt
                _beta_11 = jnp.asarray(clubb_params[:, ibeta - 1])[:, None]
                _coef_wprtp2_11 = (1.0 / 3.0) * _beta_11 * _a1_zt_11 * jnp.asarray(wp3_on_wp2_zt)
                stats_writer.update("coef_wprtp2_implicit",    _asarray(_coef_wprtp2_11, dtype=np.float64))
                stats_writer.update("coef_wpthlp2_implicit",   _asarray(_coef_wprtp2_11, dtype=np.float64))
                stats_writer.update("coef_wprtpthlp_implicit", _asarray(_coef_wprtp2_11, dtype=np.float64))

            # Compute all shared LHS terms (diffusion, MA, TP, AC+PR2, TA)
            _sh11 = compute_shared_xm_wpxp_lhs_terms(
                wm_zm=jnp.array(wm_zm), wm_zt=jnp.array(wm_zt),
                wp2=jnp.array(wp2), Kw6=jnp.array(_Kw6_11), nu6=_nu6_11,
                C7_Skw_fnc=_C7_11,
                invrs_rho_ds_zm=jnp.array(invrs_rho_ds_zm),
                rho_ds_zt=jnp.array(rho_ds_zt),
                rho_ds_zm=jnp.array(rho_ds_zm),
                invrs_rho_ds_zt=jnp.array(invrs_rho_ds_zt),
                sigma_sqd_w=jnp.array(sigma_sqd_w),
                wp3_on_wp2_zt=jnp.array(wp3_on_wp2_zt),
                gr=gr,
            )

            # Clipping parameters (matches xm_wpxp_clipping_and_stats in Fortran)
            # l_enable_relaxed_clipping: xp2_floor = 1e-7 for rtp2, 0.01 for thlp2
            _wp2_jax = jnp.array(wp2)
            _rtp2_jax = jnp.array(rtp2)
            _thlp2_jax = jnp.array(thlp2)
            if flags.l_enable_relaxed_clipping:
                _rtp2_clip = jnp.maximum(_rtp2_jax, 1e-7)
                _thlp2_clip = jnp.maximum(_thlp2_jax, 0.01)
            else:
                _rtp2_clip = _rtp2_jax
                _thlp2_clip = _thlp2_jax

            # Iter84: monotonic flux limiter setup. The turbulent-advection range is
            # field-independent — compute once, reuse for rtm/thlm/um/vm. The limiter is
            # applied after each solve and before clip_covar (matching the Fortran
            # xm_wpxp_clipping_and_stats order). It is a no-op unless w'x' exceeds the
            # monotonic bounds (fixes atex; no-op for the bit-faithful set).
            _lle_mfl, _hle_mfl = calc_turb_adv_range(
                w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm, mixt_frac_zm,
                gr, float(dt_advance))
            _rho_ds_zm_mfl = _asarray(rho_ds_zm, np.float64)
            _rho_ds_zt_mfl = _asarray(rho_ds_zt, np.float64)
            _irho_zm_mfl = _asarray(invrs_rho_ds_zm, np.float64)
            _irho_zt_mfl = _asarray(invrs_rho_ds_zt, np.float64)
            _wm_zt_mfl = _asarray(wm_zt, np.float64)

            def _apply_mfl(stype, xm_j, wpxp_j, xm_old, xp2, xm_forcing,
                           xp2_thr, xm_tol):
                # REFACTOR B2 (iter11): the field-path limiter is now JAX (lax.scan), bit-exact to the
                # NumPy reference (tests/test_mono_flux_limiter.py) and differentiable w.r.t. the fields.
                # calc_turb_adv_range (the integer ranges _lle_mfl/_hle_mfl) stays NumPy — they derive
                # from the w-PDF, not the limited fields, so they are structural constants for the grad.
                return monotonic_turbulent_flux_limit_jax(
                    stype, xm_j, wpxp_j, xm_old, xp2,
                    _wm_zt_mfl, xm_forcing,
                    _rho_ds_zm_mfl, _rho_ds_zt_mfl, _irho_zm_mfl, _irho_zt_mfl,
                    xp2_thr, xm_tol, _lle_mfl, _hle_mfl, gr, float(dt_advance))

            # Solve wprtp/rtm pair — no clipping in solve; apply separately to get pre-clip value
            _wprtp_preclip11, _rtm_jax11 = advance_xm_wpxp_jax(
                wpxp=jnp.array(_wprtp_pre11),
                xm=jnp.array(_rtm_pre11),
                wpxp_forcing=jnp.array(wprtp_forcing),
                xm_forcing=jnp.array(rtm_forcing),
                C6_Skw_fnc=_C6rt_11,
                C7_Skw_fnc=_C7_11,
                invrs_tau_C6_zm=jnp.array(invrs_tau_C6_zm),
                lhs_ta_wpxp=_sh11['lhs_ta_wprtp'],
                lhs_diff_zm=_sh11['lhs_diff_zm'],
                lhs_ma_zm=_sh11['lhs_ma_zm'],
                lhs_ma_zt=_sh11['lhs_ma_zt'],
                lhs_ta_xm=_sh11['lhs_ta_xm'],
                lhs_tp=_sh11['lhs_tp'],
                lhs_ac_pr2=_sh11['lhs_ac_pr2'],
                thv_ds_zm=jnp.array(thv_ds_zm),
                xpthvp=jnp.array(rtpthvp),
                wm_zt=jnp.array(wm_zt),
                dt=float(dt_advance),
                gr=gr,
            )
            if getattr(flags, 'l_mono_flux_lim_rtm', False):
                _rtm_jax11, _wprtp_preclip11 = _apply_mfl(
                    MFL_RTM, _rtm_jax11, _wprtp_preclip11, _rtm_pre11, rtp2,
                    rtm_forcing, _RT_TOL ** 2, _RT_TOL_MFL)
            # rtm_cl: fill_holes_vertical on the mean field (advance_xm_wpxp_module.F90:4977-5012),
            # gated fill_holes_type/=0 & solve_type/=um/vm; zt-level, threshold=rt_tol, full zt range.
            # A bitwise no-op where rtm>=rt_tol everywhere (all 15 uniform-grid cases); fires only at a
            # stretched dry top near the floor (rico k51 moist/dry interface — the step-1 rtm seed).
            if flags.fill_holes_type != 0:
                _rtm_jax11 = jnp.asarray(fill_holes_vertical_jax(
                    field=_rtm_jax11, rho_ds=jnp.asarray(rho_ds_zt),
                    dz=jnp.asarray(gr.dzt), threshold=float(rt_tol),
                    lower_k=gr.k_lb_zt, upper_k=gr.k_ub_zt,
                    fill_holes_type=flags.fill_holes_type))
            _wprtp_jax11 = clip_covar_jax(_wprtp_preclip11, _wp2_jax, _rtp2_clip)

            # Solve wpthlp/thlm pair (same lhs_ta_wprtp for ADG1)
            _wpthlp_preclip11, _thlm_jax11 = advance_xm_wpxp_jax(
                wpxp=jnp.array(_wpthlp_pre11),
                xm=jnp.array(_thlm_pre11),
                wpxp_forcing=jnp.array(wpthlp_forcing),
                xm_forcing=jnp.array(thlm_forcing),
                C6_Skw_fnc=_C6thl_11,
                C7_Skw_fnc=_C7_11,
                invrs_tau_C6_zm=jnp.array(invrs_tau_C6_zm),
                lhs_ta_wpxp=_sh11['lhs_ta_wprtp'],
                lhs_diff_zm=_sh11['lhs_diff_zm'],
                lhs_ma_zm=_sh11['lhs_ma_zm'],
                lhs_ma_zt=_sh11['lhs_ma_zt'],
                lhs_ta_xm=_sh11['lhs_ta_xm'],
                lhs_tp=_sh11['lhs_tp'],
                lhs_ac_pr2=_sh11['lhs_ac_pr2'],
                thv_ds_zm=jnp.array(thv_ds_zm),
                xpthvp=jnp.array(thlpthvp),
                wm_zt=jnp.array(wm_zt),
                dt=float(dt_advance),
                gr=gr,
            )
            if getattr(flags, 'l_mono_flux_lim_thlm', False):
                _thlm_jax11, _wpthlp_preclip11 = _apply_mfl(
                    MFL_THLM, _thlm_jax11, _wpthlp_preclip11, _thlm_pre11, thlp2,
                    thlm_forcing, _THL_TOL ** 2, _THL_TOL_MFL)
            # thlm_cl: fill_holes_vertical on thlm (threshold=thl_tol). thlm~300>>thl_tol so this is a
            # guaranteed no-op, but kept to mirror the Fortran (advance_xm_wpxp_module.F90:5008) exactly.
            if flags.fill_holes_type != 0:
                _thlm_jax11 = jnp.asarray(fill_holes_vertical_jax(
                    field=_thlm_jax11, rho_ds=jnp.asarray(rho_ds_zt),
                    dz=jnp.asarray(gr.dzt), threshold=float(thl_tol),
                    lower_k=gr.k_lb_zt, upper_k=gr.k_ub_zt,
                    fill_holes_type=flags.fill_holes_type))
            _wpthlp_jax11 = clip_covar_jax(_wpthlp_preclip11, _wp2_jax, _thlp2_clip)

            # Iter46: Fortran oracle removed; JAX results are the state.

            # ============================================================ #
            # Iter35: Override advance_xm_wpxp state with JAX values        #
            # wprtp/rtm/wpthlp/thlm verified at machine epsilon (iter11).   #
            # ============================================================ #
            wprtp  = _asarray(_wprtp_jax11,  dtype=np.float64).copy()
            rtm    = _asarray(_rtm_jax11,    dtype=np.float64).copy()
            wpthlp = _asarray(_wpthlp_jax11, dtype=np.float64).copy()
            thlm   = _asarray(_thlm_jax11,   dtype=np.float64).copy()

            # Sponge-layer damping for rtm/thlm (advance_xm_wpxp_module.F90:1053-1093).
            # No-op unless sponge_cfg enables the field (e.g. ekman).
            rtm  = _apply_sponge_field('rtm',  rtm,  rtm_ref, gr, dt_advance, sponge_cfg)
            thlm = _apply_sponge_field('thlm', thlm, thlm_ref, gr, dt_advance, sponge_cfg)

            # Iter46: clip_rcm using JAX-updated rtm (moved from after Fortran call)
            _rcm_pre24 = _asarray(rcm, dtype=np.float64).copy()
            _rcm_j24 = _asarray(clip_rcm_jax(
                rcm=jnp.asarray(_rcm_pre24),
                rtm=jnp.asarray(_asarray(rtm, dtype=np.float64)),
            ))
            rcm = _asarray(_rcm_j24, dtype=np.float64).copy()

            # ============================================================ #
            # Iter37: JAX shadow for um/upwp and vm/vpwp (advance_xm_wpxp) #
            # ARM: l_predict_upwp_vpwp=True, l_implemented=False,           #
            # l_lmm_stepping=False, l_ho_trad/nontrad_coriolis=False        #
            # ============================================================ #
            # upwp_forcing = C_uu_shr * wp2 * ddzt(um_pre)  (zm-level)
            _C_uu_shr37 = _asarray(
                clubb_params[:, iC_uu_shr - 1], dtype=np.float64
            )[:, np.newaxis]
            _wp2_37     = _asarray(wp2, dtype=np.float64)
            _ddzt_um37  = _asarray(ddzt(jnp.asarray(_um_pre37), gr))
            _ddzt_vm37  = _asarray(ddzt(jnp.asarray(_vm_pre37), gr))
            _upwp_forcing37 = _C_uu_shr37 * _wp2_37 * _ddzt_um37  # (ngrdcol, nzm)
            _vpwp_forcing37 = _C_uu_shr37 * _wp2_37 * _ddzt_vm37  # (ngrdcol, nzm)
            # Nontraditional Coriolis term for upwp (advance_xm_wpxp_module.F90:3098-3106).
            if getattr(flags, 'l_ho_nontrad_coriolis', False):
                _fcy37 = _asarray(fcor_y, dtype=np.float64)[:, np.newaxis]
                _upwp_forcing37 = _upwp_forcing37 + _fcy37 * (
                    _asarray(up2, dtype=np.float64) - _wp2_37)

            # Coriolis + large-scale forcing for um/vm  (l_implemented=False)
            # um_tndcy = um_forcing - fcor * (vg - vm_pre)
            _fcor37 = _asarray(fcor, dtype=np.float64)[:, np.newaxis]  # (ngrdcol,1)
            _um_tndcy37 = (_asarray(um_forcing, dtype=np.float64)
                           - _fcor37 * (_asarray(vg, dtype=np.float64) - _vm_pre37))
            _vm_tndcy37 = (_asarray(vm_forcing, dtype=np.float64)
                           + _fcor37 * (_asarray(ug, dtype=np.float64) - _um_pre37))

            # diagnose_upxp: upthvp via upthlp + ep1*thv_ds*uprtp + rc_coef*uprcp
            # um_smth = zt2zm2zt(um)  (smoothed zt-level)
            _um_smth37 = _asarray(zt2zm2zt(jnp.asarray(_um_pre37), gr))
            _vm_smth37 = _asarray(zt2zm2zt(jnp.asarray(_vm_pre37), gr))

            _invrs_tau_C6_37  = _asarray(invrs_tau_C6_zm, dtype=np.float64)
            _tau_C6_37        = _xp.where(_invrs_tau_C6_37 > 1e-30,
                                         1.0 / _invrs_tau_C6_37, 0.0)
            _C6thl_37 = _xp.broadcast_to(
                _asarray(clubb_params[:, iC6thl - 1], dtype=np.float64)[:, np.newaxis],
                (ngrdcol, nzm)).copy()
            _C6rt_37  = _xp.broadcast_to(
                _asarray(clubb_params[:, iC6rt - 1], dtype=np.float64)[:, np.newaxis],
                (ngrdcol, nzm)).copy()
            _C7_37    = _asarray(Cx_fnc_Richardson, dtype=np.float64)  # (ngrdcol, nzm)

            _ddzt_thlm37    = _asarray(ddzt(jnp.asarray(_asarray(_thlm_pre11,  dtype=np.float64)), gr))
            _ddzt_rtm37     = _asarray(ddzt(jnp.asarray(_asarray(_rtm_pre11,   dtype=np.float64)), gr))
            _ddzt_um_smth37 = _asarray(ddzt(jnp.asarray(_um_smth37), gr))
            _ddzt_vm_smth37 = _asarray(ddzt(jnp.asarray(_vm_smth37), gr))

            def _diag_upxp37(ypwp, ddzt_xm, wpxp, ddzt_ym_smth, tau_C6, C6x, C7):
                """diagnose_upxp formula (Fortran k=2..nzm-1 → Python[:,1:-1])."""
                _interior = ((tau_C6[:, 1:-1] / C6x[:, 1:-1]) * (
                    -ypwp[:, 1:-1] * ddzt_xm[:, 1:-1]
                    - (1.0 - C7[:, 1:-1]) * wpxp[:, 1:-1] * ddzt_ym_smth[:, 1:-1]
                ))
                return _iset(_xp.zeros_like(ypwp), np.s_[:, 1:-1], _interior)  # boundaries stay 0

            _wpthlp_37 = _asarray(_wpthlp_pre11, dtype=np.float64)
            _wprtp_37  = _asarray(_wprtp_pre11,  dtype=np.float64)

            _upthlp37 = _diag_upxp37(_upwp_pre37, _ddzt_thlm37, _wpthlp_37, _ddzt_um_smth37, _tau_C6_37, _C6thl_37, _C7_37)
            _uprtp37  = _diag_upxp37(_upwp_pre37, _ddzt_rtm37,  _wprtp_37,  _ddzt_um_smth37, _tau_C6_37, _C6rt_37,  _C7_37)
            _vpthlp37 = _diag_upxp37(_vpwp_pre37, _ddzt_thlm37, _wpthlp_37, _ddzt_vm_smth37, _tau_C6_37, _C6thl_37, _C7_37)
            _vprtp37  = _diag_upxp37(_vpwp_pre37, _ddzt_rtm37,  _wprtp_37,  _ddzt_vm_smth37, _tau_C6_37, _C6rt_37,  _C7_37)

            _ep1_37   = float(ep1)
            _thv_ds37 = _asarray(thv_ds_zm, dtype=np.float64)
            _rc_cf37  = _asarray(rc_coef_zm, dtype=np.float64)
            _uprcp37  = _asarray(uprcp, dtype=np.float64)
            _vprcp37  = _asarray(vprcp, dtype=np.float64)

            _upthvp_tmp37 = _upthlp37 + _ep1_37 * _thv_ds37 * _uprtp37 + _rc_cf37 * _uprcp37
            _vpthvp_tmp37 = _vpthlp37 + _ep1_37 * _thv_ds37 * _vprtp37 + _rc_cf37 * _vprcp37
            # smooth via zm2zt2zm
            _upthvp37 = _asarray(zm2zt2zm(jnp.asarray(_upthvp_tmp37), gr))
            _vpthvp37 = _asarray(zm2zt2zm(jnp.asarray(_vpthvp_tmp37), gr))

            # Clipping floor: up2/vp2 (l_tke_aniso=True → use up2 directly, no relaxed floor)
            _up2_37 = jnp.asarray(_asarray(up2, dtype=np.float64))
            _vp2_37 = jnp.asarray(_asarray(vp2, dtype=np.float64))

            # Solve upwp/um pair — no clipping in solve; apply separately
            _upwp_preclip37, _um_jax37 = advance_xm_wpxp_jax(
                wpxp=jnp.asarray(_upwp_pre37),
                xm=jnp.asarray(_um_pre37),
                wpxp_forcing=jnp.asarray(_upwp_forcing37),
                xm_forcing=jnp.asarray(_um_tndcy37),
                C6_Skw_fnc=_C6rt_11,
                C7_Skw_fnc=_C7_11,
                invrs_tau_C6_zm=jnp.asarray(_invrs_tau_C6_37),
                lhs_ta_wpxp=_sh11['lhs_ta_wprtp'],
                lhs_diff_zm=_sh11['lhs_diff_zm'],
                lhs_ma_zm=_sh11['lhs_ma_zm'],
                lhs_ma_zt=_sh11['lhs_ma_zt'],
                lhs_ta_xm=_sh11['lhs_ta_xm'],
                lhs_tp=_sh11['lhs_tp'],
                lhs_ac_pr2=_sh11['lhs_ac_pr2'],
                thv_ds_zm=jnp.asarray(_thv_ds37),
                xpthvp=jnp.asarray(_upthvp37),
                wm_zt=jnp.asarray(_asarray(wm_zt, dtype=np.float64)),
                dt=float(dt_advance),
                gr=gr,
            )
            if getattr(flags, 'l_mono_flux_lim_um', False):
                _um_jax37, _upwp_preclip37 = _apply_mfl(
                    MFL_UM, _um_jax37, _upwp_preclip37, _um_pre37, up2,
                    _um_tndcy37, _W_TOL_SQD, _W_TOL)
            _upwp_jax37 = clip_covar_jax(_upwp_preclip37, _wp2_jax, _up2_37)

            # Solve vpwp/vm pair — no clipping in solve; apply separately
            _vpwp_preclip37, _vm_jax37 = advance_xm_wpxp_jax(
                wpxp=jnp.asarray(_vpwp_pre37),
                xm=jnp.asarray(_vm_pre37),
                wpxp_forcing=jnp.asarray(_vpwp_forcing37),
                xm_forcing=jnp.asarray(_vm_tndcy37),
                C6_Skw_fnc=_C6rt_11,
                C7_Skw_fnc=_C7_11,
                invrs_tau_C6_zm=jnp.asarray(_invrs_tau_C6_37),
                lhs_ta_wpxp=_sh11['lhs_ta_wprtp'],
                lhs_diff_zm=_sh11['lhs_diff_zm'],
                lhs_ma_zm=_sh11['lhs_ma_zm'],
                lhs_ma_zt=_sh11['lhs_ma_zt'],
                lhs_ta_xm=_sh11['lhs_ta_xm'],
                lhs_tp=_sh11['lhs_tp'],
                lhs_ac_pr2=_sh11['lhs_ac_pr2'],
                thv_ds_zm=jnp.asarray(_thv_ds37),
                xpthvp=jnp.asarray(_vpthvp37),
                wm_zt=jnp.asarray(_asarray(wm_zt, dtype=np.float64)),
                dt=float(dt_advance),
                gr=gr,
            )
            if getattr(flags, 'l_mono_flux_lim_vm', False):
                _vm_jax37, _vpwp_preclip37 = _apply_mfl(
                    MFL_VM, _vm_jax37, _vpwp_preclip37, _vm_pre37, vp2,
                    _vm_tndcy37, _W_TOL_SQD, _W_TOL)
            _vpwp_jax37 = clip_covar_jax(_vpwp_preclip37, _wp2_jax, _vp2_37)

            # Iter46: Fortran oracle removed; JAX results are the state.

            # ============================================================ #
            # Iter37: Override advance_xm_wpxp wind state with JAX values   #
            # um/upwp/vm/vpwp verified at machine epsilon (iter37).         #
            # ============================================================ #
            upwp = _asarray(_upwp_jax37, dtype=np.float64).copy()
            um   = _asarray(_um_jax37,   dtype=np.float64).copy()
            vpwp = _asarray(_vpwp_jax37, dtype=np.float64).copy()
            vm   = _asarray(_vm_jax37,   dtype=np.float64).copy()

            # Sponge-layer damping for um/vm (advance_xm_wpxp_module.F90:1095-1123,
            # under l_predict_upwp_vpwp + uv_sponge). No-op unless sponge_cfg enables 'uv'.
            um = _apply_sponge_field('uv', um, um_ref, gr, dt_advance, sponge_cfg)
            vm = _apply_sponge_field('uv', vm, vm_ref, gr, dt_advance, sponge_cfg)

            # uv nudging toward the initial reference profiles (advance_xm_wpxp_module.F90:
            # 1126-1151, under l_predict_upwp_vpwp + l_uv_nudge). No-op unless l_uv_nudge
            # (none of the cloud/dry cases use it; coriolis_test does, ts_nudge=dt → full reset).
            if getattr(flags, 'l_uv_nudge', False):
                _nf = float(dt_advance) / float(ts_nudge)
                um = um - (um - _asarray(um_ref, dtype=np.float64)) * _nf
                vm = vm - (vm - _asarray(vm_ref, dtype=np.float64)) * _nf

            # ============================================================ #
            # Iter69: advance_xm_wpxp budget term stats writes               #
            # ============================================================ #
            if l_sample and stats_writer is not None:
                _grav_69 = float(grav)
                _gamma_69 = 1.5  # gamma_over_implicit_ts

                # --- Pre-advance snapshots ---
                stats_writer.update("rtm_old",  _asarray(_rtm_pre11,   dtype=np.float64))
                stats_writer.update("thlm_old", _asarray(_thlm_pre11,  dtype=np.float64))

                # --- Forcings ---
                stats_writer.update("rtm_forcing",  _asarray(rtm_forcing,  dtype=np.float64))
                stats_writer.update("thlm_forcing", _asarray(thlm_forcing, dtype=np.float64))

                # --- Geostrophic and Coriolis terms for um/vm ---
                _fcor_69 = _asarray(fcor, dtype=np.float64)[:, np.newaxis]  # (ngrdcol,1)
                stats_writer.update("um_gf", -_fcor_69 * _asarray(vg, dtype=np.float64))
                stats_writer.update("vm_gf",  _fcor_69 * _asarray(ug, dtype=np.float64))
                stats_writer.update("um_cf",  _fcor_69 * _vm_pre37)
                stats_writer.update("vm_cf", -_fcor_69 * _um_pre37)

                # --- diagnose_upxp results ---
                stats_writer.update("upthlp", _upthlp37)
                stats_writer.update("uprtp",  _uprtp37)
                stats_writer.update("upthvp", _upthvp37)
                stats_writer.update("vpthlp", _vpthlp37)
                stats_writer.update("vprtp",  _vprtp37)
                stats_writer.update("vpthvp", _vpthvp37)

                # --- Shared LHS terms (already computed in _sh11) ---
                _lhs_ta_wpxp_69 = _asarray(_sh11['lhs_ta_wprtp'], dtype=np.float64)
                _lhs_diff_69    = _asarray(_sh11['lhs_diff_zm'],  dtype=np.float64)
                _lhs_tp_69      = _asarray(_sh11['lhs_tp'],       dtype=np.float64)
                _lhs_ta_xm_69   = _asarray(_sh11['lhs_ta_xm'],    dtype=np.float64)
                _thv_ds_69      = _asarray(thv_ds_zm,             dtype=np.float64)

                # lhs_pr1 for each pair (variable-specific)
                _invrs_tau_C6_69  = _asarray(invrs_tau_C6_zm, dtype=np.float64)
                _lhs_pr1_rtp_69   = _asarray(wpxp_term_pr1_lhs_jax(
                    jnp.asarray(_asarray(_C6rt_11)),
                    jnp.asarray(_invrs_tau_C6_69)), dtype=np.float64)
                _lhs_pr1_thl_69   = _asarray(wpxp_term_pr1_lhs_jax(
                    jnp.asarray(_asarray(_C6thl_11)),
                    jnp.asarray(_invrs_tau_C6_69)), dtype=np.float64)

                def _wpxp_budgets_69(name_bp, name_pr3, name_ta, name_pr1,
                                     name_tp, name_dp1, name_xm_ta,
                                     wpxp_pre, wpxp_new, xm_new, xpthvp_np,
                                     C7_np, lhs_pr1_np):
                    """Write wpxp and xm budget term stats for one variable pair."""
                    # bp: C7=0 → rhs_bp = (g/thv_ds)*x'thv'
                    _bp = _xp.zeros_like(wpxp_pre)
                    _bp = _iset(_bp, np.s_[:, 1:-1], _grav_69 / _thv_ds_69[:, 1:-1] * xpthvp_np[:, 1:-1])
                    stats_writer.update(name_bp, _bp)
                    # pr3: C7_plus_one → -(g/thv_ds)*C7*x'thv'
                    _pr3 = _xp.zeros_like(wpxp_pre)
                    _pr3 = _iset(_pr3, np.s_[:, 1:-1], -_grav_69 / _thv_ds_69[:, 1:-1] * C7_np[:, 1:-1] * xpthvp_np[:, 1:-1])
                    stats_writer.update(name_pr3, _pr3)
                    # ta: begin rhs_ta=0, update ta_over, finalize implicit
                    _ta_over = _xp.zeros_like(wpxp_pre)
                    _ta_over = _iset(_ta_over, np.s_[:, 1:-1], (1.0 - _gamma_69) * (
                        -_lhs_ta_wpxp_69[0, :, 1:-1] * wpxp_pre[:, 2:]
                        - _lhs_ta_wpxp_69[1, :, 1:-1] * wpxp_pre[:, 1:-1]
                        - _lhs_ta_wpxp_69[2, :, 1:-1] * wpxp_pre[:, :-2]
                    ))
                    _ta_impl = _xp.zeros_like(wpxp_new)
                    _ta_impl = _iset(_ta_impl, np.s_[:, 1:-1], (
                        -_gamma_69 * _lhs_ta_wpxp_69[0, :, 1:-1] * wpxp_new[:, 2:]
                        - _gamma_69 * _lhs_ta_wpxp_69[1, :, 1:-1] * wpxp_new[:, 1:-1]
                        - _gamma_69 * _lhs_ta_wpxp_69[2, :, 1:-1] * wpxp_new[:, :-2]
                    ))
                    stats_writer.update(name_ta, _ta_over + _ta_impl)
                    # pr1: begin -(1-gamma)*lhs_pr1*wpxp_pre, finalize -gamma*lhs_pr1*wpxp_new
                    _pr1 = _xp.zeros_like(wpxp_pre)
                    _pr1 = _iset(_pr1, np.s_[:, 1:-1], (
                        -(1.0 - _gamma_69) * lhs_pr1_np[:, 1:-1] * wpxp_pre[:, 1:-1]
                        - _gamma_69 * lhs_pr1_np[:, 1:-1] * wpxp_new[:, 1:-1]
                    ))
                    stats_writer.update(name_pr1, _pr1)
                    # tp: implicit, uses post-solve xm
                    _tp = _xp.zeros_like(wpxp_new)
                    _tp = _iset(_tp, np.s_[:, 1:-1], (
                        -_lhs_tp_69[1, :, 1:-1] * xm_new[:, :-1]
                        - _lhs_tp_69[0, :, 1:-1] * xm_new[:, 1:]
                    ))
                    stats_writer.update(name_tp, _tp)
                    # dp1: diffusion, implicit, uses post-solve wpxp
                    _dp1 = _xp.zeros_like(wpxp_new)
                    _dp1 = _iset(_dp1, np.s_[:, 1:-1], (
                        -_lhs_diff_69[2, :, 1:-1] * wpxp_new[:, :-2]
                        - _lhs_diff_69[1, :, 1:-1] * wpxp_new[:, 1:-1]
                        - _lhs_diff_69[0, :, 1:-1] * wpxp_new[:, 2:]
                    ))
                    stats_writer.update(name_dp1, _dp1)
                    # xm_ta: implicit, uses post-solve wpxp
                    _xm_ta = (
                        -_lhs_ta_xm_69[1] * wpxp_new[:, :-1]
                        - _lhs_ta_xm_69[0] * wpxp_new[:, 1:]
                    )
                    stats_writer.update(name_xm_ta, _xm_ta)

                _C7_np_69 = _asarray(_C7_11, dtype=np.float64)
                _dt_69    = float(dt_advance)

                # wprtp/rtm budget terms — use pre-clip wprtp for dp1/ta/pr1
                _wprtp_pc11 = _asarray(_wprtp_preclip11, dtype=np.float64)
                _wpxp_budgets_69(
                    "wprtp_bp", "wprtp_pr3", "wprtp_ta", "wprtp_pr1",
                    "wprtp_tp", "wprtp_dp1", "rtm_ta",
                    _asarray(_wprtp_pre11, dtype=np.float64),
                    _wprtp_pc11,
                    _asarray(_rtm_jax11,   dtype=np.float64),
                    _asarray(rtpthvp,      dtype=np.float64),
                    _C7_np_69, _lhs_pr1_rtp_69,
                )
                # wprtp_cl: clipping rate = (post_clip - pre_clip) / dt
                _wprtp_jax11_np = _asarray(_wprtp_jax11, dtype=np.float64)
                stats_writer.update("wprtp_cl", (_wprtp_jax11_np - _wprtp_pc11) / _dt_69)

                # wpthlp/thlm budget terms — use pre-clip wpthlp
                _wpthlp_pc11 = _asarray(_wpthlp_preclip11, dtype=np.float64)
                _wpxp_budgets_69(
                    "wpthlp_bp", "wpthlp_pr3", "wpthlp_ta", "wpthlp_pr1",
                    "wpthlp_tp", "wpthlp_dp1", "thlm_ta",
                    _asarray(_wpthlp_pre11, dtype=np.float64),
                    _wpthlp_pc11,
                    _asarray(_thlm_jax11,   dtype=np.float64),
                    _asarray(thlpthvp,      dtype=np.float64),
                    _C7_np_69, _lhs_pr1_thl_69,
                )
                _wpthlp_jax11_np = _asarray(_wpthlp_jax11, dtype=np.float64)
                stats_writer.update("wpthlp_cl", (_wpthlp_jax11_np - _wpthlp_pc11) / _dt_69)

                # upwp/um budget terms — use pre-clip upwp
                _upwp_pc37 = _asarray(_upwp_preclip37, dtype=np.float64)
                _wpxp_budgets_69(
                    "upwp_bp", "upwp_pr3", "upwp_ta", "upwp_pr1",
                    "upwp_tp", "upwp_dp1", "um_ta",
                    _asarray(_upwp_pre37, dtype=np.float64),
                    _upwp_pc37,
                    _asarray(_um_jax37,   dtype=np.float64),
                    _upthvp37,
                    _C7_np_69, _lhs_pr1_rtp_69,
                )
                _upwp_jax37_np = _asarray(_upwp_jax37, dtype=np.float64)
                stats_writer.update("upwp_cl", (_upwp_jax37_np - _upwp_pc37) / _dt_69)

                # vpwp/vm budget terms — use pre-clip vpwp
                _vpwp_pc37 = _asarray(_vpwp_preclip37, dtype=np.float64)
                _wpxp_budgets_69(
                    "vpwp_bp", "vpwp_pr3", "vpwp_ta", "vpwp_pr1",
                    "vpwp_tp", "vpwp_dp1", "vm_ta",
                    _asarray(_vpwp_pre37, dtype=np.float64),
                    _vpwp_pc37,
                    _asarray(_vm_jax37,   dtype=np.float64),
                    _vpthvp37,
                    _C7_np_69, _lhs_pr1_rtp_69,
                )
                _vpwp_jax37_np = _asarray(_vpwp_jax37, dtype=np.float64)
                stats_writer.update("vpwp_cl", (_vpwp_jax37_np - _vpwp_pc37) / _dt_69)

                # upwp_pr4/vpwp_pr4: C_uu_shr * wp2 * ddzt_um/vm (zt-level gradient)
                # Mirrors Fortran advance_xm_wpxp_module: tmp_zm=C_uu_shr*wp2*ddzt_um
                stats_writer.update("upwp_pr4", _C_uu_shr37 * _wp2_37 * _ddzt_um37)
                stats_writer.update("vpwp_pr4", _C_uu_shr37 * _wp2_37 * _ddzt_vm37)

                # ---- MFL stats: mean_w_up/down + rtm/thlm MFL bounds ----
                # Uses previous-timestep PDF params (w_1_zm etc.) per ARM post-advance path.
                # Fortran: mono_flux_limiter.F90 / calc_turb_adv_range / mean_vert_vel_up_down
                _l_mfl_rt  = getattr(flags, 'l_mono_flux_lim_rtm',  False)
                _l_mfl_thl = getattr(flags, 'l_mono_flux_lim_thlm', False)
                _l_mfl_spk = getattr(flags, 'l_mono_flux_lim_spikefix', True)

                if _l_mfl_rt or _l_mfl_thl:
                    _gd_mfl  = float(gr.grid_dir)
                    _idt_mfl = 1.0 / float(dt_advance)
                    _dt_mfl  = float(dt_advance)
                    _dzm_mfl = _asarray(gr.dzm, dtype=np.float64)
                    _dzt_mfl = (_asarray(gr.zm)[:, 1:] - _asarray(gr.zm)[:, :-1]) * _gd_mfl

                    # mean_w_up/down using PREVIOUS-timestep PDF params (Fortran: mean_vert_vel_up_down)
                    _w1j   = jnp.asarray(w_1_zm,        dtype=jnp.float64)
                    _w2j   = jnp.asarray(w_2_zm,        dtype=jnp.float64)
                    _v1j   = jnp.asarray(varnce_w_1_zm, dtype=jnp.float64)
                    _v2j   = jnp.asarray(varnce_w_2_zm, dtype=jnp.float64)
                    _mfj   = jnp.asarray(mixt_frac_zm,  dtype=jnp.float64)
                    _wm_mfl = jnp.asarray(_gd_mfl * _dzm_mfl * _idt_mfl)

                    def _mwc_mfl(wi, vi, wm):
                        sig    = _safe_sqrt(vi)
                        sig_s  = jnp.where(sig > 0.0, sig, 1.0)
                        z      = (0.0 - wi) / (jnp.sqrt(2.0) * sig_s)
                        sq2pi  = jnp.sqrt(2.0 * jnp.pi)
                        ev     = jnp.exp(-z ** 2)
                        ef     = jax.scipy.special.erf(z)
                        too_weak = jnp.abs(wi) + 3.0 * sig <= wm
                        all_dn   = (~too_weak) & (wi + 3.0 * sig <= 0.0)
                        all_up   = (~too_weak) & (~all_dn) & (wi - 3.0 * sig >= 0.0)
                        mwd_m  = -sig / sq2pi * ev + wi * 0.5 * (1.0 + ef)
                        mwu_m  =  sig / sq2pi * ev + wi * 0.5 * (1.0 - ef)
                        mwd = jnp.where(too_weak, 0.0,
                              jnp.where(all_dn, wi,
                              jnp.where(all_up, 0.0, mwd_m)))
                        mwu = jnp.where(too_weak, 0.0,
                              jnp.where(all_dn, 0.0,
                              jnp.where(all_up, wi,  mwu_m)))
                        mwd = mwd.at[:, 0].set(0.0).at[:, -1].set(0.0)
                        mwu = mwu.at[:, 0].set(0.0).at[:, -1].set(0.0)
                        return mwd, mwu

                    _mwd1, _mwu1 = _mwc_mfl(_w1j, _v1j, _wm_mfl)
                    _mwd2, _mwu2 = _mwc_mfl(_w2j, _v2j, _wm_mfl)
                    _vvd_mfl = _asarray(_mfj * _mwd1 + (1.0 - _mfj) * _mwd2, dtype=np.float64)
                    _vvu_mfl = _asarray(_mfj * _mwu1 + (1.0 - _mfj) * _mwu2, dtype=np.float64)
                    stats_writer.update("mean_w_down", _vvd_mfl)
                    stats_writer.update("mean_w_up",   _vvu_mfl)

                    # calc_turb_adv_range: k=0,nzt-2,nzt-1 set as boundary; k=1..nzt-3 computed
                    # Ascending grid: j_adj=j+1 (zm) for downward, j_adj=j (zm) for upward
                    _ll_mfl = np.zeros((ngrdcol, nzt), dtype=np.int64)
                    _hl_mfl = np.zeros((ngrdcol, nzt), dtype=np.int64)
                    for _im in range(ngrdcol):
                        _ll_mfl[_im, 0]     = 0;      _hl_mfl[_im, 0]     = 0
                        _ll_mfl[_im, nzt-2] = nzt-2;  _hl_mfl[_im, nzt-2] = nzt-1
                        _ll_mfl[_im, nzt-1] = nzt-1;  _hl_mfl[_im, nzt-1] = nzt-1
                        for _km in range(1, nzt - 2):
                            _da = 0.0
                            for _jm in range(_km - 1, -1, -1):
                                _ll_mfl = _iset(_ll_mfl, np.s_[_im, _km], _jm)
                                _ja = _jm + 1   # zm index ascending
                                _vu = _vvu_mfl[_im, _ja]
                                if _vu > 0.0:
                                    _da += _gd_mfl * _dzm_mfl[_im, _ja] / _vu
                                    if _da >= _dt_mfl:
                                        break
                                else:
                                    _ll_mfl = _iset(_ll_mfl, np.s_[_im, _km], min(_jm + 1, nzt - 1))
                                    break
                            _da = 0.0
                            for _jm in range(_km + 1, nzt):
                                _hl_mfl = _iset(_hl_mfl, np.s_[_im, _km], _jm)
                                _ja = _jm      # zm index ascending
                                _vd = _vvd_mfl[_im, _ja]
                                if _vd < 0.0:
                                    _da += -_gd_mfl * _dzm_mfl[_im, _ja] / _vd
                                    if _da >= _dt_mfl:
                                        break
                                else:
                                    _hl_mfl = _iset(_hl_mfl, np.s_[_im, _km], max(_jm - 1, 0))
                                    break

                    _rdszt = _asarray(rho_ds_zt,       dtype=np.float64)
                    _rdszm = _asarray(rho_ds_zm,       dtype=np.float64)
                    _irdzm = _asarray(invrs_rho_ds_zm, dtype=np.float64)

                    def _mfl_scalar(xm_old_np, xm_new, xm_frc_np,
                                    xp2_zm, xm_tol_v, max_xp2_v, xp2_thr,
                                    wpxp_in,
                                    nm_xe, nm_xx, nm_wta, nm_xmn, nm_xmx,
                                    nm_we, nm_wx, nm_wmn, nm_wmx,
                                    l_spk):
                        _xm_n = _asarray(xm_new,   dtype=np.float64)
                        _wp   = _asarray(wpxp_in,  dtype=np.float64)
                        # Entry stats
                        stats_writer.update(nm_xe, _xm_n)
                        stats_writer.update(nm_we, _wp)
                        # xm_without_ta = xm_old + dt * xm_forcing  (m_adv_term = 0)
                        _wta = (_asarray(xm_old_np, dtype=np.float64)
                                + _dt_mfl * _asarray(xm_frc_np, dtype=np.float64))
                        stats_writer.update(nm_wta, _wta)
                        # Clip xp2 zm → zt
                        _xp2z = _xp.clip(
                            _asarray(zm2zt_jax(jnp.asarray(xp2_zm), gr), dtype=np.float64),
                            xp2_thr, max_xp2_v)
                        _mxd = _xp.maximum(2.0 * _xp.sqrt(_xp2z), xm_tol_v)
                        _mnl = _xp.maximum(_wta - _mxd, 0.0)   # positive-definite (rt, thl)
                        _mxl = _wta + _mxd
                        # Multi-level min/max over advection range
                        _mna = _xp.empty_like(_wta)
                        _mxa = _xp.empty_like(_wta)
                        for _ii in range(ngrdcol):
                            for _kk in range(nzt):
                                _lo = int(_ll_mfl[_ii, _kk])
                                _hi = int(_hl_mfl[_ii, _kk])
                                _mna = _iset(_mna, np.s_[_ii, _kk], _xp.min(_mnl[_ii, _lo:_hi+1]))
                                _mxa = _iset(_mxa, np.s_[_ii, _kk], _xp.max(_mxl[_ii, _lo:_hi+1]))
                        stats_writer.update(nm_xmn, _mna)
                        stats_writer.update(nm_xmx, _mxa)
                        # wpxp MFL bounds (Fortran lines 672-756)
                        _thr_zt = _idt_mfl * _gd_mfl * _dzt_mfl * (_wta - _mna)
                        _mxt_zt = _rdszt * _thr_zt
                        _mnt_zt = _rdszt * _idt_mfl * _gd_mfl * _dzt_mfl * (_wta - _mxa)
                        _thr_zm = _asarray(zt2zm_jax(jnp.asarray(_thr_zt), gr), dtype=np.float64)
                        _wpmx   = np.zeros((ngrdcol, nzm))
                        _wpmn   = np.zeros((ngrdcol, nzm))
                        for _ii in range(ngrdcol):
                            for _kk in range(1, nzm - 1):
                                _km1 = _kk - 1   # k - grid_dir_indx (ascending)
                                _kzt = _kk - 1   # k_zt (ascending)
                                if (l_spk
                                        and abs(_wp[_ii, _km1]) > _thr_zm[_ii, _km1]
                                        and _wp[_ii, _km1] < 0.0):
                                    _wpmx = _iset(_wpmx, np.s_[_ii, _kk], 0.0)
                                else:
                                    _wpmx = _iset(_wpmx, np.s_[_ii, _kk], _irdzm[_ii, _kk] * (
                                        _mxt_zt[_ii, _kzt] + _rdszm[_ii, _km1] * _wp[_ii, _km1]))
                                _wpmn = _iset(_wpmn, np.s_[_ii, _kk], _irdzm[_ii, _kk] * (
                                    _mnt_zt[_ii, _kzt] + _rdszm[_ii, _km1] * _wp[_ii, _km1]))
                        stats_writer.update(nm_wmx, _wpmx)
                        stats_writer.update(nm_wmn, _wpmn)
                        # Exit stats (= entry for ARM; MFL rarely adjusts)
                        stats_writer.update(nm_xx, _xm_n)
                        stats_writer.update(nm_wx, _wp)

                    if _l_mfl_rt:
                        _mfl_scalar(
                            _rtm_pre11, _rtm_jax11, rtm_forcing,
                            rtp2, 1e-4, 5e-6, float(rt_tol**2),
                            _wprtp_preclip11,
                            "rtm_enter_mfl", "rtm_exit_mfl", "rtm_without_ta",
                            "rtm_mfl_min", "rtm_mfl_max",
                            "wprtp_enter_mfl", "wprtp_exit_mfl",
                            "wprtp_mfl_min", "wprtp_mfl_max",
                            _l_mfl_spk,
                        )
                    if _l_mfl_thl:
                        _mfl_scalar(
                            _thlm_pre11, _thlm_jax11, thlm_forcing,
                            thlp2, 0.2, 5.0, float(thl_tol**2),
                            _wpthlp_preclip11,
                            "thlm_enter_mfl", "thlm_exit_mfl", "thlm_without_ta",
                            "thlm_mfl_min", "thlm_mfl_max",
                            "wpthlp_enter_mfl", "wpthlp_exit_mfl",
                            "wpthlp_mfl_min", "wpthlp_mfl_max",
                            False,
                        )

        elif advance_iter == order_xp2_xpyp_val:
            # ============================================================ #
            # Block M+++: Iteration 6 — xpyp_term_ta_pdf_lhs_jax (JAX-only)
            # Fortran oracle removed Iter51. Uses coef_wprtp2_implicit as a
            # representative xpyp coefficient for the iter8 numpy-ref assembly.
            # ============================================================ #
            _coef_ta = _asarray(pdf_implicit_coefs_terms.coef_wprtp2_implicit)
            _lhs_ta_jax = _asarray(xpyp_term_ta_pdf_lhs_jax(
                jnp.array(_coef_ta),
                jnp.array(rho_ds_zt),
                jnp.array(invrs_rho_ds_zm),
                gr,
            ))

            # ============================================================ #
            # Block M+7: dp1 pressure-damping LHS term (_Cn_np/_inv_tau/_dp1_ref), assembled in
            # numpy here and fed (jnp.array'd) into the M+10 xp2_xpyp solve.
            # Uses C2rt (uniform in z) * invrs_tau_xp2_zm, boundaries zeroed.
            # ============================================================ #
            _c2rt = float(clubb_params[0, iC2rt - 1])
            _Cn_np = np.full((ngrdcol, nzm), _c2rt, dtype=np.float64)
            _inv_tau = _asarray(invrs_tau_xp2_zm)
            # numpy reference: interior = Cn * invrs_tau_zm, boundaries = 0
            _dp1_ref = _Cn_np * _inv_tau
            _dp1_ref = _iset(_dp1_ref, np.s_[:, 0], 0.0)
            _dp1_ref = _iset(_dp1_ref, np.s_[:, -1], 0.0)

            # ============================================================ #
            # Block M+8: mean-advection LHS term (_lhs_ma_f) for the M+10 xp2_xpyp solve.
            # Assemble all components via JAX and compare against numpy reference.
            # Uses rtp2 term components as representative inputs.
            # ============================================================ #
            _gamma = 1.5   # gamma_over_implicit_ts
            _dt_adv = float(dt_advance)
            # xp2/xpyp diffusion uses Kw2 = c_K2 * Kh_zt (Fortran advance_xp2_xpyp_module.F90 line 547)
            # and background nu2 coefficient (nu_vert_res_dep%nu2 = clubb_params[inu2-1] = 1.0 m²/s)
            _c_K2 = float(clubb_params[0, ic_K2 - 1])  # = 0.025
            _Kw2 = _c_K2 * Kh_zt  # shape (ngrdcol, nzt)
            _nu2_xp2 = _asarray(nu_vert_res_dep.nu2, dtype=np.float64)  # background eddy diff [m²/s]
            # Iter53: Fortran oracle removed; use JAX (validated machine-epsilon by iter4/5)
            _lhs_diff_f = _asarray(diffusion_zm_lhs_jax(
                jnp.asarray(_Kw2), jnp.asarray(_nu2_xp2),
                jnp.asarray(invrs_rho_ds_zm), jnp.asarray(rho_ds_zt), gr,
            ))
            _lhs_ma_f = _asarray(term_ma_zm_lhs_jax(jnp.asarray(wm_zm), gr))
            # lhs_ta already computed above

            # ============================================================ #
            # Block M+9: turbulent-advection RHS term (_rhs9_ref) for the M+10 xp2_xpyp solve.
            # Computes RHS for the rtp2 case (xam=xbm=rtm, wpxap=wpxbp=wprtp).
            # rhs_ta from JAX xpyp_term_ta_pdf_rhs_jax (Fortran oracle removed Iter51).
            # Comparison is JAX vs numpy reference (same inputs), target ≤ 1e-15.
            # ============================================================ #
            _term_wp_explicit = _asarray(
                pdf_implicit_coefs_terms.term_wprtp2_explicit
            )
            _rhs_ta9 = _asarray(xpyp_term_ta_pdf_rhs_jax(
                jnp.asarray(_term_wp_explicit),
                jnp.asarray(rho_ds_zt),
                jnp.asarray(invrs_rho_ds_zm),
                gr,
            ))
            # numpy reference for the rtp2 case
            _threshold9 = float(rt_tol ** 2)
            _rtp2_np = _asarray(rtp2, dtype=np.float64).copy()
            _rtm_np  = _asarray(rtm,  dtype=np.float64).copy()
            _wprtp_np = _asarray(wprtp, dtype=np.float64).copy()
            _invrs_dzm_np = _asarray(gr.invrs_dzm, dtype=np.float64).copy()
            _rtp2_forcing_np = _asarray(rtp2_forcing, dtype=np.float64).copy()
            _one_minus_gamma9 = 1.0 - _gamma
            # term_tp_rhs (rtp2: xam=xbm=rtm, wpxap=wpxbp=wprtp)
            _rhs_tp9 = (
                -_wprtp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1] * (_rtm_np[:, 1:] - _rtm_np[:, :-1])
                - _wprtp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1] * (_rtm_np[:, 1:] - _rtm_np[:, :-1])
            )
            # term_dp1 components (interior)
            _rhs_dp1_9 = _Cn_np[:, 1:-1] * _inv_tau[:, 1:-1] * _threshold9
            _lhs_dp1_9 = _Cn_np[:, 1:-1] * _inv_tau[:, 1:-1]
            # REFACTOR B4 (iter19): removed the dead `_rhs9_ref` numpy shadow-comparison RHS — it was
            # computed but NEVER used (the real rtp2 solve below uses _rtp2_np/_rtm_np via
            # advance_xp2_xpyp_jax). Eliminating this bit-faithfulness scaffolding also unblocked grad here.
            # JAX RHS

            # ============================================================ #
            # Block M+10: Iteration 10 — full JAX advance_xp2_xpyp solve.
            # Matches Fortran upwind or centered path to produce comparable
            # LHS/RHS, then solves with JAX and compares against Fortran output.
            # fill_holes + clip_variance in Fortran are no-ops for ARM.
            # ============================================================ #
            # Iter36: save pre-call up2/vp2 for JAX shadow comparison
            _up2_pre36 = _asarray(up2, dtype=np.float64).copy()
            _vp2_pre36 = _asarray(vp2, dtype=np.float64).copy()
            _thlm_np  = _asarray(thlm,  dtype=np.float64).copy()
            _thlp2_np = _asarray(thlp2, dtype=np.float64).copy()
            _wpthlp_np = _asarray(wpthlp, dtype=np.float64).copy()
            _thlp2_forcing_np = _asarray(thlp2_forcing, dtype=np.float64).copy()
            _rtpthlp_np = _asarray(rtpthlp, dtype=np.float64).copy()
            _rtpthlp_forcing_np = _asarray(rtpthlp_forcing, dtype=np.float64).copy()
            _threshold_thlp2 = float(thl_tol ** 2)
            _threshold_rtpthlp = float(zero_threshold)
            if flags.l_upwind_xpyp_ta:
                # Upwind ADG1 path (Fortran advance_xp2_xpyp_module.F90 lines 4500+).
                # All signs use sign(wp3_on_wp2) — same for LHS and all three RHS.
                # wp_coef = (1 - beta/3) * a1^2 * wp3_on_wp2 / wp2  (lines 4237-4238)
                # coef_zm = (1/3)*beta*a1*wp3_on_wp2  (lines 4530)
                # term_explicit_zm: wprtp^2, wpthlp^2, wprtp*wpthlp (lines 4625/4681/4740)
                _wp3_on_wp2_np = _asarray(wp3_on_wp2, dtype=np.float64)
                _sigma_sqd_w_np = _asarray(sigma_sqd_w, dtype=np.float64)
                _a1_zm = 1.0 / (1.0 - _sigma_sqd_w_np)
                _beta_val = _asarray(clubb_params[:, ibeta - 1], dtype=np.float64)
                _wp2_np = _asarray(wp2, dtype=np.float64)
                _wprtp_zm_np = _asarray(wprtp, dtype=np.float64)
                _wpthlp_zm_np = _asarray(wpthlp, dtype=np.float64)
                # sign(wp3_on_wp2) is the single sign used by LHS and all RHS
                _sgn10 = _xp.where(_wp3_on_wp2_np >= 0, 1.0, -1.0)
                # LHS: coef_impl_zm = (1/3)*beta*a1*wp3_on_wp2
                _coef10_zm = (1.0/3.0) * _beta_val[:, np.newaxis] * _a1_zm * _wp3_on_wp2_np
                _coef10_zt_dummy = np.zeros((ngrdcol, nzt), dtype=np.float64)
                # Iter58: JAX upwind LHS (Fortran oracle removed)
                _lhs_ta_10 = _asarray(xpyp_term_ta_pdf_lhs_upwind_jax(
                    jnp.asarray(rho_ds_zm),
                    jnp.asarray(invrs_rho_ds_zm),
                    jnp.asarray(_sgn10),
                    jnp.asarray(_coef10_zm),
                    gr,
                    grid_dir=float(gr.grid_dir),
                ), dtype=np.float64)
                # wp_coef = (1 - beta/3)*a1^2*wp3_on_wp2/wp2
                _wp_coef = ((1.0 - (1.0/3.0)*_beta_val[:, np.newaxis])
                            * _a1_zm**2 * _wp3_on_wp2_np / _wp2_np)
                # Iter58: JAX upwind RHS (Fortran oracle removed)
                _sgn10_j = jnp.asarray(_sgn10)
                _rho_ds_zm_j = jnp.asarray(rho_ds_zm)
                _irho_ds_zm_j = jnp.asarray(invrs_rho_ds_zm)
                _gdir = float(gr.grid_dir)
                _rhs_ta10_rtp2 = _asarray(xpyp_term_ta_pdf_rhs_upwind_jax(
                    _rho_ds_zm_j, _irho_ds_zm_j, _sgn10_j,
                    jnp.asarray(_wp_coef * _wprtp_zm_np**2),
                    gr, grid_dir=_gdir,
                ), dtype=np.float64)
                _rhs_ta10_thlp2 = _asarray(xpyp_term_ta_pdf_rhs_upwind_jax(
                    _rho_ds_zm_j, _irho_ds_zm_j, _sgn10_j,
                    jnp.asarray(_wp_coef * _wpthlp_zm_np**2),
                    gr, grid_dir=_gdir,
                ), dtype=np.float64)
                _rhs_ta10_rtpthlp = _asarray(xpyp_term_ta_pdf_rhs_upwind_jax(
                    _rho_ds_zm_j, _irho_ds_zm_j, _sgn10_j,
                    jnp.asarray(_wp_coef * _wprtp_zm_np * _wpthlp_zm_np),
                    gr, grid_dir=_gdir,
                ), dtype=np.float64)
            else:
                # Centered ADG1 path: compute ta terms from primitives.
                # For ADG1 (ARM case), pdf_implicit_coefs_terms.coef_wprtp2_implicit
                # and term_wpthlp2_explicit are ZERO (not populated by ADG1 PDF closure).
                # The Fortran advance_xp2_xpyp internally computes these from:
                #   coef_zt = (1/3)*beta*a1_zt*wp3_on_wp2_zt
                #   term_zt = wp_coef_zt * wpthlp_zt^2
                # where wp_coef_zt = (1-beta/3)*a1_zt^2*wp3_on_wp2_zt/wp2_zt
                # (Fortran: calc_xp2_xpyp_ta_terms, ADG1 path, l_upwind=False)
                _a1_zm_10 = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
                _a1_zt_10 = _asarray(zm2zt(jnp.array(_a1_zm_10), gr))
                _beta_10 = _asarray(clubb_params[:, ibeta - 1], dtype=np.float64)
                _wp3_on_wp2_zt_10 = _asarray(wp3_on_wp2_zt, dtype=np.float64)
                _wp2_zt_10 = _asarray(wp2_zt, dtype=np.float64)
                # coef_zt = (1/3)*beta*a1_zt*wp3_on_wp2_zt
                _coef_ta_zt_10 = ((1.0/3.0) * _beta_10[:, np.newaxis]
                                  * _a1_zt_10 * _wp3_on_wp2_zt_10)
                # Iter51: Fortran xpyp_term_ta_pdf_lhs/rhs oracle removed; use JAX.
                _lhs_ta_10 = _asarray(xpyp_term_ta_pdf_lhs_jax(
                    jnp.asarray(_coef_ta_zt_10),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
                # wp_coef_zt = (1-beta/3)*a1_zt^2*wp3_on_wp2_zt/wp2_zt
                _wp_coef_zt_10 = ((1.0 - (1.0/3.0)*_beta_10[:, np.newaxis])
                                  * _a1_zt_10**2 * _wp3_on_wp2_zt_10 / _wp2_zt_10)
                # zm2zt interpolation of wprtp and wpthlp
                _wprtp_zt_10 = _asarray(zm2zt(
                    jnp.array(_asarray(wprtp, dtype=np.float64)), gr))
                _wpthlp_zt_10 = _asarray(zm2zt(
                    jnp.array(_asarray(wpthlp, dtype=np.float64)), gr))
                # explicit terms at zt grid
                _term_rtp2_zt_10    = _wp_coef_zt_10 * _wprtp_zt_10**2
                _term_thlp2_zt_10   = _wp_coef_zt_10 * _wpthlp_zt_10**2
                _term_rtpthlp_zt_10 = _wp_coef_zt_10 * _wprtp_zt_10 * _wpthlp_zt_10
                _zeros_zm_10 = np.zeros((ngrdcol, nzm), dtype=np.float64)
                _rhs_ta10_rtp2 = _asarray(xpyp_term_ta_pdf_rhs_jax(
                    jnp.asarray(_term_rtp2_zt_10),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
                _rhs_ta10_thlp2 = _asarray(xpyp_term_ta_pdf_rhs_jax(
                    jnp.asarray(_term_thlp2_zt_10),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
                _rhs_ta10_rtpthlp = _asarray(xpyp_term_ta_pdf_rhs_jax(
                    jnp.asarray(_term_rtpthlp_zt_10),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
            # Rebuild full LHS with correct (upwind or centered) lhs_ta
            _lhs10 = _asarray(xp2_xpyp_lhs_jax(
                jnp.array(_lhs_ta_10),
                jnp.array(_lhs_ma_f),
                jnp.array(_lhs_diff_f),
                jnp.array(_dp1_ref * _gamma),
                _dt_adv,
            ))
            # Build RHS for each variable (lhs_ta inside xp2_xpyp_rhs_jax is for
            # the over-implicit TA correction, must match the LHS used for the solve)
            _rhs10_rtp2 = _asarray(xp2_xpyp_rhs_jax(
                jnp.array(_lhs_ta_10), jnp.array(_rhs_ta10_rtp2),
                jnp.array(_Cn_np), jnp.array(_inv_tau), float(rt_tol ** 2),
                jnp.array(_rtp2_np), jnp.array(_rtm_np), jnp.array(_rtm_np),
                jnp.array(_wprtp_np), jnp.array(_wprtp_np),
                jnp.array(_invrs_dzm_np), jnp.array(_rtp2_forcing_np), _dt_adv,
            ))
            _rhs10_thlp2 = _asarray(xp2_xpyp_rhs_jax(
                jnp.array(_lhs_ta_10), jnp.array(_rhs_ta10_thlp2),
                jnp.array(_Cn_np), jnp.array(_inv_tau), _threshold_thlp2,
                jnp.array(_thlp2_np), jnp.array(_thlm_np), jnp.array(_thlm_np),
                jnp.array(_wpthlp_np), jnp.array(_wpthlp_np),
                jnp.array(_invrs_dzm_np), jnp.array(_thlp2_forcing_np), _dt_adv,
            ))
            _rhs10_rtpthlp = _asarray(xp2_xpyp_rhs_jax(
                jnp.array(_lhs_ta_10), jnp.array(_rhs_ta10_rtpthlp),
                jnp.array(_Cn_np), jnp.array(_inv_tau), _threshold_rtpthlp,
                jnp.array(_rtpthlp_np), jnp.array(_rtm_np), jnp.array(_thlm_np),
                jnp.array(_wprtp_np), jnp.array(_wpthlp_np),
                jnp.array(_invrs_dzm_np), jnp.array(_rtpthlp_forcing_np), _dt_adv,
            ))
            # Solve with single JAX LHS
            _lhs10_jax = jnp.array(_lhs10)
            _soln10_rtp2    = _asarray(tridiag_lu_solve_jax(_lhs10_jax, jnp.array(_rhs10_rtp2)))
            _soln10_thlp2   = _asarray(tridiag_lu_solve_jax(_lhs10_jax, jnp.array(_rhs10_thlp2)))
            _soln10_rtpthlp = _asarray(tridiag_lu_solve_jax(_lhs10_jax, jnp.array(_rhs10_rtpthlp)))
            # Apply l_lmm_stepping blending (0.5 * old + 0.5 * solution)
            if flags.l_lmm_stepping:
                _rtp2_jax10    = 0.5 * (_rtp2_np    + _soln10_rtp2)
                _thlp2_jax10   = 0.5 * (_thlp2_np   + _soln10_thlp2)
                _rtpthlp_jax10 = 0.5 * (_rtpthlp_np + _soln10_rtpthlp)
            else:
                _rtp2_jax10    = _soln10_rtp2
                _thlp2_jax10   = _soln10_thlp2
                _rtpthlp_jax10 = _soln10_rtpthlp

            # Iter47: Fortran advance_xp2_xpyp oracle removed; Iter54: dead diagnostic blocks removed.
            # --- M+10 post-Fortran comparison: apply fill_holes to JAX solutions to match ---
            # Fortran applies pos_definite_variances (fill_holes) to rtp2 and thlp2 after solve.
            _hf_lower = gr.k_lb_zm + gr.grid_dir_indx  # Python 0-based
            _hf_upper = gr.k_ub_zm - gr.grid_dir_indx  # Python 0-based
            # Iter51: Fortran fill_holes_vertical oracle removed; JAX-only.
            _rtp2_jax10_fh = _asarray(fill_holes_vertical_jax(
                field=jnp.asarray(_rtp2_jax10.copy()),
                rho_ds=jnp.asarray(rho_ds_zm),
                dz=jnp.asarray(gr.dzm),
                threshold=float(rt_tol**2),
                lower_k=_hf_lower, upper_k=_hf_upper,
                fill_holes_type=flags.fill_holes_type,
            ))
            _thlp2_jax10_fh = _asarray(fill_holes_vertical_jax(
                field=jnp.asarray(_thlp2_jax10.copy()),
                rho_ds=jnp.asarray(rho_ds_zm),
                dz=jnp.asarray(gr.dzm),
                threshold=float(thl_tol**2),
                lower_k=_hf_lower, upper_k=_hf_upper,
                fill_holes_type=flags.fill_holes_type,
            ))

            # Apply clip_variance (Fortran does this after pos_definite_variances).
            # When l_min_xp2_from_corr_wx, threshold is boosted to wpthlp^2/(wp2*corr^2)
            # to keep |corr(w,thl)| <= max_mag_correlation_flux=0.99 (constants_clubb.F90:348).
            _thlp2_jax10_cv = _thlp2_jax10_fh.copy()
            _rtp2_jax10_cv = _rtp2_jax10_fh.copy()
            if flags.l_min_xp2_from_corr_wx:
                _wp2_clip = _asarray(wp2, dtype=np.float64)
                _wpthlp_clip = _wpthlp_np  # zm-level, set at line ~1312
                _wprtp_clip = _asarray(wprtp, dtype=np.float64)
                _max_corr2 = 0.99**2  # max_mag_correlation_flux^2
                _thr_thlp2 = _xp.maximum(_threshold_thlp2,
                                        _wpthlp_clip**2 / (_wp2_clip * _max_corr2))
                _thr_rtp2 = _xp.maximum(float(rt_tol**2),
                                       _wprtp_clip**2 / (_wp2_clip * _max_corr2))
                _thlp2_jax10_cv = _iset(_thlp2_jax10_cv, np.s_[:, :-1],
                                        _xp.maximum(_thlp2_jax10_fh[:, :-1], _thr_thlp2[:, :-1]))
                _rtp2_jax10_cv = _iset(_rtp2_jax10_cv, np.s_[:, :-1],
                                       _xp.maximum(_rtp2_jax10_fh[:, :-1], _thr_rtp2[:, :-1]))
            else:
                _thlp2_jax10_cv = _iset(_thlp2_jax10_cv, np.s_[:, :-1],
                                        _xp.maximum(_thlp2_jax10_fh[:, :-1], _threshold_thlp2))
                _rtp2_jax10_cv = _iset(_rtp2_jax10_cv, np.s_[:, :-1],
                                       _xp.maximum(_rtp2_jax10_fh[:, :-1], float(rt_tol**2)))
            # Apply clip_covar to rtpthlp (Cauchy-Schwarz: |rtpthlp| <= 0.99*sqrt(rtp2*thlp2))
            # Iter43/47: JAX clip_covar for rtpthlp (Fortran oracle removed Iter47)
            _rtpthlp_jax10_clip = _asarray(clip_covar_jax(
                wpxp=jnp.asarray(_rtpthlp_jax10),
                wp2=jnp.asarray(_rtp2_jax10_cv),
                xp2=jnp.asarray(_thlp2_jax10_cv),
                max_mag_corr=0.99,
            ))
            # Iter47: Fortran oracle removed; JAX results are the state.

            # ============================================================ #
            # Budget stats: rtp2/thlp2/rtpthlp + scalar TA terms           #
            # ============================================================ #
            if l_sample and stats_writer is not None:
                _dt10 = float(dt_advance)
                _g10 = _gamma   # 1.5
                _omg10 = 1.0 - _g10  # -0.5

                # term_wprtp2/wpthlp2/wprtpthlp_explicit:
                # Fortran calc_xp2_xpyp_ta_terms lines 4603-4617 OVERWRITES first assignment
                # (line 4312 which set term=wpthlp2 input) with:
                #   a1_coef_zt = zm2zt(1/(1-sigma_sqd_w))
                #   wp_coef_zt = (1-beta/3)*a1_zt^2*wp3_on_wp2_zt/wp2_zt
                #   term_wpthlp2_explicit = wp_coef_zt * wpthlp_zt^2
                # This runs for l_upwind_xpyp_ta=.true. and stats%l_sample=.true. (ARM default).
                _a1_zm_st = 1.0 / (1.0 - _asarray(sigma_sqd_w, dtype=np.float64))
                _a1_zt_st = _asarray(zm2zt(jnp.asarray(_a1_zm_st), gr))
                _beta_st = _asarray(clubb_params[:, ibeta - 1], dtype=np.float64)
                _wp_coef_zt_st = ((1.0 - (1.0/3.0) * _beta_st[:, np.newaxis])
                                  * _a1_zt_st**2
                                  * _asarray(wp3_on_wp2_zt, dtype=np.float64)
                                  / _asarray(wp2_zt, dtype=np.float64))
                _wpthlp_zt_st = _asarray(zm2zt(jnp.asarray(wpthlp), gr))
                _wprtp_zt_st = _asarray(zm2zt(jnp.asarray(wprtp), gr))
                stats_writer.update("term_wprtp2_explicit",
                    _wp_coef_zt_st * _wprtp_zt_st**2)
                stats_writer.update("term_wpthlp2_explicit",
                    _wp_coef_zt_st * _wpthlp_zt_st**2)
                stats_writer.update("term_wprtpthlp_explicit",
                    _wp_coef_zt_st * _wprtp_zt_st * _wpthlp_zt_st)

                def _mm3(lhs3, x):
                    """lhs3 @ x at interior zm levels: lhs3[0]*x[k+1]+lhs3[1]*x[k]+lhs3[2]*x[k-1]"""
                    r = _xp.zeros_like(x)
                    r = _iset(r, np.s_[:, 1:-1], (lhs3[0, :, 1:-1] * x[:, 2:]
                                + lhs3[1, :, 1:-1] * x[:, 1:-1]
                                + lhs3[2, :, 1:-1] * x[:, :-2]))
                    return r

                # rtp2 budget terms
                _rt_thr10 = float(rt_tol ** 2)
                _rtp2_mix10 = _omg10 * _rtp2_np + _g10 * _soln10_rtp2
                _rtp2_tp10 = _xp.zeros_like(_rtp2_np)
                _rtp2_tp10 = _iset(_rtp2_tp10, np.s_[:, 1:-1], (
                    -_wprtp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_rtm_np[:, 1:] - _rtm_np[:, :-1])
                    - _wprtp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_rtm_np[:, 1:] - _rtm_np[:, :-1])
                ))
                stats_writer.update("rtp2_ta",
                    _rhs_ta10_rtp2 - _mm3(_lhs_ta_10, _rtp2_mix10))
                stats_writer.update("rtp2_dp1",
                    _dp1_ref * (_rt_thr10 - _rtp2_mix10))
                stats_writer.update("rtp2_dp2",
                    -_mm3(_lhs_diff_f, _soln10_rtp2))
                stats_writer.update("rtp2_tp", _rtp2_tp10)
                _rtp2_pd10 = _xp.zeros_like(_rtp2_np)
                _rtp2_pd10 = _iset(_rtp2_pd10, np.s_[:, 1:-1], (
                    (_rtp2_jax10_fh[:, 1:-1] - _soln10_rtp2[:, 1:-1]) / _dt10))
                stats_writer.update("rtp2_pd", _rtp2_pd10)
                stats_writer.update("rtp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(_rtp2_jax10_cv), gr),
                        float(rt_tol ** 2)), dtype=np.float64))

                # thlp2 budget terms
                _thl_thr10 = _threshold_thlp2
                _thlp2_mix10 = _omg10 * _thlp2_np + _g10 * _soln10_thlp2
                _thlp2_tp10 = _xp.zeros_like(_thlp2_np)
                _thlp2_tp10 = _iset(_thlp2_tp10, np.s_[:, 1:-1], (
                    -_wpthlp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_thlm_np[:, 1:] - _thlm_np[:, :-1])
                    - _wpthlp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_thlm_np[:, 1:] - _thlm_np[:, :-1])
                ))
                stats_writer.update("thlp2_ta",
                    _rhs_ta10_thlp2 - _mm3(_lhs_ta_10, _thlp2_mix10))
                stats_writer.update("thlp2_dp1",
                    _dp1_ref * (_thl_thr10 - _thlp2_mix10))
                stats_writer.update("thlp2_dp2",
                    -_mm3(_lhs_diff_f, _soln10_thlp2))
                stats_writer.update("thlp2_tp", _thlp2_tp10)
                _thlp2_pd10 = _xp.zeros_like(_thlp2_np)
                _thlp2_pd10 = _iset(_thlp2_pd10, np.s_[:, 1:-1], (
                    (_thlp2_jax10_fh[:, 1:-1] - _soln10_thlp2[:, 1:-1]) / _dt10))
                stats_writer.update("thlp2_pd", _thlp2_pd10)
                stats_writer.update("thlp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(_thlp2_jax10_cv), gr),
                        float(thl_tol ** 2)), dtype=np.float64))

                # rtpthlp budget terms
                _rtpthlp_mix10 = _omg10 * _rtpthlp_np + _g10 * _soln10_rtpthlp
                _rtpthlp_tp1_10 = _xp.zeros_like(_rtpthlp_np)
                _rtpthlp_tp1_10 = _iset(_rtpthlp_tp1_10, np.s_[:, 1:-1], (
                    -_wprtp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_thlm_np[:, 1:] - _thlm_np[:, :-1])
                ))
                _rtpthlp_tp2_10 = _xp.zeros_like(_rtpthlp_np)
                _rtpthlp_tp2_10 = _iset(_rtpthlp_tp2_10, np.s_[:, 1:-1], (
                    -_wpthlp_np[:, 1:-1] * _invrs_dzm_np[:, 1:-1]
                    * (_rtm_np[:, 1:] - _rtm_np[:, :-1])
                ))
                stats_writer.update("rtpthlp_ta",
                    _rhs_ta10_rtpthlp - _mm3(_lhs_ta_10, _rtpthlp_mix10))
                stats_writer.update("rtpthlp_dp1",
                    _dp1_ref * (_threshold_rtpthlp - _rtpthlp_mix10))
                stats_writer.update("rtpthlp_dp2",
                    -_mm3(_lhs_diff_f, _soln10_rtpthlp))
                stats_writer.update("rtpthlp_tp1", _rtpthlp_tp1_10)
                stats_writer.update("rtpthlp_tp2", _rtpthlp_tp2_10)
                # rtpthlp_cl: Cauchy-Schwarz clip_covar effect
                stats_writer.update("rtpthlp_cl",
                    (_rtpthlp_jax10_clip - _soln10_rtpthlp) / _dt10)

            # ============================================================ #
            # Iter36: JAX shadow for up2/vp2 (advance_xp2_xpyp).           #
            # LHS: same TA as rtp2 (ADG1); Kw9 diffusion; C4/C14 dp1.      #
            # ============================================================ #
            # LHS diffusion: Kw9 = c_K9 * Kh_zt
            _c_K9_36 = _asarray(clubb_params[:, ic_K9 - 1], dtype=np.float64)
            _Kw9_zt_36 = _c_K9_36[:, np.newaxis] * _asarray(Kh_zt, dtype=np.float64)
            _nu9_36 = _asarray(nu_vert_res_dep.nu9, dtype=np.float64)
            _lhs_diff_uv36 = _asarray(diffusion_zm_lhs_jax(
                jnp.asarray(_Kw9_zt_36),
                jnp.asarray(_nu9_36),
                jnp.asarray(_asarray(invrs_rho_ds_zm, dtype=np.float64)),
                jnp.asarray(_asarray(rho_ds_zt, dtype=np.float64)), gr,
            ))
            # LHS dp1: (2/3*C4*invrs_tau_C4 + 1/3*C14*invrs_tau_C14) * gamma
            _C4_36 = _asarray(clubb_params[:, iC4 - 1], dtype=np.float64)[:, np.newaxis]
            _C14_36 = _asarray(clubb_params[:, iC14 - 1], dtype=np.float64)[:, np.newaxis]
            _c4_1d_36 = (2.0/3.0) * _C4_36 * np.ones((ngrdcol, nzm), dtype=np.float64)
            _c14_1d_36 = (1.0/3.0) * _C14_36 * np.ones((ngrdcol, nzm), dtype=np.float64)
            _invrs_tau_C4_zm_36 = _asarray(invrs_tau_C4_zm, dtype=np.float64)
            _invrs_tau_C14_zm_36 = _asarray(invrs_tau_C14_zm, dtype=np.float64)
            _lhs_dp1_C4_36 = _asarray(term_dp1_lhs_jax(
                jnp.asarray(_c4_1d_36), jnp.asarray(_invrs_tau_C4_zm_36)
            ))
            _lhs_dp1_C14_36 = _asarray(term_dp1_lhs_jax(
                jnp.asarray(_c14_1d_36), jnp.asarray(_invrs_tau_C14_zm_36)
            ))
            _lhs_dp1_uv36 = (_lhs_dp1_C4_36 + _lhs_dp1_C14_36) * _gamma
            # LHS assembly (TA same as rtp2, different diff/dp1)
            _lhs_uv36 = _asarray(xp2_xpyp_lhs_jax(
                jnp.asarray(_lhs_ta_10),
                jnp.asarray(_lhs_ma_f),
                jnp.asarray(_lhs_diff_uv36),
                jnp.asarray(_lhs_dp1_uv36),
                _dt_adv,
            ))
            # RHS TA for up2/vp2 (ADG1 path — same coefficient as rtp2 but uses upwp/vpwp)
            _upwp_np36 = _asarray(upwp, dtype=np.float64)
            _vpwp_np36 = _asarray(vpwp, dtype=np.float64)
            if flags.l_upwind_xpyp_ta:
                # Iter58: JAX upwind RHS for up2/vp2 (Fortran oracle removed)
                _rho_ds_zm_j36 = jnp.asarray(rho_ds_zm)
                _irho_ds_zm_j36 = jnp.asarray(invrs_rho_ds_zm)
                _sgn10_j36 = jnp.asarray(_sgn10)
                _gdir36 = float(gr.grid_dir)
                _rhs_ta36_up2 = _asarray(xpyp_term_ta_pdf_rhs_upwind_jax(
                    _rho_ds_zm_j36, _irho_ds_zm_j36, _sgn10_j36,
                    jnp.asarray(_wp_coef * _upwp_np36**2),
                    gr, grid_dir=_gdir36,
                ), dtype=np.float64)
                _rhs_ta36_vp2 = _asarray(xpyp_term_ta_pdf_rhs_upwind_jax(
                    _rho_ds_zm_j36, _irho_ds_zm_j36, _sgn10_j36,
                    jnp.asarray(_wp_coef * _vpwp_np36**2),
                    gr, grid_dir=_gdir36,
                ), dtype=np.float64)
            else:
                # Iter51: Fortran xpyp_term_ta_pdf_rhs oracle removed; use JAX.
                _upwp_zt_36 = _asarray(zm2zt_jax(jnp.asarray(_upwp_np36), gr))
                _vpwp_zt_36 = _asarray(zm2zt_jax(jnp.asarray(_vpwp_np36), gr))
                _rhs_ta36_up2 = _asarray(xpyp_term_ta_pdf_rhs_jax(
                    jnp.asarray(_wp_coef_zt_10 * _upwp_zt_36**2),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
                _rhs_ta36_vp2 = _asarray(xpyp_term_ta_pdf_rhs_jax(
                    jnp.asarray(_wp_coef_zt_10 * _vpwp_zt_36**2),
                    jnp.asarray(rho_ds_zt),
                    jnp.asarray(invrs_rho_ds_zm),
                    gr,
                ))
            # Auxiliary inputs for pressure and production terms
            _C_uu_shr36 = _asarray(clubb_params[:, iC_uu_shr - 1], dtype=np.float64)[:, np.newaxis]
            _C_uu_buoy36 = _asarray(clubb_params[:, iC_uu_buoy - 1], dtype=np.float64)[:, np.newaxis]
            _um_np36 = _asarray(um, dtype=np.float64)
            _vm_np36 = _asarray(vm, dtype=np.float64)
            _wp2_np36 = _asarray(wp2, dtype=np.float64)
            _wpthvp_np36 = _asarray(wpthvp, dtype=np.float64)
            _thv_ds_zm_np36 = _asarray(thv_ds_zm, dtype=np.float64)
            _lhs_splat_36 = _asarray(lhs_splat_wp2, dtype=np.float64)
            _omg36 = 1.0 - _gamma
            # Wind gradients at interior zm levels: d(um)/dz = (um[k] - um[k-1]) * invrs_dzm[k]
            _du_dz_36 = _invrs_dzm_np[:, 1:-1] * (_um_np36[:, 1:] - _um_np36[:, :-1])
            _dv_dz_36 = _invrs_dzm_np[:, 1:-1] * (_vm_np36[:, 1:] - _vm_np36[:, :-1])
            # PR2 (same for up2 and vp2): max((2/3)*(C_uu_buoy*g/thv_ds*wpthvp + C_uu_shr*shear), 0)
            _pr2_36 = (2.0/3.0) * (
                _C_uu_buoy36 * (float(grav) / _thv_ds_zm_np36[:, 1:-1]) * _wpthvp_np36[:, 1:-1]
                + _C_uu_shr36 * (
                    -_upwp_np36[:, 1:-1] * _du_dz_36
                    - _vpwp_np36[:, 1:-1] * _dv_dz_36
                )
            )
            _pr2_36 = _xp.maximum(_pr2_36, float(zero_threshold))
            # RHS for up2
            _rhs36_up2 = np.zeros((ngrdcol, nzm), dtype=np.float64)
            _rhs36_up2 = _iset(_rhs36_up2, np.s_[:, 1:-1], (
                _rhs_ta36_up2[:, 1:-1]
                + 0.5 * _lhs_splat_36[:, 1:-1] * _wp2_np36[:, 1:-1]
                + _omg36 * (
                    -_lhs_ta_10[0, :, 1:-1] * _up2_pre36[:, 2:]
                    - _lhs_ta_10[1, :, 1:-1] * _up2_pre36[:, 1:-1]
                    - _lhs_ta_10[2, :, 1:-1] * _up2_pre36[:, :-2]
                )
                + (1.0 - _C_uu_shr36) * (
                    -_upwp_np36[:, 1:-1] * _du_dz_36
                    - _upwp_np36[:, 1:-1] * _du_dz_36
                )
                + (
                    (1.0/3.0) * _C4_36 * (_vp2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1]) * _invrs_tau_C4_zm_36[:, 1:-1]
                    - (1.0/3.0) * _C14_36 * (_vp2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1]) * _invrs_tau_C14_zm_36[:, 1:-1]
                    + _C14_36 * _invrs_tau_C14_zm_36[:, 1:-1] * float(w_tol_sqd)
                )
                + _omg36 * (-_lhs_dp1_C4_36[:, 1:-1] - _lhs_dp1_C14_36[:, 1:-1]) * _up2_pre36[:, 1:-1]
                + _pr2_36
                + (1.0 / _dt_adv) * _up2_pre36[:, 1:-1]
            ))
            # Nontraditional Coriolis term for up2 (advance_xp2_xpyp_module.F90:772).
            if getattr(flags, 'l_ho_nontrad_coriolis', False):
                _fcy36 = _asarray(fcor_y, dtype=np.float64)[:, np.newaxis]
                _rhs36_up2 = _iset(_rhs36_up2, np.s_[:, 1:-1], (_rhs36_up2[:, 1:-1]
                                       - 2.0 * _fcy36 * _upwp_np36[:, 1:-1]))
            _rhs36_up2 = _iset(_rhs36_up2, np.s_[:, 0], _up2_pre36[:, 0])
            _rhs36_up2 = _iset(_rhs36_up2, np.s_[:, -1], float(w_tol_sqd))
            # RHS for vp2 (swap up2↔vp2, u↔v)
            _rhs36_vp2 = np.zeros((ngrdcol, nzm), dtype=np.float64)
            _rhs36_vp2 = _iset(_rhs36_vp2, np.s_[:, 1:-1], (
                _rhs_ta36_vp2[:, 1:-1]
                + 0.5 * _lhs_splat_36[:, 1:-1] * _wp2_np36[:, 1:-1]
                + _omg36 * (
                    -_lhs_ta_10[0, :, 1:-1] * _vp2_pre36[:, 2:]
                    - _lhs_ta_10[1, :, 1:-1] * _vp2_pre36[:, 1:-1]
                    - _lhs_ta_10[2, :, 1:-1] * _vp2_pre36[:, :-2]
                )
                + (1.0 - _C_uu_shr36) * (
                    -_vpwp_np36[:, 1:-1] * _dv_dz_36
                    - _vpwp_np36[:, 1:-1] * _dv_dz_36
                )
                + (
                    (1.0/3.0) * _C4_36 * (_up2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1]) * _invrs_tau_C4_zm_36[:, 1:-1]
                    - (1.0/3.0) * _C14_36 * (_up2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1]) * _invrs_tau_C14_zm_36[:, 1:-1]
                    + _C14_36 * _invrs_tau_C14_zm_36[:, 1:-1] * float(w_tol_sqd)
                )
                + _omg36 * (-_lhs_dp1_C4_36[:, 1:-1] - _lhs_dp1_C14_36[:, 1:-1]) * _vp2_pre36[:, 1:-1]
                + _pr2_36
                + (1.0 / _dt_adv) * _vp2_pre36[:, 1:-1]
            ))
            _rhs36_vp2 = _iset(_rhs36_vp2, np.s_[:, 0], _vp2_pre36[:, 0])
            _rhs36_vp2 = _iset(_rhs36_vp2, np.s_[:, -1], float(w_tol_sqd))
            # Solve both with same LHS (ADG1)
            _lhs_uv36_jax = jnp.asarray(_lhs_uv36)
            _soln36_up2 = _asarray(tridiag_lu_solve_jax(_lhs_uv36_jax, jnp.asarray(_rhs36_up2)))
            _soln36_vp2 = _asarray(tridiag_lu_solve_jax(_lhs_uv36_jax, jnp.asarray(_rhs36_vp2)))
            if flags.l_lmm_stepping:
                _up2_jax36 = 0.5 * (_up2_pre36 + _soln36_up2)
                _vp2_jax36 = 0.5 * (_vp2_pre36 + _soln36_vp2)
            else:
                _up2_jax36 = _soln36_up2
                _vp2_jax36 = _soln36_vp2
            # fill_holes (pos_definite_variances) with threshold=w_tol_sqd (Iter41: JAX)
            _up2_jax36_fh = _asarray(fill_holes_vertical_jax(
                field=jnp.asarray(_up2_jax36),
                rho_ds=jnp.asarray(rho_ds_zm),
                dz=jnp.asarray(gr.dzm),
                threshold=float(w_tol_sqd),
                lower_k=_hf_lower, upper_k=_hf_upper,
                fill_holes_type=flags.fill_holes_type,
            ))
            _vp2_jax36_fh = _asarray(fill_holes_vertical_jax(
                field=jnp.asarray(_vp2_jax36),
                rho_ds=jnp.asarray(rho_ds_zm),
                dz=jnp.asarray(gr.dzm),
                threshold=float(w_tol_sqd),
                lower_k=_hf_lower, upper_k=_hf_upper,
                fill_holes_type=flags.fill_holes_type,
            ))
            # clip_variance: floor to w_tol_sqd
            _up2_jax36_cv = _up2_jax36_fh.copy()
            _vp2_jax36_cv = _vp2_jax36_fh.copy()
            _up2_jax36_cv = _iset(_up2_jax36_cv, np.s_[:, :-1], _xp.maximum(_up2_jax36_fh[:, :-1], float(w_tol_sqd)))
            _vp2_jax36_cv = _iset(_vp2_jax36_cv, np.s_[:, :-1], _xp.maximum(_vp2_jax36_fh[:, :-1], float(w_tol_sqd)))
            # Iter47: Fortran oracle removed; JAX results are the state.

            # ============================================================ #
            # Budget stats: up2/vp2 + pressure rotation (upwp_pr4/vpwp_pr4) #
            # ============================================================ #
            if l_sample and stats_writer is not None:
                _dt36 = float(dt_advance)
                _g36 = _gamma   # 1.5
                _omg36b = 1.0 - _g36  # -0.5

                # up2/vp2 mixed values (pre-solve and post-solve)
                _up2_mix36 = _omg36b * _up2_pre36 + _g36 * _soln36_up2
                _vp2_mix36 = _omg36b * _vp2_pre36 + _g36 * _soln36_vp2

                # TA terms: rhs_ta - lhs_ta @ mixed (shared LHS with rtp2)
                _up2_ta36 = _rhs_ta36_up2 - _mm3(_lhs_ta_10, _up2_mix36)
                _vp2_ta36 = _rhs_ta36_vp2 - _mm3(_lhs_ta_10, _vp2_mix36)
                stats_writer.update("up2_ta", _up2_ta36)
                stats_writer.update("vp2_ta", _vp2_ta36)

                # TP: (1-C_uu_shr) * (-2*upwp*invrs_dzm*(um[k]-um[k-1]))
                _up2_tp36 = _xp.zeros_like(_up2_pre36)
                _vp2_tp36 = _xp.zeros_like(_vp2_pre36)
                _up2_tp36 = _iset(_up2_tp36, np.s_[:, 1:-1], (
                    (1.0 - _C_uu_shr36[:, 0:1]) * (
                        -_upwp_np36[:, 1:-1] * _du_dz_36
                        - _upwp_np36[:, 1:-1] * _du_dz_36
                    )
                ))
                _vp2_tp36 = _iset(_vp2_tp36, np.s_[:, 1:-1], (
                    (1.0 - _C_uu_shr36[:, 0:1]) * (
                        -_vpwp_np36[:, 1:-1] * _dv_dz_36
                        - _vpwp_np36[:, 1:-1] * _dv_dz_36
                    )
                ))
                stats_writer.update("up2_tp", _up2_tp36)
                stats_writer.update("vp2_tp", _vp2_tp36)

                # PR2: max((2/3)*(C_uu_buoy*g/thv*wpthvp + C_uu_shr*(-upwp*du_dz - vpwp*dv_dz)), 0)
                _pr2_full36 = _xp.zeros_like(_up2_pre36)
                _pr2_full36 = _iset(_pr2_full36, np.s_[:, 1:-1], _pr2_36)
                stats_writer.update("up2_pr2", _pr2_full36)
                stats_writer.update("vp2_pr2", _pr2_full36)

                # PR1 stats: rhs_pr1_C4 - lhs_dp1_C4 * mixed
                # stats_pr1 (C4 only) = (1/3)*C4*(vp2_old+wp2)*invrs_tau_C4
                # stats_pr2_C14 (C14 only) = -(1/3)*C14*(vp2_old+wp2)*invrs_tau_C14 + C14*invrs_tau_C14*w_tol_sqd
                _rhs_pr1_C4_up = _xp.zeros_like(_up2_pre36)
                _rhs_pr1_C14_up = _xp.zeros_like(_up2_pre36)
                _rhs_pr1_C4_vp = _xp.zeros_like(_vp2_pre36)
                _rhs_pr1_C14_vp = _xp.zeros_like(_vp2_pre36)
                _rhs_pr1_C4_up = _iset(_rhs_pr1_C4_up, np.s_[:, 1:-1], (
                    (1.0/3.0) * _C4_36[:, 0:1]
                    * (_vp2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1])
                    * _invrs_tau_C4_zm_36[:, 1:-1]
                ))
                _rhs_pr1_C14_up = _iset(_rhs_pr1_C14_up, np.s_[:, 1:-1], (
                    -(1.0/3.0) * _C14_36[:, 0:1]
                    * (_vp2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1])
                    * _invrs_tau_C14_zm_36[:, 1:-1]
                    + _C14_36[:, 0:1] * _invrs_tau_C14_zm_36[:, 1:-1] * float(w_tol_sqd)
                ))
                _rhs_pr1_C4_vp = _iset(_rhs_pr1_C4_vp, np.s_[:, 1:-1], (
                    (1.0/3.0) * _C4_36[:, 0:1]
                    * (_up2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1])
                    * _invrs_tau_C4_zm_36[:, 1:-1]
                ))
                _rhs_pr1_C14_vp = _iset(_rhs_pr1_C14_vp, np.s_[:, 1:-1], (
                    -(1.0/3.0) * _C14_36[:, 0:1]
                    * (_up2_pre36[:, 1:-1] + _wp2_np36[:, 1:-1])
                    * _invrs_tau_C14_zm_36[:, 1:-1]
                    + _C14_36[:, 0:1] * _invrs_tau_C14_zm_36[:, 1:-1] * float(w_tol_sqd)
                ))
                # _lhs_dp1_C4_36 = (2/3)*C4*invrs_tau_C4 (already unscaled, no gamma factor)
                # _lhs_dp1_C14_36 = (1/3)*C14*invrs_tau_C14 (already unscaled)
                _up2_pr1_36 = _rhs_pr1_C4_up - _lhs_dp1_C4_36 * _up2_mix36
                _up2_dp1_36 = _rhs_pr1_C14_up - _lhs_dp1_C14_36 * _up2_mix36
                _vp2_pr1_36 = _rhs_pr1_C4_vp - _lhs_dp1_C4_36 * _vp2_mix36
                _vp2_dp1_36 = _rhs_pr1_C14_vp - _lhs_dp1_C14_36 * _vp2_mix36
                stats_writer.update("up2_pr1", _up2_pr1_36)
                stats_writer.update("up2_dp1", _up2_dp1_36)
                stats_writer.update("vp2_pr1", _vp2_pr1_36)
                stats_writer.update("vp2_dp1", _vp2_dp1_36)

                # DP2 (diffusion): -lhs_diff_uv36 @ new
                stats_writer.update("up2_dp2", -_mm3(_lhs_diff_uv36, _soln36_up2))
                stats_writer.update("vp2_dp2", -_mm3(_lhs_diff_uv36, _soln36_vp2))

                # PD (fill_holes effect): (fh - solve) / dt
                _up2_pd36 = _xp.zeros_like(_up2_pre36)
                _vp2_pd36 = _xp.zeros_like(_vp2_pre36)
                _up2_pd36 = _iset(_up2_pd36, np.s_[:, 1:-1], (
                    (_up2_jax36_fh[:, 1:-1] - _soln36_up2[:, 1:-1]) / _dt36))
                _vp2_pd36 = _iset(_vp2_pd36, np.s_[:, 1:-1], (
                    (_vp2_jax36_fh[:, 1:-1] - _soln36_vp2[:, 1:-1]) / _dt36))
                stats_writer.update("up2_pd", _up2_pd36)
                stats_writer.update("vp2_pd", _vp2_pd36)

                # ZT: post-clip value interpolated to zt
                stats_writer.update("up2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(_up2_jax36_cv), gr),
                        float(w_tol_sqd)), dtype=np.float64))
                stats_writer.update("vp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(_vp2_jax36_cv), gr),
                        float(w_tol_sqd)), dtype=np.float64))

                # upwp_pr4/vpwp_pr4 are written in the iter69 stats block (advance_xm_wpxp)
                # using ddzt_um/ddzt_vm (zt-level gradient), not du_dz_zm here.

            # ============================================================ #
            # Iter35+36: Override advance_xp2_xpyp state with JAX values    #
            # rtp2/thlp2/rtpthlp verified at machine epsilon (iter10).      #
            # up2/vp2 verified at machine epsilon (iter36).                 #
            # ============================================================ #
            rtp2    = _asarray(_rtp2_jax10_cv,      dtype=np.float64).copy()
            thlp2   = _asarray(_thlp2_jax10_cv,     dtype=np.float64).copy()
            rtpthlp = _asarray(_rtpthlp_jax10_clip, dtype=np.float64).copy()
            up2     = _asarray(_up2_jax36_cv,        dtype=np.float64).copy()
            vp2     = _asarray(_vp2_jax36_cv,        dtype=np.float64).copy()

            # Iter45: JAX clip_covars_denom directly (Fortran oracle removed; verified 0.000e+00)
            # ARM: l_linearize_pbl_winds=False → upwp_pert/vpwp_pert unchanged
            # ARM: sclr_dim=0 → wpsclrp unchanged
            (_wprtp_j24a, _wpthlp_j24a, _upwp_j24a, _vpwp_j24a) = clip_covars_denom_jax(
                wprtp=jnp.asarray(_asarray(wprtp, dtype=np.float64)),
                wpthlp=jnp.asarray(_asarray(wpthlp, dtype=np.float64)),
                upwp=jnp.asarray(_asarray(upwp, dtype=np.float64)),
                vpwp=jnp.asarray(_asarray(vpwp, dtype=np.float64)),
                wp2=jnp.asarray(_asarray(wp2, dtype=np.float64)),
                rtp2=jnp.asarray(_asarray(rtp2, dtype=np.float64)),
                thlp2=jnp.asarray(_asarray(thlp2, dtype=np.float64)),
                up2=jnp.asarray(_asarray(up2, dtype=np.float64)),
                vp2=jnp.asarray(_asarray(vp2, dtype=np.float64)),
                l_tke_aniso=flags.l_tke_aniso,
            )
            wprtp  = _asarray(_wprtp_j24a,  dtype=np.float64).copy()
            wpthlp = _asarray(_wpthlp_j24a, dtype=np.float64).copy()
            upwp   = _asarray(_upwp_j24a,   dtype=np.float64).copy()
            vpwp   = _asarray(_vpwp_j24a,   dtype=np.float64).copy()
            err_code = err_info.err_code
            if err_code is not None and np.any(_asarray(err_code) == CLUBB_FATAL_ERROR):
                return

        elif advance_iter == order_wp2_wp3_val:
            # ---- Save pre-call state for iter12 shadow comparison ----
            _wp2_pre12 = _asarray(wp2).copy()
            _wp3_pre12 = _asarray(wp3).copy()
            _up2_pre12 = _asarray(up2).copy()
            _vp2_pre12 = _asarray(vp2).copy()

            # Iter48: Fortran advance_wp2_wp3 oracle removed.

            # ============================================================ #
            # Block W: Iteration 12 shadow comparison                       #
            # advance_wp2_wp3 JAX vs Fortran (ARM: ADG1, l_damp_wp2_em,   #
            # l_damp_wp3_Skw_squared, l_tke_aniso, l_wp2_fill_holes_tke)  #
            # ============================================================ #
            _nu1_12 = float(_asarray(nu_vert_res_dep.nu1, dtype=np.float64).flat[0])
            _nu8_12 = float(_asarray(nu_vert_res_dep.nu8, dtype=np.float64).flat[0])

            (_wp2_jax12_raw, _wp3_jax12_raw,
             _C1_Skw_fnc_12, _C11_Skw_fnc_12, _sd12) = advance_wp2_wp3_jax(
                wp2=jnp.array(_wp2_pre12),
                wp3=jnp.array(_wp3_pre12),
                up2=jnp.array(_up2_pre12),
                vp2=jnp.array(_vp2_pre12),
                sigma_sqd_w=jnp.array(sigma_sqd_w),
                wp3_on_wp2=jnp.array(wp3_on_wp2),
                em=jnp.array(em),
                wpup2=jnp.array(wpup2),
                wpvp2=jnp.array(wpvp2),
                wp2up2=jnp.array(wp2up2),
                wp2vp2=jnp.array(wp2vp2),
                wp4=jnp.array(wp4),
                wpthvp=jnp.array(wpthvp),
                wp2thvp=jnp.array(wp2thvp),
                um=jnp.array(um),
                vm=jnp.array(vm),
                upwp=jnp.array(upwp),
                vpwp=jnp.array(vpwp),
                wm_zm=jnp.array(wm_zm),
                wm_zt=jnp.array(wm_zt),
                Kh_zm=jnp.array(Kh_zm),
                Kh_zt=jnp.array(Kh_zt),
                invrs_tau_C4_zm=jnp.array(invrs_tau_C4_zm),
                invrs_tau_wp3_zt=jnp.array(invrs_tau_wp3_zt),
                invrs_tau_C1_zm=jnp.array(invrs_tau_C1_zm),
                Skw_zm=jnp.array(Skw_zm),
                Skw_zt=jnp.array(Skw_zt),
                rho_ds_zm=jnp.array(rho_ds_zm),
                rho_ds_zt=jnp.array(rho_ds_zt),
                invrs_rho_ds_zm=jnp.array(invrs_rho_ds_zm),
                invrs_rho_ds_zt=jnp.array(invrs_rho_ds_zt),
                thv_ds_zm=jnp.array(thv_ds_zm),
                thv_ds_zt=jnp.array(thv_ds_zt),
                lhs_splat_wp2=jnp.array(lhs_splat_wp2),
                lhs_splat_wp3=jnp.array(lhs_splat_wp3),
                wprtp=jnp.array(wprtp),
                wpthlp=jnp.array(wpthlp),
                rtp2=jnp.array(rtp2),
                thlp2=jnp.array(thlp2),
                clubb_params=jnp.array(clubb_params),
                dt=float(dt_advance),
                nu1=_nu1_12,
                nu8=_nu8_12,
                gr=gr,
                l_ho_nontrad_coriolis=getattr(flags, 'l_ho_nontrad_coriolis', False),
                fcor_y=fcor_y,
                wp2up=wp2up,
            )

            # Apply post-solve steps to JAX result, matching Fortran wp23_solve
            _wp2_jax12 = _asarray(_wp2_jax12_raw, dtype=np.float64).copy()
            _wp3_jax12 = _asarray(_wp3_jax12_raw, dtype=np.float64).copy()

            # fill_holes_vertical on wp2 — Iter53: Fortran oracle removed
            _hf_lower12 = gr.k_lb_zm + gr.grid_dir_indx   # Python 0-based
            _hf_upper12 = gr.k_ub_zm - gr.grid_dir_indx   # Python 0-based
            _wp2_jax12 = _asarray(fill_holes_vertical_jax(
                field=jnp.asarray(_wp2_jax12),
                rho_ds=jnp.asarray(rho_ds_zm),
                dz=jnp.asarray(gr.dzm),
                threshold=float(w_tol_sqd),
                lower_k=_hf_lower12, upper_k=_hf_upper12,
                fill_holes_type=flags.fill_holes_type,
            ), dtype=np.float64)

            # fill_holes_wp2_from_horz_tke (ARM: l_wp2_fill_holes_tke=True) — Iter53: Fortran oracle removed
            _up2_out12 = _up2_pre12
            _vp2_out12 = _vp2_pre12
            if flags.l_wp2_fill_holes_tke:
                # Fortran lower_hf_level=1 (1-based) → Python lower_k=0
                # Fortran upper_hf_level=nzm-2 (1-based) → Python upper_k=nzm-3
                (_wp2_jax12, _up2_out12, _vp2_out12) = [
                    _asarray(x, dtype=np.float64)
                    for x in fill_holes_wp2_from_horz_tke_jax(
                        wp2=jnp.asarray(_wp2_jax12),
                        up2=jnp.asarray(_asarray(_up2_pre12, dtype=np.float64)),
                        vp2=jnp.asarray(_asarray(_vp2_pre12, dtype=np.float64)),
                        threshold=float(w_tol_sqd),
                        lower_k=0,
                        upper_k=nzm - 3,
                    )
                ]

            # clip_variance on wp2 (solve_type=12, wp2_min=w_tol_sqd for l_min_wp2_from_corr_wx=False)
            _wp2_min12 = _xp.full_like(_wp2_jax12, float(w_tol_sqd))
            if flags.l_min_wp2_from_corr_wx:
                _corr_max2 = 0.99 ** 2
                _wprtp_np12 = _asarray(wprtp, dtype=np.float64)
                _wpthlp_np12 = _asarray(wpthlp, dtype=np.float64)
                _upwp_np12 = _asarray(upwp, dtype=np.float64)
                _vpwp_np12 = _asarray(vpwp, dtype=np.float64)
                _rtp2_np12 = _asarray(rtp2, dtype=np.float64)
                _thlp2_np12 = _asarray(thlp2, dtype=np.float64)
                _up2_np12 = _asarray(up2, dtype=np.float64)
                _vp2_np12 = _asarray(vp2, dtype=np.float64)
                _wp2_min12 = _xp.minimum(1.0, _xp.maximum(_wp2_min12,
                    _wprtp_np12 ** 2 / (_rtp2_np12 * _corr_max2),
                    _wpthlp_np12 ** 2 / (_thlp2_np12 * _corr_max2),
                    _upwp_np12 ** 2 / (_up2_np12 * _corr_max2),
                    _vpwp_np12 ** 2 / (_vp2_np12 * _corr_max2),
                ))
            # clip_variance on wp2 — Iter53: Fortran oracle removed
            _wp2_jax12 = _asarray(clip_variance_jax(
                xp2=jnp.asarray(_wp2_jax12),
                threshold_lo=jnp.asarray(_wp2_min12),
            ), dtype=np.float64)

            # zm2zt to get wp2_zt
            _wp2_zt_jax12 = _asarray(zm2zt(_wp2_jax12, gr), dtype=np.float64)
            _wp2_zt_jax12 = _xp.maximum(_wp2_zt_jax12, w_tol_sqd)   # positive definite

            # clip_skewness on wp3 — Iter53: Fortran oracle removed
            _skw_max12 = clubb_params[:, iSkw_max_mag - 1]
            _wp3_jax12 = _asarray(clip_skewness_jax(
                wp3=jnp.asarray(_wp3_jax12),
                wp2_zt=jnp.asarray(_wp2_zt_jax12),
                zt=jnp.asarray(gr.zt),
                sfc_elevation=jnp.asarray(sfc_elevation),
                Skw_max_mag=jnp.asarray(_skw_max12),
                l_use_wp3_lim_with_smth_Heaviside=flags.l_use_wp3_lim_with_smth_Heaviside,
            ), dtype=np.float64)

            # Iter48: Fortran oracle removed; JAX results are the state.

            # Iter68: advance_wp2_wp3_module.F90 stats writes (C1/C11_Skw_fnc, budgets)
            if l_sample and stats_writer is not None:
                stats_writer.update("C1_Skw_fnc",  _asarray(_C1_Skw_fnc_12,  dtype=np.float64))
                stats_writer.update("C11_Skw_fnc", _asarray(_C11_Skw_fnc_12, dtype=np.float64))

                # ---- post-advance wp2_zt and wp3_zm (fix: written here, not earlier) ----
                _wp3_zm_jax12 = _asarray(zt2zm(jnp.asarray(_wp3_jax12), gr), dtype=np.float64)
                stats_writer.update("wp2_zt", _wp2_zt_jax12)
                stats_writer.update("wp3_zm", _wp3_zm_jax12)

                # ---- wp2/wp3 clipping budget stats ----
                _dt12 = float(dt_advance)
                stats_writer.update("wp3_cl",
                    (_asarray(_wp3_jax12, dtype=np.float64) - _asarray(_wp3_jax12_raw, dtype=np.float64)) / _dt12)
                stats_writer.update("wp2_pd",
                    (_asarray(_wp2_jax12, dtype=np.float64) - _asarray(_wp2_jax12_raw, dtype=np.float64)) / _dt12)
                _up2_out12_np = _asarray(_up2_out12 if flags.l_wp2_fill_holes_tke else _up2_pre12, dtype=np.float64)
                _vp2_out12_np = _asarray(_vp2_out12 if flags.l_wp2_fill_holes_tke else _vp2_pre12, dtype=np.float64)
                # Mirror Fortran: up2_pd/vp2_pd contribution from wp2 block uses
                # l_count_sample=.false. so nsamples is NOT incremented here.
                _up2_pre12_np = _asarray(_up2_pre12, dtype=np.float64)
                _vp2_pre12_np = _asarray(_vp2_pre12, dtype=np.float64)
                stats_writer.begin_budget("up2_pd", _up2_pre12_np / _dt12)
                stats_writer.begin_budget("vp2_pd", _vp2_pre12_np / _dt12)
                stats_writer.finalize_budget("up2_pd", _up2_out12_np / _dt12, l_count_sample=False)
                stats_writer.finalize_budget("vp2_pd", _vp2_out12_np / _dt12, l_count_sample=False)

                # ---- wp2/wp3 budget terms (pre/post advance values) ----
                _g = 1.5   # gamma_over_implicit_ts
                _wp2_pre = _asarray(_wp2_pre12, dtype=np.float64)
                _wp3_pre = _asarray(_wp3_pre12, dtype=np.float64)
                _wp2_post = _asarray(_wp2_jax12_raw, dtype=np.float64)  # pre-clip post-solve
                _wp3_post = _asarray(_wp3_jax12_raw, dtype=np.float64)  # pre-clip post-solve
                _wp2_mix = (1.0 - _g) * _wp2_pre + _g * _wp2_post
                _wp3_mix = (1.0 - _g) * _wp3_pre + _g * _wp3_post

                # ------ wp2 budget terms ------

                # wp2_bp: C_uu_buoy=0 → 2*g/thv*wpthvp
                stats_writer.update("wp2_bp",   _asarray(_sd12['rhs_bp_wp2'], dtype=np.float64))
                # wp2_pr3: explicit RHS pressure term 3
                stats_writer.update("wp2_pr3",  _asarray(_sd12['rhs_pr3_wp2'], dtype=np.float64))
                # wp2_splat: in Fortran stats_update("wp2_splat", -lhs_splat_wp2*wp2_old) before solve
                stats_writer.update("wp2_splat", -(_asarray(lhs_splat_wp2, dtype=np.float64) * _wp2_pre))

                # wp2_dp1: rhs_dp1_wp2 - lhs_dp1_wp2 * wp2_mix
                _lhs_dp1 = _asarray(_sd12['lhs_dp1_wp2'], dtype=np.float64)
                _rhs_dp1 = _asarray(_sd12['rhs_dp1_wp2'], dtype=np.float64)
                _wp2_dp1 = _xp.zeros_like(_wp2_pre)
                _wp2_dp1 = _iset(_wp2_dp1, np.s_[:, 1:-1], _rhs_dp1[:, 1:-1] - _lhs_dp1[:, 1:-1] * _wp2_mix[:, 1:-1])
                stats_writer.update("wp2_dp1", _wp2_dp1)

                # wp2_pr1: rhs_pr1_wp2 - lhs_pr1_wp2 * wp2_mix  (l_tke_aniso=True)
                _lhs_pr1 = _asarray(_sd12['lhs_pr1_wp2'], dtype=np.float64)
                _rhs_pr1 = _asarray(_sd12['rhs_pr1_wp2'], dtype=np.float64)
                _wp2_pr1 = _xp.zeros_like(_wp2_pre)
                _wp2_pr1 = _iset(_wp2_pr1, np.s_[:, 1:-1], _rhs_pr1[:, 1:-1] - _lhs_pr1[:, 1:-1] * _wp2_mix[:, 1:-1])
                stats_writer.update("wp2_pr1", _wp2_pr1)

                # wp2_pr2: rhs_pr2_wp2 - lhs_wp2_pr2_term * wp2_post
                _rhs_pr2 = _asarray(_sd12['rhs_pr2_wp2'], dtype=np.float64)
                _lhs_pr2t = _asarray(_sd12['lhs_wp2_pr2_term'], dtype=np.float64)
                _wp2_pr2 = _xp.zeros_like(_wp2_pre)
                _wp2_pr2 = _iset(_wp2_pr2, np.s_[:, 1:-1], _rhs_pr2[:, 1:-1] - _lhs_pr2t[:, 1:-1] * _wp2_post[:, 1:-1])
                stats_writer.update("wp2_pr2", _wp2_pr2)

                # wp2_dp2: fully implicit, -(lhs_diff_zm @ wp2_post) tri-band
                _lhs_dz = _asarray(_sd12['lhs_diff_zm'], dtype=np.float64)  # (3, ngrdcol, nzm)
                _wp2_dp2 = _xp.zeros_like(_wp2_pre)
                _wp2_dp2 = _iset(_wp2_dp2, np.s_[:, 1:-1], -(
                    _lhs_dz[0, :, 1:-1] * _wp2_post[:, 2:]
                    + _lhs_dz[1, :, 1:-1] * _wp2_post[:, 1:-1]
                    + _lhs_dz[2, :, 1:-1] * _wp2_post[:, :-2]
                ))
                stats_writer.update("wp2_dp2", _wp2_dp2)

                # wp2_ta: fully implicit, -(lhs_ta_wp2 @ wp3_post)
                _lhs_ta2 = _asarray(_sd12['lhs_ta_wp2'], dtype=np.float64)  # (2, ngrdcol, nzm)
                _wp2_ta = _xp.zeros_like(_wp2_pre)
                _wp2_ta = _iset(_wp2_ta, np.s_[:, 1:-1], -(
                    _lhs_ta2[0, :, 1:-1] * _wp3_post[:, 1:]   # wp3[k_py]
                    + _lhs_ta2[1, :, 1:-1] * _wp3_post[:, :-1]  # wp3[k_py-1]
                ))
                stats_writer.update("wp2_ta", _wp2_ta)

                # wp2_pr_dfsn already in stats? Skip (passes). wp2_ac, wp2_ma → already pass.

                # ------ wp3 budget terms ------

                # wp3_bp1: C11_Skw_fnc=0 → 3*g/thv*wp2thvp
                stats_writer.update("wp3_bp1",     _asarray(_sd12['rhs_bp1_wp3'], dtype=np.float64))
                # wp3_pr_turb: explicit RHS
                stats_writer.update("wp3_pr_turb", _asarray(_sd12['rhs_pr_turb_wp3'], dtype=np.float64))

                # wp3_pr1: rhs_pr1_wp3 - lhs_pr1_wp3 * wp3_mix
                _lhs_pr1_3 = _asarray(_sd12['lhs_pr1_wp3'], dtype=np.float64)
                _rhs_pr1_3 = _asarray(_sd12['rhs_pr1_wp3'], dtype=np.float64)
                _wp3_pr1 = _xp.zeros_like(_wp3_pre)
                _wp3_pr1 = _iset(_wp3_pr1, np.s_[:, 1:-1], _rhs_pr1_3[:, 1:-1] - _lhs_pr1_3[:, 1:-1] * _wp3_mix[:, 1:-1])
                stats_writer.update("wp3_pr1", _wp3_pr1)

                # wp3_pr2: rhs_pr2_wp3 - lhs_wp3_pr2_term * wp3_post
                _rhs_pr2_3 = _asarray(_sd12['rhs_pr2_wp3'], dtype=np.float64)
                _lhs_pr2t_3 = _asarray(_sd12['lhs_wp3_pr2_term'], dtype=np.float64)
                _wp3_pr2 = _xp.zeros_like(_wp3_pre)
                _wp3_pr2 = _iset(_wp3_pr2, np.s_[:, 1:-1], _rhs_pr2_3[:, 1:-1] - _lhs_pr2t_3[:, 1:-1] * _wp3_post[:, 1:-1])
                stats_writer.update("wp3_pr2", _wp3_pr2)

                # wp3_dp1: fully implicit (ARM: l_crank_nich_diff=False), -(lhs_diff_zt @ wp3_post)
                _lhs_dt = _asarray(_sd12['lhs_diff_zt'], dtype=np.float64)  # (3, ngrdcol, nzt)
                _wp3_dp1 = _xp.zeros_like(_wp3_pre)
                _wp3_dp1 = _iset(_wp3_dp1, np.s_[:, 1:-1], -(
                    _lhs_dt[0, :, 1:-1] * _wp3_post[:, 2:]
                    + _lhs_dt[1, :, 1:-1] * _wp3_post[:, 1:-1]
                    + _lhs_dt[2, :, 1:-1] * _wp3_post[:, :-2]
                ))
                stats_writer.update("wp3_dp1", _wp3_dp1)

                # wp3_ta: -(lhs_ta_wp3 @ mixed) 5-band
                _lhs_ta3 = _asarray(_sd12['lhs_ta_wp3'], dtype=np.float64)  # (5, ngrdcol, nzt)
                _wp3_ta = _xp.zeros_like(_wp3_pre)
                _wp3_ta = _iset(_wp3_ta, np.s_[:, 1:-1], -(
                    _lhs_ta3[0, :, 1:-1] * _wp3_mix[:, 2:]     # wp3[k+1]
                    + _lhs_ta3[1, :, 1:-1] * _wp2_mix[:, 2:-1]   # wp2[k+1]
                    + _lhs_ta3[2, :, 1:-1] * _wp3_mix[:, 1:-1]   # wp3[k]
                    + _lhs_ta3[3, :, 1:-1] * _wp2_mix[:, 1:-2]   # wp2[k]
                    + _lhs_ta3[4, :, 1:-1] * _wp3_mix[:, :-2]    # wp3[k-1]
                ))
                stats_writer.update("wp3_ta", _wp3_ta)

                # wp3_tp: -(lhs_adv_tp_wp3 @ wp2_mix)
                _lhs_tp3 = _asarray(_sd12['lhs_adv_tp_wp3'], dtype=np.float64)  # (2, ngrdcol, nzt)
                _wp3_tp = _xp.zeros_like(_wp3_pre)
                _wp3_tp = _iset(_wp3_tp, np.s_[:, 1:-1], -(
                    _lhs_tp3[0, :, 1:-1] * _wp2_mix[:, 2:-1]   # wp2[k+1]
                    + _lhs_tp3[1, :, 1:-1] * _wp2_mix[:, 1:-2]   # wp2[k]
                ))
                stats_writer.update("wp3_tp", _wp3_tp)

            # ============================================================ #
            # Iter35: Override advance_wp2_wp3 state with JAX values        #
            # wp2/wp3/wp2_zt verified at machine epsilon (iter12).          #
            # ============================================================ #
            wp2    = _asarray(_wp2_jax12,    dtype=np.float64).copy()
            wp3    = _asarray(_wp3_jax12,    dtype=np.float64).copy()
            wp2_zt = _asarray(_wp2_zt_jax12, dtype=np.float64).copy()

            # Iter45: JAX clip_covars_denom directly (Fortran oracle removed; verified 0.000e+00)
            # ARM: l_linearize_pbl_winds=False → upwp_pert/vpwp_pert unchanged
            # ARM: sclr_dim=0 → wpsclrp unchanged
            (_wprtp_j24b, _wpthlp_j24b, _upwp_j24b, _vpwp_j24b) = clip_covars_denom_jax(
                wprtp=jnp.asarray(_asarray(wprtp, dtype=np.float64)),
                wpthlp=jnp.asarray(_asarray(wpthlp, dtype=np.float64)),
                upwp=jnp.asarray(_asarray(upwp, dtype=np.float64)),
                vpwp=jnp.asarray(_asarray(vpwp, dtype=np.float64)),
                wp2=jnp.asarray(_asarray(wp2, dtype=np.float64)),
                rtp2=jnp.asarray(_asarray(rtp2, dtype=np.float64)),
                thlp2=jnp.asarray(_asarray(thlp2, dtype=np.float64)),
                up2=jnp.asarray(_asarray(up2, dtype=np.float64)),
                vp2=jnp.asarray(_asarray(vp2, dtype=np.float64)),
                l_tke_aniso=flags.l_tke_aniso,
            )
            wprtp  = _asarray(_wprtp_j24b,  dtype=np.float64).copy()
            wpthlp = _asarray(_wpthlp_j24b, dtype=np.float64).copy()
            upwp   = _asarray(_upwp_j24b,   dtype=np.float64).copy()
            vpwp   = _asarray(_vpwp_j24b,   dtype=np.float64).copy()
            err_code = err_info.err_code
            if err_code is not None and np.any(_asarray(err_code) == CLUBB_FATAL_ERROR):
                return

        elif advance_iter == order_windm_val:
            # ---- Save pre-call state for iter13 shadow comparison ----
            _um_pre13   = _asarray(um).copy()
            _vm_pre13   = _asarray(vm).copy()
            _upwp_pre13 = _asarray(upwp).copy()
            _vpwp_pre13 = _asarray(vpwp).copy()

            # Iter49: Fortran advance_windm_edsclrm oracle removed.

            # ============================================================ #
            # Block X: Iteration 13 shadow comparison                       #
            # advance_windm_edsclrm JAX vs Fortran                          #
            # ARM: l_predict_upwp_vpwp=True → no-op, expect 0 error        #
            # ============================================================ #
            _nu10_13 = float(_asarray(nu_vert_res_dep.nu10, dtype=np.float64).flat[0])
            _um_jax13, _vm_jax13, _upwp_jax13, _vpwp_jax13 = advance_windm_edsclrm_jax(
                um=jnp.array(_um_pre13),
                vm=jnp.array(_vm_pre13),
                upwp=jnp.array(_upwp_pre13),
                vpwp=jnp.array(_vpwp_pre13),
                wp2=jnp.array(wp2),
                up2=jnp.array(up2),
                vp2=jnp.array(vp2),
                wm_zt=jnp.array(wm_zt),
                Kh_zm=jnp.array(Kh_zm),
                ug=jnp.array(ug),
                vg=jnp.array(vg),
                um_forcing=jnp.array(um_forcing),
                vm_forcing=jnp.array(vm_forcing),
                rho_ds_zm=jnp.array(rho_ds_zm),
                rho_ds_zt=jnp.array(rho_ds_zt),
                invrs_rho_ds_zt=jnp.array(invrs_rho_ds_zt),
                fcor=jnp.array(fcor),
                clubb_params=jnp.array(clubb_params),
                nu10=_nu10_13,
                dt=float(dt),
                gr=gr,
                l_predict_upwp_vpwp=bool(flags.l_predict_upwp_vpwp),
                l_upwind_xm_ma=bool(flags.l_upwind_xm_ma),
                l_tke_aniso=bool(flags.l_tke_aniso),
            )
            # Iter49: Fortran oracle removed; JAX results are the state.
            # Iter42: override advance_windm_edsclrm state with JAX values
            # (no-op for ARM with l_predict_upwp_vpwp=True, but explicit for completeness)
            um   = _asarray(_um_jax13,   dtype=np.float64).copy()
            vm   = _asarray(_vm_jax13,   dtype=np.float64).copy()
            upwp = _asarray(_upwp_jax13, dtype=np.float64).copy()
            vpwp = _asarray(_vpwp_jax13, dtype=np.float64).copy()

    # Update local aliases after advance loop
    wp2 = wp2
    wp2_zt = _xp.maximum(
        _asarray(zm2zt(wp2, gr)),
        w_tol_sqd,
    )

    # ================================================================== #
    # Block T: Advance or diagnose third-order moments (xp3)
    # ================================================================== #
    if l_advance_xp3_flag and flags.iiPDF_type != iiPDF_ADG1:
        # Iter61: JAX advance_xp3 for non-ADG1 PDF (steady-state, sclr_dim=0)
        rtp3, thlp3, up3, vp3 = [
            _asarray(x, dtype=np.float64)
            for x in advance_xp3_jax(
                jnp.asarray(rtm), jnp.asarray(thlm),
                jnp.asarray(rtp2), jnp.asarray(thlp2),
                jnp.asarray(wprtp), jnp.asarray(wpthlp),
                jnp.asarray(wprtp2), jnp.asarray(wpthlp2),
                jnp.asarray(rho_ds_zm), jnp.asarray(invrs_rho_ds_zt),
                jnp.asarray(invrs_tau_zt), jnp.asarray(tau_max_zt),
                jnp.asarray(wp2), jnp.asarray(wp3),
                jnp.asarray(upwp), jnp.asarray(vpwp),
                jnp.asarray(up2), jnp.asarray(vp2),
                jnp.asarray(thvm), jnp.asarray(clubb_params),
                gr,
            )
        ]
        # sclrp3 unchanged (sclr_dim=0)
    else:
        # Iter52: JAX-only compute_xp3 (Fortran oracle removed)
        if flags.iiPDF_type == iiPDF_ADG1:
            rtp3, thlp3, up3, vp3 = [
                _asarray(x, dtype=np.float64)
                for x in compute_xp3_jax(
                    jnp.asarray(wp2), jnp.asarray(wp3),
                    jnp.asarray(wprtp), jnp.asarray(wpthlp),
                    jnp.asarray(rtp2), jnp.asarray(thlp2),
                    jnp.asarray(upwp), jnp.asarray(vpwp),
                    jnp.asarray(up2), jnp.asarray(vp2),
                    jnp.asarray(sigma_sqd_w),
                    jnp.asarray(clubb_params),
                    gr,
                )
            ]
            # Iter68: advance_xp3_module.F90 stats (compute_xp3 path = ADG1)
            if l_sample and stats_writer is not None:
                _ssw_zt_68 = _asarray(
                    jnp.maximum(zm2zt_jax(jnp.asarray(sigma_sqd_w), gr), zero_threshold),
                    dtype=np.float64,
                )
                stats_writer.update("sigma_sqd_w_zt", _ssw_zt_68)
                # wprtp_zt, wpthlp_zt, upwp_zt, vpwp_zt: zm2zt of post-advance values
                stats_writer.update("wprtp_zt",  _asarray(zm2zt_jax(jnp.asarray(wprtp),  gr), dtype=np.float64))
                stats_writer.update("wpthlp_zt", _asarray(zm2zt_jax(jnp.asarray(wpthlp), gr), dtype=np.float64))
                stats_writer.update("upwp_zt",   _asarray(zm2zt_jax(jnp.asarray(upwp),   gr), dtype=np.float64))
                stats_writer.update("vpwp_zt",   _asarray(zm2zt_jax(jnp.asarray(vpwp),   gr), dtype=np.float64))
                # rtp2_zt, thlp2_zt, up2_zt, vp2_zt: zm2zt with floor (from advance_xp3 Fortran path)
                stats_writer.update("rtp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(rtp2), gr), float(rt_tol**2)), dtype=np.float64))
                stats_writer.update("thlp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(thlp2), gr), float(thl_tol**2)), dtype=np.float64))
                stats_writer.update("up2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(up2), gr), float(w_tol_sqd)), dtype=np.float64))
                stats_writer.update("vp2_zt",
                    _asarray(jnp.maximum(
                        zm2zt_jax(jnp.asarray(vp2), gr), float(w_tol_sqd)), dtype=np.float64))

    # ================================================================== #
    # Block U: Post-advance PDF closure (conditional)
    # ================================================================== #
    if (flags.ipdf_call_placement == ipdf_post_advance_fields
            or flags.ipdf_call_placement == ipdf_pre_post_advance_fields):

        if flags.ipdf_call_placement == ipdf_post_advance_fields:
            l_samp_stats = True
        else:
            l_samp_stats = False  # already sampled in pre-advance call

        # Iter50: Fortran pdf_closure_driver oracle removed.
        # All state updates are handled by iter34 override below using JAX-computed values.

        # Iter25 shadow (post-advance path): compare ADG1_pdf_driver_jax vs Fortran
        if flags.iiPDF_type == iiPDF_ADG1:
            # Recompute zt-level fields using post-advance state (mirrors Fortran internals)
            _Skw_zt_j25 = skx_func_jax(
                jnp.asarray(wp2_zt), jnp.asarray(wp3), w_tol, jnp.asarray(clubb_params))
            _rtp2_zt_j25 = jnp.maximum(
                zm2zt_jax(jnp.asarray(rtp2), gr), rt_tol ** 2)
            _thlp2_zt_j25 = jnp.maximum(
                zm2zt_jax(jnp.asarray(thlp2), gr), thl_tol ** 2)
            _up2_zt_j25 = jnp.maximum(zm2zt_jax(jnp.asarray(up2), gr), w_tol_sqd)
            _vp2_zt_j25 = jnp.maximum(zm2zt_jax(jnp.asarray(vp2), gr), w_tol_sqd)

            # Fortran pdf_closure_driver recomputes sigma_sqd_w from post-advance state
            # before calling ADG1_pdf_driver.  Replicate that here so _adg1_j25 gets
            # the correct post-advance sigma_sqd_w (not the pre-advance value from iter18/39).
            _wp3_zm_j34 = _asarray(zt2zm_jax(jnp.asarray(wp3), gr))
            _Skw_zm_j34 = _asarray(skx_func_jax(
                jnp.asarray(wp2), jnp.asarray(_wp3_zm_j34), w_tol, jnp.asarray(clubb_params)))
            _gc_j34 = clubb_params[:, igamma_coef - 1]
            _gb_j34 = clubb_params[:, igamma_coefb - 1]
            _gcf_j34 = clubb_params[:, igamma_coefc - 1]
            _gamma_j34 = np.empty((ngrdcol, nzm))
            if l_gamma_skw:
                for _k34 in range(nzm):
                    for _i34 in range(ngrdcol):
                        if abs(_gc_j34[_i34] - _gb_j34[_i34]) > abs(_gc_j34[_i34] + _gb_j34[_i34]) * eps / 2:
                            _gamma_j34 = _iset(_gamma_j34, np.s_[_i34, _k34], _gb_j34[_i34] + (_gc_j34[_i34] - _gb_j34[_i34]) * _xp.exp(
                                -0.5 * (_Skw_zm_j34[_i34, _k34] / _gcf_j34[_i34]) ** 2))
                        else:
                            _gamma_j34 = _iset(_gamma_j34, np.s_[_i34, _k34], _gc_j34[_i34])
            else:
                _gamma_j34 = _xp.broadcast_to(
                    clubb_params[:, igamma_coef - 1:igamma_coef], (ngrdcol, nzm)).copy()
            _ssw_j34 = _asarray(compute_sigma_sqd_w_jax(
                jnp.asarray(_gamma_j34),
                jnp.asarray(wp2), jnp.asarray(thlp2), jnp.asarray(rtp2),
                jnp.asarray(up2), jnp.asarray(vp2),
                jnp.asarray(wpthlp), jnp.asarray(wprtp),
                jnp.asarray(upwp), jnp.asarray(vpwp),
                flags.l_predict_upwp_vpwp, gr,
            ))
            _sigma_sqd_w_zt_j25 = jnp.maximum(
                zm2zt_jax(jnp.asarray(_ssw_j34), gr), zero_threshold)
            _wprtp_zt_j25 = zm2zt_jax(jnp.asarray(wprtp), gr)
            _wpthlp_zt_j25 = zm2zt_jax(jnp.asarray(wpthlp), gr)
            _upwp_zt_j25 = zm2zt_jax(jnp.asarray(upwp), gr)
            _vpwp_zt_j25 = zm2zt_jax(jnp.asarray(vpwp), gr)
            _sqrt_wp2_zt_j25 = jnp.sqrt(jnp.asarray(wp2_zt))
            _beta_j25 = jnp.asarray(clubb_params[:, ibeta - 1])

            _adg1_j25 = ADG1_pdf_driver_jax(
                wm=jnp.asarray(wm_zt),
                rtm=jnp.asarray(rtm),
                thlm=jnp.asarray(thlm),
                um=jnp.asarray(um),
                vm=jnp.asarray(vm),
                wp2=jnp.asarray(wp2_zt),
                rtp2=_rtp2_zt_j25,
                thlp2=_thlp2_zt_j25,
                up2=_up2_zt_j25,
                vp2=_vp2_zt_j25,
                Skw=_Skw_zt_j25,
                wprtp=_wprtp_zt_j25,
                wpthlp=_wpthlp_zt_j25,
                upwp=_upwp_zt_j25,
                vpwp=_vpwp_zt_j25,
                sqrt_wp2=_sqrt_wp2_zt_j25,
                sigma_sqd_w=_sigma_sqd_w_zt_j25,
                beta=_beta_j25,
                mixt_frac_max_mag=mixt_frac_max_mag,
            )

            # Iter50: Fortran oracle removed; comparison vs pdf_params deactivated.
            # Iter40: save Block U ADG1 result for next timestep's Block P override.
            # REFACTOR B4 (iter21): do NOT write the module global under a JAX trace — storing tracers in
            # global state leaks them across calls (UnexpectedTracerError) and makes grad non-composable.
            # The carry is a cross-timestep convenience, not needed for differentiation of a single step.
            if not _is_tracer_arg(list(_adg1_j25.values())):
                _prev_adg1_j25 = _adg1_j25

            if l_sample and l_samp_stats and stats_writer is not None:
                stats_writer.update("mixt_frac",    _asarray(_adg1_j25['mixt_frac'],    dtype=np.float64))
                stats_writer.update("w_1",          _asarray(_adg1_j25['w_1'],          dtype=np.float64))
                stats_writer.update("w_2",          _asarray(_adg1_j25['w_2'],          dtype=np.float64))
                stats_writer.update("varnce_w_1",   _asarray(_adg1_j25['varnce_w_1'],   dtype=np.float64))
                stats_writer.update("varnce_w_2",   _asarray(_adg1_j25['varnce_w_2'],   dtype=np.float64))
                stats_writer.update("rt_1",         _asarray(_adg1_j25['rt_1'],         dtype=np.float64))
                stats_writer.update("rt_2",         _asarray(_adg1_j25['rt_2'],         dtype=np.float64))
                stats_writer.update("varnce_rt_1",  _asarray(_adg1_j25['varnce_rt_1'],  dtype=np.float64))
                stats_writer.update("varnce_rt_2",  _asarray(_adg1_j25['varnce_rt_2'],  dtype=np.float64))
                stats_writer.update("thl_1",        _asarray(_adg1_j25['thl_1'],        dtype=np.float64))
                stats_writer.update("thl_2",        _asarray(_adg1_j25['thl_2'],        dtype=np.float64))
                stats_writer.update("varnce_thl_1", _asarray(_adg1_j25['varnce_thl_1'], dtype=np.float64))
                stats_writer.update("varnce_thl_2", _asarray(_adg1_j25['varnce_thl_2'], dtype=np.float64))

            # Iter26 shadow: calc_comp_corrs_binormal for rt-thl (always for ADG1)
            _rtpthlp_zt_j26 = zm2zt_jax(jnp.asarray(rtpthlp), gr)
            (_corr_rt_thl_1_j26,
             _corr_rt_thl_2_j26) = calc_comp_corrs_binormal_jax(
                xpyp=_rtpthlp_zt_j26,
                xm=jnp.asarray(rtm),
                ym=jnp.asarray(thlm),
                mu_x_1=_adg1_j25['rt_1'],
                mu_x_2=_adg1_j25['rt_2'],
                mu_y_1=_adg1_j25['thl_1'],
                mu_y_2=_adg1_j25['thl_2'],
                sigma_x_1_sqd=_adg1_j25['varnce_rt_1'],
                sigma_x_2_sqd=_adg1_j25['varnce_rt_2'],
                sigma_y_1_sqd=_adg1_j25['varnce_thl_1'],
                sigma_y_2_sqd=_adg1_j25['varnce_thl_2'],
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # Iter50: Fortran oracle removed; comparison vs pdf_params deactivated.

            # ============================================================ #
            # Iter60 (Block U): JAX rcm/cloud_frac from post-advance PDF   #
            # Mirrors Fortran pdf_closure_driver → transform_pdf_chi_eta   #
            # + calc_liquid_cloud_frac_component for the post-advance path. #
            # rcm is returned and becomes the NEXT timestep's Block I/K    #
            # input, replacing what Fortran's Block U oracle used to give.  #
            # ============================================================ #
            _mf60    = _adg1_j25['mixt_frac']
            _rt1_60  = _adg1_j25['rt_1'];   _rt2_60  = _adg1_j25['rt_2']
            _thl1_60 = _adg1_j25['thl_1'];  _thl2_60 = _adg1_j25['thl_2']
            _vrt1_60 = _adg1_j25['varnce_rt_1'];  _vrt2_60 = _adg1_j25['varnce_rt_2']
            _vthl1_60 = _adg1_j25['varnce_thl_1']; _vthl2_60 = _adg1_j25['varnce_thl_2']

            # Liquid water temperature: tl_i = thl_i * exner (zt grid)
            _exner60 = jnp.asarray(exner)
            _p60     = jnp.asarray(p_in_Pa)
            _tl1_60  = _thl1_60 * _exner60
            _tl2_60  = _thl2_60 * _exner60

            # Saturation mixing ratio
            _rsatl1_60 = sat_mixrat_liq_jax(_p60, _tl1_60, flags.saturation_formula)
            _rsatl2_60 = sat_mixrat_liq_jax(_p60, _tl2_60, flags.saturation_formula)

            # Chi-eta transform: chi_i, crt_i, cthl_i, stdev_chi_i, stdev_eta_i, corr_chi_eta_i
            # Mirrors Fortran transform_pdf_chi_eta_component (pdf_closure_module.F90:1699)
            def _chi60(tl, rsatl, rt, exner_in, varnce_rt, varnce_thl, corr_rt_thl):
                beta       = ep * Lv**2 / (Rd * Cp * tl**2)
                invrs      = 1.0 / (1.0 + beta * rsatl)
                chi        = (rt - rsatl) * invrs
                crt        = invrs
                cthl       = ((1.0 + beta * rt) * invrs**2
                              * (Cp / Lv) * beta * rsatl * exner_in)
                vrnc_rt_t  = crt**2 * varnce_rt
                vrnc_thl_t = cthl**2 * varnce_thl
                corr_t     = (2.0 * corr_rt_thl * crt * cthl
                              * jnp.sqrt(varnce_rt * varnce_thl))
                vrnc_chi   = vrnc_rt_t - corr_t + vrnc_thl_t
                vrnc_eta   = vrnc_rt_t + corr_t + vrnc_thl_t
                stdev_chi  = _safe_sqrt(vrnc_chi)
                stdev_eta  = _safe_sqrt(vrnc_eta)
                covar_ce   = vrnc_rt_t - vrnc_thl_t
                # smooth_corr_quotient (pdf_utilities.F90:1360)
                _denom_thresh = chi_tol**2
                _smth = min(min_max_smth_mag, _denom_thresh)
                denom = stdev_chi * stdev_eta
                tmp_d = smooth_max_jax(jnp.abs(covar_ce) / max_mag_correlation, denom, _smth)
                tmp_d = smooth_max_jax(tmp_d, _denom_thresh, _smth)
                corr_chi_eta = covar_ce / tmp_d
                return chi, stdev_chi, crt, cthl, stdev_eta, corr_chi_eta

            (_chi1_60, _schi1_60, _crt1_60, _cthl1_60,
             _seta1_60, _corr_ce1_60) = _chi60(
                _tl1_60, _rsatl1_60, _rt1_60, _exner60,
                _vrt1_60, _vthl1_60, _corr_rt_thl_1_j26)
            (_chi2_60, _schi2_60, _crt2_60, _cthl2_60,
             _seta2_60, _corr_ce2_60) = _chi60(
                _tl2_60, _rsatl2_60, _rt2_60, _exner60,
                _vrt2_60, _vthl2_60, _corr_rt_thl_2_j26)

            # Component cloud fraction (Gaussian CDF with ±max_num_stdevs truncation)
            def _liq_cf60(mean_chi, stdev_chi_in):
                is_clear = (
                    ((jnp.abs(mean_chi) <= eps) & (stdev_chi_in <= chi_tol))
                    | (mean_chi < -max_num_stdevs * stdev_chi_in)
                )
                is_full = mean_chi > max_num_stdevs * stdev_chi_in
                safe_s  = jnp.maximum(stdev_chi_in, 1.0e-100)
                zeta    = mean_chi / safe_s
                cf_mid  = 0.5 * (1.0 + jax.scipy.special.erf(zeta / sqrt_2))
                rc_mid  = (mean_chi * cf_mid
                           + stdev_chi_in * jnp.exp(-0.5 * zeta**2) / sqrt_2pi)
                cf = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, cf_mid))
                rc = jnp.where(is_clear, 0.0, jnp.where(is_full, mean_chi, rc_mid))
                return cf, rc

            _cf1_60, _rc1_60 = _liq_cf60(_chi1_60, _schi1_60)
            _cf2_60, _rc2_60 = _liq_cf60(_chi2_60, _schi2_60)

            # Combine PDF components → rcm, cloud_frac for NEXT timestep's Block I/K
            _cloud_frac_60 = _mf60 * _cf1_60 + (1.0 - _mf60) * _cf2_60
            _rcm_60        = jnp.maximum(0.0, _mf60 * _rc1_60 + (1.0 - _mf60) * _rc2_60)

            rcm        = _asarray(_rcm_60,        dtype=np.float64)
            cloud_frac = _asarray(_cloud_frac_60, dtype=np.float64)

            # Update ice_supersat_frac from Block U ADG1 output (l_calc_ice_supersat_frac
            # is hardcoded .true. in pdf_closure_module.F90:928). Faithful port of
            # calc_ice_cloud_frac_component (pdf_closure_module.F90:2490): for levels above
            # freezing it equals the liquid cloud_frac; for below-freezing levels it is the
            # PDF fraction supersaturated w.r.t. ICE (chi above chi_at_ice_sat). The old
            # warm-only shortcut (=cloud_frac) gave 0 at cold, ice-supersaturated layers
            # (e.g. ekman's 10 km top, T~203 K), corrupting the splat Brunt-Vaisala term.
            def _ice_cf60(mean_chi, stdev_chi_in, crt, rsatl, tl, cf_liq):
                rsat_ice = sat_mixrat_ice_jax(_p60, tl)
                delta    = mean_chi - crt * (rsat_ice - rsatl)   # chi - chi_at_ice_sat
                is_clear = (((jnp.abs(delta) <= eps) & (stdev_chi_in <= chi_tol))
                            | (delta < -max_num_stdevs * stdev_chi_in))
                is_full  = delta > max_num_stdevs * stdev_chi_in
                safe_s   = jnp.maximum(stdev_chi_in, 1.0e-100)
                zeta     = delta / safe_s
                ssf_mid  = 0.5 * (1.0 + jax.scipy.special.erf(zeta / sqrt_2))
                ssf      = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, ssf_mid))
                # Above freezing: same as the liquid cloud-fraction component.
                return jnp.where(tl > T_freeze_K, cf_liq, ssf)

            _issf1_60 = _ice_cf60(_chi1_60, _schi1_60, _crt1_60, _rsatl1_60, _tl1_60, _cf1_60)
            _issf2_60 = _ice_cf60(_chi2_60, _schi2_60, _crt2_60, _rsatl2_60, _tl2_60, _cf2_60)
            _issf_60  = _mf60 * _issf1_60 + (1.0 - _mf60) * _issf2_60
            ice_supersat_frac = _asarray(_issf_60, dtype=np.float64)

            # Iter156: propagate the zt-level PDF component moments into the returned pdf_params so the
            # KK microphysics (kk_microphys_step) can read them. The JAX otherwise computes them as Block-U
            # locals that flow only to stats; pdf_params was zero-initialized (it is a fallback for
            # non-ADG1/non-ARM, :1815), and the 15 non-microphysics cases never read these fields — so
            # populating them is safe (verified: ARM/bomex still bit-faithful).
            pdf_params = pdf_params._replace(
                chi_1=_asarray(_chi1_60, np.float64), chi_2=_asarray(_chi2_60, np.float64),
                stdev_chi_1=_asarray(_schi1_60, np.float64), stdev_chi_2=_asarray(_schi2_60, np.float64),
                cloud_frac_1=_asarray(_cf1_60, np.float64), cloud_frac_2=_asarray(_cf2_60, np.float64),
                rc_1=_asarray(_rc1_60, np.float64), rc_2=_asarray(_rc2_60, np.float64),
                mixt_frac=_asarray(_mf60, np.float64),
                thl_1=_asarray(_thl1_60, np.float64), thl_2=_asarray(_thl2_60, np.float64),
                ice_supersat_frac_1=_asarray(_issf1_60, np.float64),
                ice_supersat_frac_2=_asarray(_issf2_60, np.float64),
                # Iter172: the additional component moments the KK second-moment covariance driver
                # (KK_upscaled_covar_driver) consumes — w/eta/rt means+stdevs, the chi-eta transform
                # coefficients, and corr_chi_eta. corr_w_chi/corr_w_eta stay 0 (ADG1, pdf_closure:1037).
                w_1=_asarray(_adg1_j25['w_1'], np.float64), w_2=_asarray(_adg1_j25['w_2'], np.float64),
                varnce_w_1=_asarray(_adg1_j25['varnce_w_1'], np.float64),
                varnce_w_2=_asarray(_adg1_j25['varnce_w_2'], np.float64),
                rt_1=_asarray(_rt1_60, np.float64), rt_2=_asarray(_rt2_60, np.float64),
                stdev_eta_1=_asarray(_seta1_60, np.float64), stdev_eta_2=_asarray(_seta2_60, np.float64),
                crt_1=_asarray(_crt1_60, np.float64), crt_2=_asarray(_crt2_60, np.float64),
                cthl_1=_asarray(_cthl1_60, np.float64), cthl_2=_asarray(_cthl2_60, np.float64),
                corr_chi_eta_1=_asarray(_corr_ce1_60, np.float64),
                corr_chi_eta_2=_asarray(_corr_ce2_60, np.float64))

            # ============================================================ #
            # Iter61 (Block U): JAX cloud water flux variables             #
            # Mirrors Fortran calc_xprcp_component (pdf_closure_module.F90 #
            # lines 3089-3104). For ADG1 no corr_w_chi correction (lines  #
            # 3112-3138 only run for non-ADG1 PDF types).                  #
            # After mixing: convert zt→zm, zero k_ub_zm (lines 4233-4261).#
            # ============================================================ #
            _wm_zt_61  = jnp.asarray(wm_zt)
            _rtm_61    = jnp.asarray(rtm)
            _thlm_61   = jnp.asarray(thlm)
            _um_61     = jnp.asarray(um)
            _vm_61     = jnp.asarray(vm)
            _rcm_zt_61 = _rcm_60  # on zt grid

            # Per-component contributions: wprcp_i = (w_i - wm) * (rc_i - rcm)
            _drc1_61 = _rc1_60 - _rcm_zt_61
            _drc2_61 = _rc2_60 - _rcm_zt_61
            _wprcp_c1  = (_adg1_j25['w_1'] - _wm_zt_61) * _drc1_61
            _wprcp_c2  = (_adg1_j25['w_2'] - _wm_zt_61) * _drc2_61

            # wp2rcp_i = ((w_i - wm)^2 + varnce_w_i) * (rc_i - rcm)
            _wp2rcp_c1 = ((_adg1_j25['w_1'] - _wm_zt_61)**2
                          + _adg1_j25['varnce_w_1']) * _drc1_61
            _wp2rcp_c2 = ((_adg1_j25['w_2'] - _wm_zt_61)**2
                          + _adg1_j25['varnce_w_2']) * _drc2_61

            # rtprcp_i = (rt_i - rtm)*(rc_i - rcm)
            #           + (corr_chi_eta_i*stdev_eta_i + stdev_chi_i) / (2*crt_i)
            #             * stdev_chi_i * cloud_frac_i
            # Safe: crt = 1/(1+beta*rsatl) > 0 always
            _rtprcp_c1 = ((_adg1_j25['rt_1'] - _rtm_61) * _drc1_61
                          + (_corr_ce1_60 * _seta1_60 + _schi1_60)
                            / (2.0 * _crt1_60) * _schi1_60 * _cf1_60)
            _rtprcp_c2 = ((_adg1_j25['rt_2'] - _rtm_61) * _drc2_61
                          + (_corr_ce2_60 * _seta2_60 + _schi2_60)
                            / (2.0 * _crt2_60) * _schi2_60 * _cf2_60)

            # thlprcp_i = (thl_i - thlm)*(rc_i - rcm)
            #            + (corr_chi_eta_i*stdev_eta_i - stdev_chi_i) / (2*cthl_i)
            #              * stdev_chi_i * cloud_frac_i
            # Guard against cthl=0 (rsatl=0 limit); cf=0 masks result
            _cthl1_safe = jnp.where(_cthl1_60 == 0.0, 1.0, _cthl1_60)
            _cthl2_safe = jnp.where(_cthl2_60 == 0.0, 1.0, _cthl2_60)
            _thlprcp_c1 = ((_adg1_j25['thl_1'] - _thlm_61) * _drc1_61
                           + (_corr_ce1_60 * _seta1_60 - _schi1_60)
                             / (2.0 * _cthl1_safe) * _schi1_60 * _cf1_60)
            _thlprcp_c2 = ((_adg1_j25['thl_2'] - _thlm_61) * _drc2_61
                           + (_corr_ce2_60 * _seta2_60 - _schi2_60)
                             / (2.0 * _cthl2_safe) * _schi2_60 * _cf2_60)

            # uprcp_i = (u_i - um) * (rc_i - rcm)
            _uprcp_c1  = (_adg1_j25['u_1'] - _um_61) * _drc1_61
            _uprcp_c2  = (_adg1_j25['u_2'] - _um_61) * _drc2_61

            # vprcp_i = (v_i - vm) * (rc_i - rcm)
            _vprcp_c1  = (_adg1_j25['v_1'] - _vm_61) * _drc1_61
            _vprcp_c2  = (_adg1_j25['v_2'] - _vm_61) * _drc2_61

            # Mix with mixt_frac (on zt grid)
            _wprcp_zt_61  = _mf60 * _wprcp_c1  + (1.0 - _mf60) * _wprcp_c2
            _wp2rcp_zt_61 = _mf60 * _wp2rcp_c1 + (1.0 - _mf60) * _wp2rcp_c2
            _rtprcp_zt_61 = _mf60 * _rtprcp_c1 + (1.0 - _mf60) * _rtprcp_c2
            _thlprcp_zt_61= _mf60 * _thlprcp_c1+ (1.0 - _mf60) * _thlprcp_c2
            _uprcp_zt_61  = _mf60 * _uprcp_c1  + (1.0 - _mf60) * _uprcp_c2
            _vprcp_zt_61  = _mf60 * _vprcp_c1  + (1.0 - _mf60) * _vprcp_c2

            # Convert zt→zm; zero at k_ub_zm (Fortran lines 4233-4261)
            _k_ub61 = gr.k_ub_zm
            _wprcp_zm_61   = zt2zm_jax(_wprcp_zt_61,   gr).at[:, _k_ub61].set(0.0)
            _rtprcp_zm_61  = zt2zm_jax(_rtprcp_zt_61,  gr).at[:, _k_ub61].set(0.0)
            _thlprcp_zm_61 = zt2zm_jax(_thlprcp_zt_61, gr).at[:, _k_ub61].set(0.0)
            _uprcp_zm_61   = zt2zm_jax(_uprcp_zt_61,   gr).at[:, _k_ub61].set(0.0)
            _vprcp_zm_61   = zt2zm_jax(_vprcp_zt_61,   gr).at[:, _k_ub61].set(0.0)

            wprcp_out = _asarray(_wprcp_zm_61,   dtype=np.float64)
            _wp2rcp   = _asarray(_wp2rcp_zt_61,  dtype=np.float64)  # stays on zt
            _rtprcp   = _asarray(_rtprcp_zm_61,  dtype=np.float64)
            thlprcp   = _asarray(_thlprcp_zm_61, dtype=np.float64)
            uprcp     = _asarray(_uprcp_zm_61,   dtype=np.float64)
            vprcp     = _asarray(_vprcp_zm_61,   dtype=np.float64)

            # Iter27 shadow: calc_wp2xp_pdf for wp2rtp, wp2thlp, wp2up
            # For ADG1: corr_w_rt = corr_w_thl = corr_u_w = 0
            _zero27 = jnp.zeros_like(_adg1_j25['mixt_frac'])
            _wp2rtp_j27 = calc_wp2xp_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(rtm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['rt_1'], x_2=_adg1_j25['rt_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_rt_1'], varnce_x_2=_adg1_j25['varnce_rt_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wp2thlp_j27 = calc_wp2xp_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(thlm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['thl_1'], x_2=_adg1_j25['thl_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_thl_1'], varnce_x_2=_adg1_j25['varnce_thl_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wp2up_j27 = calc_wp2xp_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(um),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['u_1'], x_2=_adg1_j25['u_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_u_1'], varnce_x_2=_adg1_j25['varnce_u_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter28 shadow: calc_wpxp2_pdf for wpup2, wpvp2
            # For ADG1: corr_u_w = corr_v_w = 0
            _wpup2_j28 = calc_wpxp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(um),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['u_1'], x_2=_adg1_j25['u_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_u_1'], varnce_x_2=_adg1_j25['varnce_u_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wpvp2_j28 = calc_wpxp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(vm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['v_1'], x_2=_adg1_j25['v_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_v_1'], varnce_x_2=_adg1_j25['varnce_v_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter29 shadow: calc_wp2xp2_pdf for wp2up2, wp2vp2
            # Fortran computes on zt then converts to zm; we do the same.
            _wp2up2_zt_j29 = calc_wp2xp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(um),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['u_1'], x_2=_adg1_j25['u_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_u_1'], varnce_x_2=_adg1_j25['varnce_u_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wp2vp2_zt_j29 = calc_wp2xp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(vm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['v_1'], x_2=_adg1_j25['v_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_v_1'], varnce_x_2=_adg1_j25['varnce_v_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wp2up2_zm_j29 = zt2zm_jax(_wp2up2_zt_j29, gr)
            _wp2vp2_zm_j29 = zt2zm_jax(_wp2vp2_zt_j29, gr)
            # Fortran zeroes k_ub_zm after zt2zm_api (no zm_min for these)
            _k_ub29 = gr.k_ub_zm
            _wp2up2_zm_j29 = _wp2up2_zm_j29.at[:, _k_ub29].set(0.0)
            _wp2vp2_zm_j29 = _wp2vp2_zm_j29.at[:, _k_ub29].set(0.0)
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter30 shadow: calc_wp4_pdf
            # Fortran computes on zt then converts to zm.
            _wp4_zt_j30 = calc_wp4_pdf_jax(
                wm=jnp.asarray(wm_zt),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # zm_min=0.0 (zero_threshold) applied inside zt2zm_jax, then Fortran
            # zeroes both k_lb_zm and k_ub_zm explicitly.
            _wp4_zm_j30 = zt2zm_jax(_wp4_zt_j30, gr, zm_min=0.0)
            _k_lb30 = gr.k_lb_zm
            _k_ub30 = gr.k_ub_zm
            _wp4_zm_j30 = _wp4_zm_j30.at[:, _k_lb30].set(0.0)
            _wp4_zm_j30 = _wp4_zm_j30.at[:, _k_ub30].set(0.0)
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter31 shadow: wprtp2, wpthlp2 via calc_wpxp2_pdf
            # For ADG1: corr_w_rt = corr_w_thl = 0
            _wprtp2_j31 = calc_wpxp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(rtm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['rt_1'], x_2=_adg1_j25['rt_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_rt_1'], varnce_x_2=_adg1_j25['varnce_rt_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            _wpthlp2_j31 = calc_wpxp2_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(thlm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['thl_1'], x_2=_adg1_j25['thl_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_thl_1'], varnce_x_2=_adg1_j25['varnce_thl_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter32 shadow: wprtpthlp via calc_wpxpyp_pdf
            # For ADG1: corr_w_rt = corr_w_thl = 0; corr_rt_thl from iter26
            _wprtpthlp_j32 = calc_wpxpyp_pdf_jax(
                wm=jnp.asarray(wm_zt), xm=jnp.asarray(rtm), ym=jnp.asarray(thlm),
                w_1=_adg1_j25['w_1'], w_2=_adg1_j25['w_2'],
                x_1=_adg1_j25['rt_1'], x_2=_adg1_j25['rt_2'],
                y_1=_adg1_j25['thl_1'], y_2=_adg1_j25['thl_2'],
                varnce_w_1=_adg1_j25['varnce_w_1'], varnce_w_2=_adg1_j25['varnce_w_2'],
                varnce_x_1=_adg1_j25['varnce_rt_1'], varnce_x_2=_adg1_j25['varnce_rt_2'],
                varnce_y_1=_adg1_j25['varnce_thl_1'], varnce_y_2=_adg1_j25['varnce_thl_2'],
                corr_w_x_1=_zero27, corr_w_x_2=_zero27,   # corr_w_rt = 0 for ADG1
                corr_w_y_1=_zero27, corr_w_y_2=_zero27,   # corr_w_thl = 0 for ADG1
                corr_x_y_1=_corr_rt_thl_1_j26,
                corr_x_y_2=_corr_rt_thl_2_j26,
                mixt_frac=_adg1_j25['mixt_frac'],
            )
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Iter33 shadow: wpthvp, wp2thvp, rtpthvp, thlpthvp
            # Formula: wpthvp_zt = wpthlp_zt + ep1*thv_ds_zt*wprtp_zt + rc_coef_zt*wprcp_zt
            # rc_coef_zt = Lv/(exner*Cp) - ep2*thv_ds_zt
            # exner, thv_ds_zt, wp2thlp, wp2rtp, _wp2rcp are zt-grid;
            # rtpthlp, rtp2, thlp2, _rtprcp, thlprcp, wprcp_out are zm-grid → zm2zt
            _rc_coef_j33 = (Lv / (jnp.asarray(exner) * Cp)
                            - ep2 * jnp.asarray(thv_ds_zt))
            # Use the native zt-grid rc-flux moments from Iter61 (NOT a zt→zm→zt
            # round-trip of the zm output). Fortran (pdf_closure_module.F90:1130-1155)
            # computes wprcp(i,k) and wpthvp(i,k)=...+rc_coef*wprcp on the SAME pdf
            # grid in one pass. The round-trip smooths sharp cloud-top wprcp gradients
            # — negligible for ARM (small cloud frac) but a ~5e-4 error for thick-cloud
            # cases (DYCOMS from step 1, BOMEX at cloud onset).
            _wprcp_zt_j33 = _wprcp_zt_61
            _rtprcp_zt_j33 = _rtprcp_zt_61
            _thlprcp_zt_j33 = _thlprcp_zt_61
            _rtpthlp_zt_j33 = zm2zt_jax(jnp.asarray(rtpthlp), gr)
            _rtp2_zt_j33 = zm2zt_jax(jnp.asarray(rtp2), gr)
            _thlp2_zt_j33 = zm2zt_jax(jnp.asarray(thlp2), gr)
            _wpthvp_zt_j33 = (_wpthlp_zt_j25
                               + ep1 * jnp.asarray(thv_ds_zt) * _wprtp_zt_j25
                               + _rc_coef_j33 * _wprcp_zt_j33)
            _wp2thvp_zt_j33 = (_wp2thlp_j27
                                + ep1 * jnp.asarray(thv_ds_zt) * _wp2rtp_j27
                                + _rc_coef_j33 * jnp.asarray(_wp2rcp))
            _rtpthvp_zt_j33 = (_rtpthlp_zt_j33
                                + ep1 * jnp.asarray(thv_ds_zt) * _rtp2_zt_j33
                                + _rc_coef_j33 * _rtprcp_zt_j33)
            _thlpthvp_zt_j33 = (_thlp2_zt_j33
                                  + ep1 * jnp.asarray(thv_ds_zt) * _rtpthlp_zt_j33
                                  + _rc_coef_j33 * _thlprcp_zt_j33)
            # Convert to zm and zero k_ub_zm (wpthvp, rtpthvp, thlpthvp are zm; wp2thvp is zt)
            _wpthvp_zm_j33 = zt2zm_jax(_wpthvp_zt_j33, gr)
            _rtpthvp_zm_j33 = zt2zm_jax(_rtpthvp_zt_j33, gr)
            _thlpthvp_zm_j33 = zt2zm_jax(_thlpthvp_zt_j33, gr)
            _k_ub33 = gr.k_ub_zm
            _wpthvp_zm_j33 = _wpthvp_zm_j33.at[:, _k_ub33].set(0.0)
            _rtpthvp_zm_j33 = _rtpthvp_zm_j33.at[:, _k_ub33].set(0.0)
            _thlpthvp_zm_j33 = _thlpthvp_zm_j33.at[:, _k_ub33].set(0.0)
            # Iter50: Fortran oracle removed; JAX results are the state.

            # ============================================================ #
            # Iter34: Override pdf_closure_driver state with JAX values     #
            # Replaces Fortran pdf_closure_driver for ARM ADG1 dry case.    #
            # Uses already-computed iter25-33 shadow values.                #
            # _ssw_j34 already computed above (before _adg1_j25 call).     #
            # ============================================================ #
            # Iter50: Fortran oracle removed; JAX results are the state.

            # Override state with JAX-computed values.
            sigma_sqd_w = _ssw_j34
            wpthvp    = _asarray(_wpthvp_zm_j33)
            wp2thvp   = _asarray(_wp2thvp_zt_j33)
            rtpthvp   = _asarray(_rtpthvp_zm_j33)
            thlpthvp  = _asarray(_thlpthvp_zm_j33)
            # rc_coef_zm = zt2zm(rc_coef) with k_ub zeroed (pdf_closure_module.F90:4234).
            # Carried (post-advance placement) to the NEXT step's diagnose_upxp/upthvp
            # cloud term (rc_coef_zm*uprcp). Previously left stale at its zero init —
            # harmless for ARM (uprcp≈0) but wrong for cloudy cases (dycoms/bomex).
            rc_coef_zm = _asarray(
                zt2zm_jax(_rc_coef_j33, gr).at[:, gr.k_ub_zm].set(0.0),
                dtype=np.float64)
            wpup2     = _asarray(_wpup2_j28)
            wpvp2     = _asarray(_wpvp2_j28)
            wp2up2    = _asarray(_wp2up2_zm_j29)
            wp2vp2    = _asarray(_wp2vp2_zm_j29)
            wp4       = _asarray(_wp4_zm_j30)
            wp2rtp    = _asarray(_wp2rtp_j27)
            wp2thlp   = _asarray(_wp2thlp_j27)
            wp2up     = _asarray(_wp2up_j27)
            wprtp2    = _asarray(_wprtp2_j31)
            wpthlp2   = _asarray(_wpthlp2_j31)
            wprtpthlp = _asarray(_wprtpthlp_j32)
            # Update carry variables: Fortran's advance_clubb_core local wpthlp2/wprtp2/wprtpthlp
            # persist on the stack between calls for ipdf_post_advance_fields. JAX explicitly
            # carries them so next timestep's advance_xp2_xpyp sees the post-advance values.
            _wprtp2    = _asarray(_wprtp2_j31)
            _wpthlp2   = _asarray(_wpthlp2_j31)
            _wprtpthlp = _asarray(_wprtpthlp_j32)

            if l_sample and l_samp_stats and stats_writer is not None:
                # Iter68: pdf_closure_module.F90 stats for Block U ADG1 variables
                # Skw and sigma_sqd_w: written from inside Fortran pdf_closure using post-advance
                # state (Fortran pdf_closure_module.F90:4446, 4447, 4454, 4512).
                # _Skw_zm_j34 and _Skw_zt_j25 are post-advance Skw; _ssw_j34 is post-advance sigma_sqd_w.
                stats_writer.update("Skw_zm", _asarray(_Skw_zm_j34, dtype=np.float64))
                stats_writer.update("Skw_zt", _asarray(_Skw_zt_j25, dtype=np.float64))
                stats_writer.update("sigma_sqd_w", _ssw_j34)
                stats_writer.update("gamma_Skw_fnc", _gamma_j34)
                stats_writer.update("corr_rt_thl_1", _asarray(_corr_rt_thl_1_j26, dtype=np.float64))
                stats_writer.update("corr_rt_thl_2", _asarray(_corr_rt_thl_2_j26, dtype=np.float64))
                stats_writer.update("wp2rtp",    _asarray(_wp2rtp_j27,    dtype=np.float64))
                stats_writer.update("wp2thlp",   _asarray(_wp2thlp_j27,   dtype=np.float64))
                stats_writer.update("wp2up",     _asarray(_wp2up_j27,     dtype=np.float64))
                stats_writer.update("wpup2",     _asarray(_wpup2_j28,     dtype=np.float64))
                stats_writer.update("wpvp2",     _asarray(_wpvp2_j28,     dtype=np.float64))
                stats_writer.update("wp2up2",    _asarray(_wp2up2_zm_j29, dtype=np.float64))
                stats_writer.update("wp2vp2",    _asarray(_wp2vp2_zm_j29, dtype=np.float64))
                stats_writer.update("wp4",       _asarray(_wp4_zm_j30,    dtype=np.float64))
                stats_writer.update("wprtp2",    _asarray(_wprtp2_j31,    dtype=np.float64))
                stats_writer.update("wpthlp2",   _asarray(_wpthlp2_j31,   dtype=np.float64))
                stats_writer.update("wprtpthlp", _asarray(_wprtpthlp_j32, dtype=np.float64))
                stats_writer.update("wpthvp",    _asarray(wpthvp,         dtype=np.float64))
                stats_writer.update("wp2thvp",   _asarray(wp2thvp,        dtype=np.float64))
                stats_writer.update("rtpthvp",   _asarray(rtpthvp,        dtype=np.float64))
                stats_writer.update("thlpthvp",  _asarray(thlpthvp,       dtype=np.float64))
                # PDF cloud-water fluxes (computed in Block 61; Fortran writes these).
                # Outputting them lets the diagnostic compare verify the cloud-flux
                # physics directly instead of reading an unwritten 0 (see Iter97 note).
                stats_writer.update("wprcp",     _asarray(_wprcp_zm_61,   dtype=np.float64))
                stats_writer.update("rtprcp",    _asarray(_rtprcp_zm_61,  dtype=np.float64))
                stats_writer.update("thlprcp",   _asarray(_thlprcp_zm_61, dtype=np.float64))
                stats_writer.update("uprcp",     _asarray(_uprcp_zm_61,   dtype=np.float64))
                stats_writer.update("vprcp",     _asarray(_vprcp_zm_61,   dtype=np.float64))

                # Iter69: rc_coef and rc_coef_zm (Fortran pdf_closure_module.F90 line 4503-4519)
                stats_writer.update("rc_coef", _asarray(_rc_coef_j33, dtype=np.float64))
                _rc_coef_zm_69 = zt2zm_jax(_rc_coef_j33, gr)
                _rc_coef_zm_69 = _rc_coef_zm_69.at[:, gr.k_ub_zm].set(0.0)
                stats_writer.update("rc_coef_zm", _asarray(_rc_coef_zm_69, dtype=np.float64))

                # Iter69: Skrt, Skthl on zt and zm (Fortran pdf_closure_module.F90 line 4448-4451)
                _Skrt_zt_69 = skx_func_jax(jnp.asarray(_rtp2_zt_j25), jnp.asarray(rtp3),
                                            rt_tol, jnp.asarray(clubb_params))
                _Skthl_zt_69 = skx_func_jax(jnp.asarray(_thlp2_zt_j25), jnp.asarray(thlp3),
                                             thl_tol, jnp.asarray(clubb_params))
                _rtp3_zm_69  = zt2zm_jax(jnp.asarray(rtp3), gr)
                _thlp3_zm_69 = zt2zm_jax(jnp.asarray(thlp3), gr)
                _Skrt_zm_69  = skx_func_jax(jnp.asarray(rtp2), _rtp3_zm_69,
                                             rt_tol, jnp.asarray(clubb_params))
                _Skthl_zm_69 = skx_func_jax(jnp.asarray(thlp2), _thlp3_zm_69,
                                             thl_tol, jnp.asarray(clubb_params))
                stats_writer.update("Skrt_zt",  _asarray(_Skrt_zt_69,  dtype=np.float64))
                stats_writer.update("Skthl_zt", _asarray(_Skthl_zt_69, dtype=np.float64))
                stats_writer.update("Skrt_zm",  _asarray(_Skrt_zm_69,  dtype=np.float64))
                stats_writer.update("Skthl_zm", _asarray(_Skthl_zm_69, dtype=np.float64))

                # Iter69: Skw_velocity (zm-grid, Fortran pdf_closure_module.F90 line 4465)
                # Skw_velocity = (1/(1-sigma_sqd_w)) * wp3_zm / max(wp2, w_tol_sqd)
                _ssw_zm_69 = jnp.asarray(sigma_sqd_w)  # zm-grid (overridden at line 3332)
                _wp3_zm_69 = jnp.asarray(_wp3_zm_j34)   # already computed above
                _Skw_vel_69 = ((1.0 / (1.0 - _ssw_zm_69))
                               * (_wp3_zm_69 / jnp.maximum(jnp.asarray(wp2), w_tol_sqd)))
                stats_writer.update("Skw_velocity", _asarray(_Skw_vel_69, dtype=np.float64))

                # Iter69: rsatl_1/2, chi_1/2, crt/cthl, stdev_chi/eta,
                #         covar/corr_chi_eta, chi, chip2
                # (Fortran pdf_closure_module.F90:transform_pdf_chi_eta_component)
                _tl1_69 = _adg1_j25['thl_1'] * jnp.asarray(exner)  # liquid T comp 1
                _tl2_69 = _adg1_j25['thl_2'] * jnp.asarray(exner)  # liquid T comp 2
                _rsatl1_69 = sat_mixrat_liq_jax(
                    jnp.asarray(p_in_Pa), _tl1_69, flags.saturation_formula)
                _rsatl2_69 = sat_mixrat_liq_jax(
                    jnp.asarray(p_in_Pa), _tl2_69, flags.saturation_formula)
                stats_writer.update("rsatl_1", _asarray(_rsatl1_69, dtype=np.float64))
                stats_writer.update("rsatl_2", _asarray(_rsatl2_69, dtype=np.float64))

                def _chi_eta_comp(tl, rsatl, rt, varnce_thl, varnce_rt, corr_rt_thl):
                    """transform_pdf_chi_eta_component JAX (pdf_closure_module.F90)."""
                    _beta_c = ep * Lv**2 / (Rd * Cp * tl**2)
                    _inv_bp1 = 1.0 / (1.0 + _beta_c * rsatl)
                    chi_c = (rt - rsatl) * _inv_bp1
                    crt_c = _inv_bp1
                    cthl_c = ((1.0 + _beta_c * rt) * _inv_bp1**2
                              * (Cp / Lv) * _beta_c * rsatl * jnp.asarray(exner))
                    vrnc_rt_t = crt_c**2 * varnce_rt
                    vrnc_thl_t = cthl_c**2 * varnce_thl
                    covar_ce = vrnc_rt_t - vrnc_thl_t
                    corr_t = (2.0 * corr_rt_thl * crt_c * cthl_c
                              * jnp.sqrt(varnce_rt * varnce_thl))
                    vrnc_chi = vrnc_rt_t - corr_t + vrnc_thl_t
                    vrnc_eta = vrnc_rt_t + corr_t + vrnc_thl_t
                    stdev_chi_c = _safe_sqrt(vrnc_chi)
                    stdev_eta_c = _safe_sqrt(vrnc_eta)
                    denom = stdev_chi_c * stdev_eta_c
                    _chi_tol = 1.0e-8
                    _denom_thresh = _chi_tol ** 2
                    _smth = min(min_max_smth_mag, _denom_thresh)
                    tmp_d = smooth_max_jax(jnp.abs(covar_ce) / max_mag_correlation,
                                          denom, _smth)
                    tmp_d = smooth_max_jax(tmp_d, _denom_thresh, _smth)
                    corr_ce = covar_ce / tmp_d
                    return chi_c, crt_c, cthl_c, stdev_chi_c, stdev_eta_c, covar_ce, corr_ce

                (_chi1_69, _crt1_69, _cthl1_69,
                 _stdev_chi1_69, _stdev_eta1_69,
                 _covar_ce1_69, _corr_ce1_69) = _chi_eta_comp(
                    _tl1_69, _rsatl1_69,
                    _adg1_j25['rt_1'], _adg1_j25['varnce_thl_1'],
                    _adg1_j25['varnce_rt_1'], _corr_rt_thl_1_j26)
                (_chi2_69, _crt2_69, _cthl2_69,
                 _stdev_chi2_69, _stdev_eta2_69,
                 _covar_ce2_69, _corr_ce2_69) = _chi_eta_comp(
                    _tl2_69, _rsatl2_69,
                    _adg1_j25['rt_2'], _adg1_j25['varnce_thl_2'],
                    _adg1_j25['varnce_rt_2'], _corr_rt_thl_2_j26)

                stats_writer.update("chi_1",          _asarray(_chi1_69,       dtype=np.float64))
                stats_writer.update("chi_2",          _asarray(_chi2_69,       dtype=np.float64))
                stats_writer.update("crt_1",          _asarray(_crt1_69,       dtype=np.float64))
                stats_writer.update("crt_2",          _asarray(_crt2_69,       dtype=np.float64))
                stats_writer.update("cthl_1",         _asarray(_cthl1_69,      dtype=np.float64))
                stats_writer.update("cthl_2",         _asarray(_cthl2_69,      dtype=np.float64))
                stats_writer.update("stdev_chi_1",    _asarray(_stdev_chi1_69, dtype=np.float64))
                stats_writer.update("stdev_chi_2",    _asarray(_stdev_chi2_69, dtype=np.float64))
                stats_writer.update("stdev_eta_1",    _asarray(_stdev_eta1_69, dtype=np.float64))
                stats_writer.update("stdev_eta_2",    _asarray(_stdev_eta2_69, dtype=np.float64))
                stats_writer.update("covar_chi_eta_1",_asarray(_covar_ce1_69,  dtype=np.float64))
                stats_writer.update("covar_chi_eta_2",_asarray(_covar_ce2_69,  dtype=np.float64))
                stats_writer.update("corr_chi_eta_1", _asarray(_corr_ce1_69,   dtype=np.float64))
                stats_writer.update("corr_chi_eta_2", _asarray(_corr_ce2_69,   dtype=np.float64))

                # chi = mixt_frac * chi_1 + (1-mixt_frac) * chi_2
                _mf_69 = _adg1_j25['mixt_frac']
                _chi_69 = _mf_69 * _chi1_69 + (1.0 - _mf_69) * _chi2_69
                stats_writer.update("chi", _asarray(_chi_69, dtype=np.float64))

                # chip2 = Var(chi) = mf*(chi1-chi)^2 + (1-mf)*(chi2-chi)^2
                #                  + mf*stdev_chi_1^2 + (1-mf)*stdev_chi_2^2
                _chip2_69 = (_mf_69 * (_chi1_69 - _chi_69)**2
                             + (1.0 - _mf_69) * (_chi2_69 - _chi_69)**2
                             + _mf_69 * _stdev_chi1_69**2
                             + (1.0 - _mf_69) * _stdev_chi2_69**2)
                stats_writer.update("chip2", _asarray(_chip2_69, dtype=np.float64))


    # ================================================================== #
    # Block V: Stats — accumulate and finalize budgets
    # ================================================================== #
    if l_sample and stats_writer is not None:
        _stats_accumulate_py(
            stats_writer,
            gr=gr,
            nzm=nzm, nzt=nzt, ngrdcol=ngrdcol,
            sclr_dim=sclr_dim, edsclr_dim=edsclr_dim,
            dt=dt,
            l_implemented=l_implemented,
            l_host_applies_sfc_fluxes=flags.l_host_applies_sfc_fluxes,
            l_stability_correct_tau_zm=flags.l_stability_correct_tau_zm,
            um=um, vm=vm,
            upwp=upwp, vpwp=vpwp,
            up2=up2, vp2=vp2,
            thlm=thlm, rtm=rtm,
            thlm_before=thlm_before, rtm_before=rtm_before,
            thlm_forcing=thlm_forcing, rtm_forcing=rtm_forcing,
            wpthlp_sfc=wpthlp_sfc, wprtp_sfc=wprtp_sfc,
            wprtp=wprtp, wpthlp=wpthlp,
            wp2=wp2, wp3=wp3,
            rtp2=rtp2, rtp3=rtp3,
            thlp2=thlp2, thlp3=thlp3,
            rtpthlp=rtpthlp,
            p_in_Pa=p_in_Pa, exner=exner,
            rho=rho, rho_zm=rho_zm,
            rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
            thv_ds_zm=thv_ds_zm, thv_ds_zt=thv_ds_zt,
            wm_zt=wm_zt, wm_zm=wm_zm,
            rcm=rcm,
            cloud_frac=cloud_frac,
            ice_supersat_frac=ice_supersat_frac,
            thvm=thvm,
            ug=ug, vg=vg,
            ddzt_umvm_sqd=ddzt_umvm_sqd,
            stability_correction=stability_correction,
            Kh_zt=Kh_zt,
            rsat=rsat,
            Kh_zm=Kh_zm,
            em=em,
            wp3_on_wp2=wp3_on_wp2,
            wp3_on_wp2_zt=wp3_on_wp2_zt,
            sclrm=sclrm,
            sclrp2=sclrp2,
            sclrprtp=sclrprtp,
            sclrpthlp=sclrpthlp,
            sclrm_forcing=sclrm_forcing,
            wpsclrp=wpsclrp,
            wpedsclrp=wpedsclrp,
            edsclrm=edsclrm,
            edsclrm_forcing=edsclrm_forcing,
            saturation_formula=flags.saturation_formula,
        )

        stats_writer.finalize_budget("wp2_bt", wp2 / dt)
        stats_writer.finalize_budget("vp2_bt", vp2 / dt)
        stats_writer.finalize_budget("up2_bt", up2 / dt)
        stats_writer.finalize_budget("wprtp_bt", wprtp / dt)
        stats_writer.finalize_budget("wpthlp_bt", wpthlp / dt)
        if flags.l_predict_upwp_vpwp:
            stats_writer.finalize_budget("upwp_bt", upwp / dt)
            stats_writer.finalize_budget("vpwp_bt", vpwp / dt)
        stats_writer.finalize_budget("rtp2_bt", rtp2 / dt)
        stats_writer.finalize_budget("thlp2_bt", thlp2 / dt)
        stats_writer.finalize_budget("rtpthlp_bt", rtpthlp / dt)
        stats_writer.finalize_budget("rtm_bt", rtm / dt)
        stats_writer.finalize_budget("thlm_bt", thlm / dt)
        stats_writer.finalize_budget("um_bt", um / dt)
        stats_writer.finalize_budget("vm_bt", vm / dt)
        stats_writer.finalize_budget("wp3_bt", wp3 / dt)

    if debug_level >= 2:
        err_info = parameterization_check_jax(
            err_info=err_info,
            nzm=nzm, nzt=nzt, ngrdcol=ngrdcol, sclr_dim=sclr_dim, edsclr_dim=edsclr_dim,
            thlm_forcing=thlm_forcing, rtm_forcing=rtm_forcing,
            um_forcing=um_forcing, vm_forcing=vm_forcing,
            wm_zm=wm_zm, wm_zt=wm_zt, p_in_Pa=p_in_Pa,
            rho_zm=rho_zm, rho=rho, exner=exner,
            rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
            invrs_rho_ds_zm=invrs_rho_ds_zm, invrs_rho_ds_zt=invrs_rho_ds_zt,
            thv_ds_zm=thv_ds_zm, thv_ds_zt=thv_ds_zt,
            wpthlp_sfc=wpthlp_sfc, wprtp_sfc=wprtp_sfc,
            upwp_sfc=upwp_sfc, vpwp_sfc=vpwp_sfc, p_sfc=p_sfc,
            um=um, upwp=upwp, vm=vm, vpwp=vpwp,
            up2=up2, vp2=vp2, rtm=rtm, wprtp=wprtp,
            thlm=thlm, wpthlp=wpthlp, wp2=wp2, wp3=wp3,
            rtp2=rtp2, thlp2=thlp2, rtpthlp=rtpthlp,
            prefix="end of ", wpsclrp_sfc=wpsclrp_sfc, wpedsclrp_sfc=wpedsclrp_sfc,
            sclrm=sclrm, wpsclrp=wpsclrp, sclrp2=sclrp2,
            sclrprtp=sclrprtp, sclrpthlp=sclrpthlp,
            sclrm_forcing=sclrm_forcing, edsclrm=edsclrm,
            edsclrm_forcing=edsclrm_forcing,
        )
        err_code = err_info.err_code
        if err_code is not None and np.any(_asarray(err_code) == CLUBB_FATAL_ERROR):
            return

    # ================================================================== #
    return (
        um, vm, up3, vp3, thlm, rtm, rtp3, thlp3, wp3,
        upwp, vpwp, up2, vp2, wprtp, wpthlp, rtp2, thlp2, rtpthlp, wp2,
        sclrm, sclrp3, wpsclrp, sclrp2, sclrprtp, sclrpthlp,
        p_in_Pa, exner, rcm, cloud_frac, wp2thvp, wp2up,
        wpthvp, rtpthvp, thlpthvp, sclrpthvp,
        wp2rtp, wp2thlp, wpup2, wpvp2, ice_supersat_frac,
        uprcp, vprcp, rc_coef_zm, wp4, wp2up2, wp2vp2,
        um_pert, vm_pert, upwp_pert, vpwp_pert,
        edsclrm,
        pdf_params, pdf_params_zm,
        pdf_implicit_coefs_terms, err_info,
        rcm_in_layer, cloud_cover, w_up_in_cloud, w_down_in_cloud,
        cloudy_updraft_frac, cloudy_downdraft_frac, wprcp_out, invrs_tau_zm,
        Kh_zt, Kh_zm, thlprcp, Lscale,
        _sigma_sqd_w,
        _rc_coef, _rcp2_zt, _wprtp2,
        _wpthlp2, _wprtpthlp, _wp2rcp,
        _rtprcp, _rcp2, _skw_velocity,
        _cloud_frac_zm, _ice_supersat_frac_zm,
        _rtm_zm, _thlm_zm, _rcm_zm,
        _rcm_supersat_adj, _wp2sclrp,
        _wpsclrp2, _sclrprcp,
        _wpsclrprtp, _wpsclrpthlp,
    )
