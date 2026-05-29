"""Pure-Python port of CLUBB_core/model_flags.F90.

Provides get_default_config_flags_jax to replace the Fortran API call.
"""
from __future__ import annotations

from clubb_python.derived_types.config_flags import ConfigFlags


def get_default_config_flags_jax() -> ConfigFlags:
    """Return CLUBB default ConfigFlags matching set_default_clubb_config_flags_api.

    Values are faithfully ported from model_flags.F90 lines 576-639.
    Integer constants:
      iiPDF_ADG1=1, ipdf_post_advance_fields=2, penta_lu=2, tridiag_lu=2,
      saturation_flatau=3, ppm_remap=2, no_grid_adaptation=0, sliding_window=2
    """
    return ConfigFlags(
        # ── Integer flags ──────────────────────────────────────────────────
        iiPDF_type=1,                   # iiPDF_ADG1
        ipdf_call_placement=2,          # ipdf_post_advance_fields
        penta_solve_method=2,           # penta_lu
        tridiag_solve_method=2,         # tridiag_lu
        saturation_formula=3,           # saturation_flatau
        grid_remap_method=2,            # ppm_remap
        grid_adapt_in_time_method=0,    # no_grid_adaptation
        fill_holes_type=2,              # sliding_window
        # ── Logical flags ──────────────────────────────────────────────────
        l_use_precip_frac=True,
        l_predict_upwp_vpwp=True,
        l_ho_nontrad_coriolis=False,
        l_ho_trad_coriolis=False,
        l_min_wp2_from_corr_wx=False,
        l_min_xp2_from_corr_wx=True,
        l_C2_cloud_frac=False,
        l_diffuse_rtm_and_thlm=False,
        l_stability_correct_Kh_N2_zm=False,
        l_calc_thlp2_rad=True,
        l_upwind_xpyp_ta=True,
        l_upwind_xm_ma=True,
        l_uv_nudge=False,
        l_rtm_nudge=False,
        l_tke_aniso=True,
        l_vert_avg_closure=False,
        l_trapezoidal_rule_zt=False,
        l_trapezoidal_rule_zm=False,
        l_call_pdf_closure_twice=False,
        l_standard_term_ta=False,
        l_partial_upwind_wp3=False,
        l_godunov_upwind_wpxp_ta=False,
        l_godunov_upwind_xpyp_ta=False,
        l_use_cloud_cover=False,
        l_diagnose_correlations=False,
        l_calc_w_corr=False,
        l_const_Nc_in_cloud=False,
        l_fix_w_chi_eta_correlations=True,
        l_stability_correct_tau_zm=False,
        l_damp_wp2_using_em=True,
        l_do_expldiff_rtm_thlm=False,
        l_Lscale_plume_centered=False,
        l_diag_Lscale_from_tau=True,
        l_use_C7_Richardson=True,
        l_use_C11_Richardson=False,
        l_use_shear_Richardson=False,
        l_brunt_vaisala_freq_moist=False,
        l_use_thvm_in_bv_freq=False,
        l_rcm_supersat_adj=True,
        l_damp_wp3_Skw_squared=True,
        l_prescribed_avg_deltaz=False,
        l_lmm_stepping=False,
        l_e3sm_config=False,
        l_vary_convect_depth=False,
        l_use_tke_in_wp3_pr_turb_term=True,
        l_use_tke_in_wp2_wp3_K_dfsn=False,
        l_use_wp3_lim_with_smth_Heaviside=False,
        l_smooth_Heaviside_tau_wpxp=False,
        l_modify_limiters_for_cnvg_test=False,
        l_enable_relaxed_clipping=False,
        l_linearize_pbl_winds=False,
        l_mono_flux_lim_thlm=True,
        l_mono_flux_lim_rtm=True,
        l_mono_flux_lim_um=True,
        l_mono_flux_lim_vm=True,
        l_mono_flux_lim_spikefix=True,
        l_host_applies_sfc_fluxes=False,
        l_wp2_fill_holes_tke=True,
        l_add_dycore_grid=False,
    )
