"""Pure-Python port of CLUBB_core/numerical_check.F90 and related validation.

Provides:
  parameterization_check_jax  — replaces clubb_api.parameterization_check
  check_clubb_settings_jax    — replaces clubb_api.check_clubb_settings
  check_parameters_jax        — replaces clubb_api.check_parameters
"""
from __future__ import annotations

import sys

import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg  # REFACTOR B5: detect jax.grad trace

CLUBB_NO_ERROR    = 0
CLUBB_FATAL_ERROR = 99

# ── model_flags constants (compile-time in Fortran, fixed here) ──────────────
_iiPDF_ADG1            = 1
_iiPDF_ADG2            = 2
_iiPDF_3D_Luhar        = 3
_iiPDF_new             = 4
_iiPDF_TSDADG          = 5
_iiPDF_LY93            = 6
_iiPDF_new_hybrid      = 7
_ipdf_pre_advance      = 1
_ipdf_post_advance     = 2
_ipdf_pre_post_advance = 3
_sat_bolton            = 1
_sat_gfdl              = 2
_sat_flatau            = 3
_sat_lookup            = 4
_order_xm_wpxp         = 1   # module constants from model_flags.F90 lines 22-25
_order_xp2_xpyp        = 2
_order_wp2_wp3         = 3
_order_windm           = 4
_l_explicit_turb_adv   = False  # model_flags.F90 line 127

# ── Parameter name → 0-based index map (matches PARAM_NAMES in parameters_tunable.py) ─
_PNAME_IDX: dict[str, int] = {
    "C1": 0, "C1b": 1, "C1c": 2,
    "C2rt": 3, "C2thl": 4, "C2rtthl": 5,
    "C4": 6, "C_uu_shr": 7, "C_uu_buoy": 8,
    "C6rt": 9, "C6rtb": 10, "C6rtc": 11,
    "C6thl": 12, "C6thlb": 13, "C6thlc": 14,
    "C7": 15, "C7b": 16, "C7c": 17,
    "C8": 18, "C8b": 19,
    "C10": 20,
    "C11": 21, "C11b": 22, "C11c": 23,
    "C12": 24, "C13": 25, "C14": 26,
    "C_wp2_pr_dfsn": 27, "C_wp3_pr_tp": 28, "C_wp3_pr_turb": 29,
    "C_wp3_pr_dfsn": 30, "C_wp2_splat": 31,
    "C6rt_Lscale0": 32, "C6thl_Lscale0": 33, "C7_Lscale0": 34,
    "wpxp_L_thresh": 35,
    "c_K": 36, "c_K1": 37, "nu1": 38,
    "c_K2": 39, "nu2": 40,
    "c_K6": 41, "nu6": 42,
    "c_K8": 43, "nu8": 44,
    "c_K9": 45, "nu9": 46, "nu10": 47,
    "c_K_hm": 48, "c_K_hmb": 49, "K_hm_min_coef": 50, "nu_hm": 51,
    "slope_coef_spread_DG_means_w": 52, "pdf_component_stdev_factor_w": 53,
    "coef_spread_DG_means_rt": 54, "coef_spread_DG_means_thl": 55,
    "gamma_coef": 56, "gamma_coefb": 57, "gamma_coefc": 58,
    "mu": 59, "beta": 60, "lmin_coef": 61,
    "omicron": 62, "zeta_vrnce_rat": 63, "upsilon_precip_frac_rat": 64,
    "lambda0_stability_coef": 65, "mult_coef": 66,
    "taumin": 67, "taumax": 68,
    "Lscale_mu_coef": 69, "Lscale_pert_coef": 70,
    "alpha_corr": 71, "Skw_denom_coef": 72,
    "c_K10": 73, "c_K10h": 74,
    "thlp2_rad_coef": 75, "thlp2_rad_cloud_frac_thresh": 76,
    "up2_sfc_coef": 77,
    "Skw_max_mag": 78,
    "C_invrs_tau_bkgnd": 79, "C_invrs_tau_sfc": 80, "C_invrs_tau_shear": 81,
    "C_invrs_tau_N2": 82, "C_invrs_tau_N2_wp2": 83, "C_invrs_tau_N2_xp2": 84,
    "C_invrs_tau_N2_wpxp": 85, "C_invrs_tau_N2_clear_wp3": 86,
    "C_invrs_tau_wpxp_Ri": 87, "C_invrs_tau_wpxp_N2_thresh": 88,
    "xp3_coef_base": 89, "xp3_coef_slope": 90,
    "altitude_threshold": 91, "rtp2_clip_coef": 92,
    "Cx_min": 93, "Cx_max": 94,
    "Richardson_num_min": 95, "Richardson_num_max": 96,
    "a3_coef_min": 97, "a_const": 98, "bv_efold": 99,
    "wpxp_Ri_exp": 100, "z_displace": 101,
}

_EPS64 = np.finfo(np.float64).eps  # machine epsilon for float64 ≈ 2.22e-16


def _check_nan(arr: np.ndarray, name: str, location: str, err_code: np.ndarray) -> None:
    """Set err_code to CLUBB_FATAL_ERROR if any element of arr is NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        print(f"{name} is NaN in {location}", file=sys.stderr)
        err_code[:] = CLUBB_FATAL_ERROR


def _check_negative(arr: np.ndarray, name: str, location: str, err_code: np.ndarray) -> None:
    """Set err_code to CLUBB_FATAL_ERROR if any element of arr is < 0."""
    if np.any(arr < 0.0):
        print(f"{name} < 0 in {location}", file=sys.stderr)
        err_code[:] = CLUBB_FATAL_ERROR


def parameterization_check_jax(
    err_info,
    nzm: int, nzt: int, ngrdcol: int, sclr_dim: int, edsclr_dim: int,
    thlm_forcing, rtm_forcing, um_forcing, vm_forcing,
    wm_zm, wm_zt, p_in_Pa,
    rho_zm, rho, exner, rho_ds_zm, rho_ds_zt,
    invrs_rho_ds_zm, invrs_rho_ds_zt,
    thv_ds_zm, thv_ds_zt,
    wpthlp_sfc, wprtp_sfc, upwp_sfc, vpwp_sfc, p_sfc,
    um, upwp, vm, vpwp, up2, vp2,
    rtm, wprtp, thlm, wpthlp, wp2, wp3,
    rtp2, thlp2, rtpthlp,
    prefix: str,
    wpsclrp_sfc, wpedsclrp_sfc,
    sclrm, wpsclrp, sclrp2, sclrprtp, sclrpthlp, sclrm_forcing,
    edsclrm, edsclrm_forcing,
):
    """Port of numerical_check.F90:parameterization_check.

    Checks all input arrays for NaN/Inf and selected arrays for negative values.
    Mirrors the Fortran logic exactly:
      1. NaN check all arrays → fatal error if any non-finite found
      2. Return early if fatal error
      3. Negativity check selected arrays → fatal error if any < 0
      4. If prefix=="beginning of " and negative found: clear error (host model)
      5. Check thlm < 190K in bottom 10 levels (warning only, no error)

    Returns a new err_info (or err_info._replace) with updated err_code.
    """
    proc_name = "advance_clubb_core"
    location = prefix + proc_name

    # Differentiability (REFACTOR B5): this is a pure NaN/Inf/negativity diagnostic — under a jax.grad
    # trace the fields are tracers, NaN-checking a tracer is meaningless, and np.asarray() on one errors.
    # Skip the diagnostic (return err_info unchanged) when tracing; concrete runs are unaffected.
    if _is_tracer_arg([thlm, rtm, um, vm, wp2, wp3, thlm_forcing, rtm_forcing]):
        return err_info

    # Start fresh: CLUBB resets err_code before each step
    err_code = np.zeros(ngrdcol, dtype=np.int32)

    # ── 1. NaN checks (all arrays) ────────────────────────────────────────────
    _check_nan(np.asarray(thlm_forcing),    "thlm_forcing",    location, err_code)
    _check_nan(np.asarray(rtm_forcing),     "rtm_forcing",     location, err_code)
    _check_nan(np.asarray(um_forcing),      "um_forcing",      location, err_code)
    _check_nan(np.asarray(vm_forcing),      "vm_forcing",      location, err_code)

    _check_nan(np.asarray(wm_zm),           "wm_zm",           location, err_code)
    _check_nan(np.asarray(wm_zt),           "wm_zt",           location, err_code)
    _check_nan(np.asarray(p_in_Pa),         "p_in_Pa",         location, err_code)
    _check_nan(np.asarray(rho_zm),          "rho_zm",          location, err_code)
    _check_nan(np.asarray(rho),             "rho",             location, err_code)
    _check_nan(np.asarray(exner),           "exner",           location, err_code)
    _check_nan(np.asarray(rho_ds_zm),       "rho_ds_zm",       location, err_code)
    _check_nan(np.asarray(rho_ds_zt),       "rho_ds_zt",       location, err_code)
    _check_nan(np.asarray(invrs_rho_ds_zm), "invrs_rho_ds_zm", location, err_code)
    _check_nan(np.asarray(invrs_rho_ds_zt), "invrs_rho_ds_zt", location, err_code)
    _check_nan(np.asarray(thv_ds_zm),       "thv_ds_zm",       location, err_code)
    _check_nan(np.asarray(thv_ds_zt),       "thv_ds_zt",       location, err_code)

    _check_nan(np.asarray(um),      "um",      location, err_code)
    _check_nan(np.asarray(upwp),    "upwp",    location, err_code)
    _check_nan(np.asarray(vm),      "vm",      location, err_code)
    _check_nan(np.asarray(vpwp),    "vpwp",    location, err_code)
    _check_nan(np.asarray(up2),     "up2",     location, err_code)
    _check_nan(np.asarray(vp2),     "vp2",     location, err_code)
    _check_nan(np.asarray(rtm),     "rtm",     location, err_code)
    _check_nan(np.asarray(wprtp),   "wprtp",   location, err_code)
    _check_nan(np.asarray(thlm),    "thlm",    location, err_code)
    _check_nan(np.asarray(wpthlp),  "wpthlp",  location, err_code)
    _check_nan(np.asarray(wp2),     "wp2",     location, err_code)
    _check_nan(np.asarray(wp3),     "wp3",     location, err_code)
    _check_nan(np.asarray(rtp2),    "rtp2",    location, err_code)
    _check_nan(np.asarray(thlp2),   "thlp2",   location, err_code)
    _check_nan(np.asarray(rtpthlp), "rtpthlp", location, err_code)

    _check_nan(np.asarray(wpthlp_sfc), "wpthlp_sfc", location, err_code)
    _check_nan(np.asarray(wprtp_sfc),  "wprtp_sfc",  location, err_code)
    _check_nan(np.asarray(upwp_sfc),   "upwp_sfc",   location, err_code)
    _check_nan(np.asarray(vpwp_sfc),   "vpwp_sfc",   location, err_code)
    _check_nan(np.asarray(p_sfc),      "p_sfc",      location, err_code)

    sclrm_arr         = np.asarray(sclrm)
    wpsclrp_sfc_arr   = np.asarray(wpsclrp_sfc)
    wpsclrp_arr       = np.asarray(wpsclrp)
    sclrp2_arr        = np.asarray(sclrp2)
    sclrprtp_arr      = np.asarray(sclrprtp)
    sclrpthlp_arr     = np.asarray(sclrpthlp)
    sclrm_forcing_arr = np.asarray(sclrm_forcing)
    for s in range(sclr_dim):
        _check_nan(sclrm_forcing_arr[..., s], "sclrm_forcing", location, err_code)
        _check_nan(wpsclrp_sfc_arr[:, s],     "wpsclrp_sfc",   location, err_code)
        _check_nan(sclrm_arr[..., s],          "sclrm",         location, err_code)
        _check_nan(wpsclrp_arr[..., s],        "wpsclrp",       location, err_code)
        _check_nan(sclrp2_arr[..., s],         "sclrp2",        location, err_code)
        _check_nan(sclrprtp_arr[..., s],       "sclrprtp",      location, err_code)
        _check_nan(sclrpthlp_arr[..., s],      "sclrpthlp",     location, err_code)

    edsclrm_arr         = np.asarray(edsclrm)
    edsclrm_forcing_arr = np.asarray(edsclrm_forcing)
    wpedsclrp_sfc_arr   = np.asarray(wpedsclrp_sfc)
    for e in range(edsclr_dim):
        _check_nan(edsclrm_forcing_arr[..., e], "edsclrm_forcing", location, err_code)
        _check_nan(wpedsclrp_sfc_arr[:, e],     "wpedsclrp_sfc",   location, err_code)
        _check_nan(edsclrm_arr[..., e],          "edsclrm",         location, err_code)

    # ── 2. Return early if NaN fatal error found ──────────────────────────────
    if np.any(err_code == CLUBB_FATAL_ERROR):
        return err_info._replace(err_code=err_code)

    # ── 3. Negativity checks ──────────────────────────────────────────────────
    _check_negative(np.asarray(rtm),           "rtm",           location, err_code)
    _check_negative(np.asarray(p_in_Pa),       "p_in_Pa",       location, err_code)
    _check_negative(np.asarray(rho),           "rho",           location, err_code)
    _check_negative(np.asarray(rho_zm),        "rho_zm",        location, err_code)
    _check_negative(np.asarray(exner),         "exner",         location, err_code)
    _check_negative(np.asarray(rho_ds_zm),     "rho_ds_zm",     location, err_code)
    _check_negative(np.asarray(rho_ds_zt),     "rho_ds_zt",     location, err_code)
    _check_negative(np.asarray(invrs_rho_ds_zm), "invrs_rho_ds_zm", location, err_code)
    _check_negative(np.asarray(invrs_rho_ds_zt), "invrs_rho_ds_zt", location, err_code)
    _check_negative(np.asarray(thv_ds_zm),     "thv_ds_zm",     location, err_code)
    _check_negative(np.asarray(thv_ds_zt),     "thv_ds_zt",     location, err_code)
    _check_negative(np.asarray(up2),           "up2",           location, err_code)
    _check_negative(np.asarray(vp2),           "vp2",           location, err_code)
    _check_negative(np.asarray(wp2),           "wp2",           location, err_code)
    _check_negative(np.asarray(thlm),          "thlm",          location, err_code)
    _check_negative(np.asarray(rtp2),          "rtp2",          location, err_code)
    _check_negative(np.asarray(thlp2),         "thlp2",         location, err_code)

    # ── 4. Clear error if prefix=="beginning of " (host model generated it) ───
    if prefix == "beginning of " and np.any(err_code == CLUBB_FATAL_ERROR):
        err_code[:] = CLUBB_NO_ERROR

    # ── 5. Check thlm < 190K in bottom 10 levels (warning only) ──────────────
    thlm_arr = np.asarray(thlm)
    for i in range(ngrdcol):
        for k in range(min(10, thlm_arr.shape[1])):
            if thlm_arr[i, k] < 190.0:
                print(
                    f"Liquid water potential temperature (thlm) < 190K "
                    f"at grid column i = {i+1} and grid level k = {k+1}: "
                    f"thlm({i+1},{k+1}) = {thlm_arr[i,k]}",
                    file=sys.stderr,
                )

    return err_info._replace(err_code=err_code)


# ── check_clubb_settings_jax ─────────────────────────────────────────────────

def check_clubb_settings_jax(
    ngrdcol: int,
    params: np.ndarray,
    config_flags,
    err_info,
    l_implemented: bool = False,
    l_input_fields: bool = False,
):
    """Port of numerical_check.F90:check_clubb_settings_api.

    Validates configuration flags and parameter consistency.
    Mirrors the Fortran logic exactly: fatal errors set err_code and return;
    warnings only print to stderr.
    Returns updated err_info.
    """
    err_code = np.zeros(ngrdcol, dtype=np.int32)

    def _fatal(msg: str) -> None:
        print(msg, file=sys.stderr)
        print("Fatal error in check_clubb_settings_api", file=sys.stderr)
        err_code[:] = CLUBB_FATAL_ERROR

    # ── 1. l_damp_wp2_using_em: requires C1==C14 and not l_stability_correct_tau_zm ──
    if config_flags.l_damp_wp2_using_em:
        C1  = params[:, _PNAME_IDX["C1"]]
        C14 = params[:, _PNAME_IDX["C14"]]
        denom = np.abs(C1 + C14) / 2.0
        not_equal = np.any(np.abs(C1 - C14) > np.where(denom > 0, denom * _EPS64, _EPS64))
        if not_equal or config_flags.l_stability_correct_tau_zm:
            _fatal("l_damp_wp2_using_em = T requires C1=C14 and l_stability_correct_tau_zm = F")
            print(f"C1 = {C1}", file=sys.stderr)
            print(f"C14 = {C14}", file=sys.stderr)
            print(f"l_stability_correct_tau_zm = {config_flags.l_stability_correct_tau_zm}", file=sys.stderr)
            return err_info._replace(err_code=err_code)

    # ── 2. saturation_formula must be in [1,4] ────────────────────────────────
    sf = config_flags.saturation_formula
    if sf not in (_sat_bolton, _sat_gfdl, _sat_flatau, _sat_lookup):
        _fatal(f"Unknown approx. of saturation vapor pressure: {sf}")
        return err_info._replace(err_code=err_code)

    # ── 3. iiPDF_type must be in [ADG1..new_hybrid] ───────────────────────────
    pdf = config_flags.iiPDF_type
    if pdf < _iiPDF_ADG1 or pdf > _iiPDF_new_hybrid:
        _fatal(f"Unknown type of double Gaussian PDF selected: {pdf}")
        print(f"iiPDF_type = {pdf}", file=sys.stderr)
        return err_info._replace(err_code=err_code)

    # ── 4. PDFs that require l_input_fields ──────────────────────────────────
    _input_only_pdfs = {
        _iiPDF_ADG2:       "The ADG2 PDF",
        _iiPDF_3D_Luhar:   "The 3D Luhar PDF",
        _iiPDF_new:        "The new PDF",
        _iiPDF_TSDADG:     "The new TSDADG PDF",
        _iiPDF_LY93:       "The Lewellen and Yoh PDF",
    }
    if pdf in _input_only_pdfs and not l_input_fields:
        name = _input_only_pdfs[pdf]
        _fatal(f"{name} can only be used with input fields (l_input_fields = .true.)")
        print(f"iiPDF_type = {pdf}, l_input_fields = {l_input_fields}", file=sys.stderr)
        return err_info._replace(err_code=err_code)

    # ── 5. ipdf_call_placement must be in [1,3] ───────────────────────────────
    icp = config_flags.ipdf_call_placement
    if icp < _ipdf_pre_advance or icp > _ipdf_pre_post_advance:
        _fatal(f"Invalid option selected for ipdf_call_placement: {icp}")
        return err_info._replace(err_code=err_code)

    # ── 6. l_predict_upwp_vpwp constraints ───────────────────────────────────
    if config_flags.l_predict_upwp_vpwp:
        if _l_explicit_turb_adv:
            _fatal("The l_explicit_turbulent_adv_wpxp option is not set up for l_predict_upwp_vpwp")
            return err_info._replace(err_code=err_code)
        if pdf not in (_iiPDF_ADG1, _iiPDF_new_hybrid):
            _fatal("Only ADG1 and new_hybrid PDFs are set up for l_predict_upwp_vpwp")
            return err_info._replace(err_code=err_code)

    # ── 7. l_min_xp2_from_corr_wx XOR l_enable_relaxed_clipping ─────────────
    xp2  = config_flags.l_min_xp2_from_corr_wx
    relx = config_flags.l_enable_relaxed_clipping
    if xp2 and relx:
        _fatal("Invalid: l_min_xp2_from_corr_wx = T and l_enable_relaxed_clipping = T (must be opposite)")
        return err_info._replace(err_code=err_code)
    elif not xp2 and not relx:
        # Warning-only in Fortran (fatal path was commented out)
        print("WARNING: l_min_xp2_from_corr_wx = F and l_enable_relaxed_clipping = F (should be opposite)", file=sys.stderr)

    # ── 8. order_ constants validation (1-4, all distinct) ───────────────────
    orders = {
        "order_xm_wpxp": _order_xm_wpxp,
        "order_wp2_wp3": _order_wp2_wp3,
        "order_xp2_xpyp": _order_xp2_xpyp,
        "order_windm": _order_windm,
    }
    for oname, val in orders.items():
        if val < 1 or val > 4:
            _fatal(f"The variable {oname} must have a value between 1 and 4 (= {val})")
            return err_info._replace(err_code=err_code)
    order_vals = list(orders.values())
    if len(set(order_vals)) != len(order_vals):
        _fatal(f"order_ variables must all be unique: {orders}")
        return err_info._replace(err_code=err_code)

    # ── 9. l_diag_Lscale_from_tau: Cx params must all equal 1 (warnings only) ─
    if config_flags.l_diag_Lscale_from_tau:
        _tau_params = ["C1", "C1b", "C2rt", "C2thl", "C2rtthl",
                       "C6rt", "C6rtb", "C6thl", "C6thlb", "C14"]
        for pname in _tau_params:
            col = params[:, _PNAME_IDX[pname]]
            if np.any(col != 1.0):
                print(
                    f"When the l_diag_Lscale_from_tau flag is enabled, "
                    f"{pname} must have a value of 1.",
                    file=sys.stderr,
                )
                print(f"{pname} = {col}", file=sys.stderr)
                print("Warning in check_clubb_settings_api", file=sys.stderr)

    # ── 10. l_implemented constraints (not used in standalone SCM) ───────────
    if l_implemented:
        if config_flags.l_rtm_nudge:
            _fatal("l_rtm_nudge must be set to .false. when l_implemented = .true.")
        if config_flags.l_uv_nudge:
            _fatal("l_uv_nudge must be set to .false. when l_implemented = .true.")

    return err_info._replace(err_code=err_code)


# ── check_parameters_jax ─────────────────────────────────────────────────────

def check_parameters_jax(
    ngrdcol: int,
    clubb_params: np.ndarray,
    lmin: float,
    err_info,
):
    """Port of parameters_tunable.F90:check_parameters_api.

    Validates that tunable parameter values satisfy required constraints.
    Returns updated err_info.
    """
    err_code = np.zeros(ngrdcol, dtype=np.int32)

    def _err(msg: str, col: int) -> None:
        print(msg, file=sys.stderr)
        err_code[col] = CLUBB_FATAL_ERROR

    # ── lmin must be >= 1.0 ──────────────────────────────────────────────────
    if lmin < 1.0:
        print(f"lmin = {lmin}", file=sys.stderr)
        print("lmin is < 1.0", file=sys.stderr)
        err_code[ngrdcol - 1] = CLUBB_FATAL_ERROR

    izeta = _PNAME_IDX["zeta_vrnce_rat"]
    for i in range(ngrdcol):
        p = clubb_params[i, :]

        # All params >= 0 (except zeta_vrnce_rat which requires >= -1)
        for k in range(len(p)):
            if k != izeta and p[k] < 0.0:
                pname = _idx_to_name(k)
                _err(f"{pname} = {p[k]}  ({pname} must satisfy 0.0 <= {pname})", i)
            elif k == izeta and p[k] < -1.0:
                _err(f"zeta_vrnce_rat = {p[k]}  (must satisfy -1.0 <= zeta_vrnce_rat)", i)

        C1                          = p[_PNAME_IDX["C1"]]
        C6rt                        = p[_PNAME_IDX["C6rt"]]
        C6rtb                       = p[_PNAME_IDX["C6rtb"]]
        C6rtc                       = p[_PNAME_IDX["C6rtc"]]
        C6thl                       = p[_PNAME_IDX["C6thl"]]
        C6thlb                      = p[_PNAME_IDX["C6thlb"]]
        C6thlc                      = p[_PNAME_IDX["C6thlc"]]
        C6rt_Lscale0                = p[_PNAME_IDX["C6rt_Lscale0"]]
        C6thl_Lscale0               = p[_PNAME_IDX["C6thl_Lscale0"]]
        C7                          = p[_PNAME_IDX["C7"]]
        C7b                         = p[_PNAME_IDX["C7b"]]
        C11                         = p[_PNAME_IDX["C11"]]
        C11b                        = p[_PNAME_IDX["C11b"]]
        C_wp2_splat                 = p[_PNAME_IDX["C_wp2_splat"]]
        slope_coef_spread_DG_means_w = p[_PNAME_IDX["slope_coef_spread_DG_means_w"]]
        pdf_component_stdev_factor_w = p[_PNAME_IDX["pdf_component_stdev_factor_w"]]
        coef_spread_DG_means_rt     = p[_PNAME_IDX["coef_spread_DG_means_rt"]]
        coef_spread_DG_means_thl    = p[_PNAME_IDX["coef_spread_DG_means_thl"]]
        omicron                     = p[_PNAME_IDX["omicron"]]
        zeta_vrnce_rat              = p[_PNAME_IDX["zeta_vrnce_rat"]]
        upsilon_precip_frac_rat     = p[_PNAME_IDX["upsilon_precip_frac_rat"]]
        mu                          = p[_PNAME_IDX["mu"]]
        beta                        = p[_PNAME_IDX["beta"]]

        if beta < 0.0 or beta > 3.0:
            _err(f"beta = {beta}  (beta cannot be < 0 or > 3)", i)
        if slope_coef_spread_DG_means_w <= 0.0:
            _err(f"slope_coef_spread_DG_means_w = {slope_coef_spread_DG_means_w} (must be > 0)", i)
        if pdf_component_stdev_factor_w <= 0.0:
            _err(f"pdf_component_stdev_factor_w = {pdf_component_stdev_factor_w} (must be > 0)", i)
        if coef_spread_DG_means_rt < 0.0 or coef_spread_DG_means_rt >= 1.0:
            _err(f"coef_spread_DG_means_rt = {coef_spread_DG_means_rt} (must be 0 <= x < 1)", i)
        if coef_spread_DG_means_thl < 0.0 or coef_spread_DG_means_thl >= 1.0:
            _err(f"coef_spread_DG_means_thl = {coef_spread_DG_means_thl} (must be 0 <= x < 1)", i)
        if omicron <= 0.0 or omicron > 1.0:
            _err(f"omicron = {omicron}  (omicron cannot be <= 0 or > 1)", i)
        if zeta_vrnce_rat <= -1.0:
            _err(f"zeta_vrnce_rat = {zeta_vrnce_rat}  (cannot be <= -1)", i)
        if upsilon_precip_frac_rat < 0.0 or upsilon_precip_frac_rat > 1.0:
            _err(f"upsilon_precip_frac_rat = {upsilon_precip_frac_rat} (must be 0 <= x <= 1)", i)
        if mu < 0.0:
            _err(f"mu = {mu}  (mu cannot be < 0)", i)

        def _check_equal(a, b, aname, bname):
            denom = abs(a + b) / 2.0
            if abs(a - b) > (denom * _EPS64 if denom > 0 else _EPS64):
                _err(f"{aname} = {a}, {bname} = {b}  ({aname} and {bname} must be equal)", i)

        _check_equal(C6rt, C6thl, "C6rt", "C6thl")
        _check_equal(C6rtb, C6thlb, "C6rtb", "C6thlb")
        _check_equal(C6rtc, C6thlc, "C6rtc", "C6thlc")
        _check_equal(C6rt_Lscale0, C6thl_Lscale0, "C6rt_Lscale0", "C6thl_Lscale0")

        if C1 < 0.0:
            _err(f"C1 = {C1}  (C1 must satisfy 0.0 <= C1)", i)
        if C7 < 0.0 or C7 > 1.0:
            _err(f"C7 = {C7}  (C7 must satisfy 0.0 <= C7 <= 1.0)", i)
        if C7b < 0.0 or C7b > 1.0:
            _err(f"C7b = {C7b}  (C7b must satisfy 0.0 <= C7b <= 1.0)", i)
        if C11 < 0.0 or C11 > 1.0:
            _err(f"C11 = {C11}  (C11 must satisfy 0.0 <= C11 <= 1.0)", i)
        if C11b < 0.0 or C11b > 1.0:
            _err(f"C11b = {C11b}  (C11b must satisfy 0.0 <= C11b <= 1.0)", i)
        if C_wp2_splat < 0.0:
            _err(f"C_wp2_splat = {C_wp2_splat}  (must satisfy C_wp2_splat >= 0)", i)

    return err_info._replace(err_code=err_code)


def _idx_to_name(k: int) -> str:
    """Reverse lookup: 0-based parameter index → parameter name."""
    for name, idx in _PNAME_IDX.items():
        if idx == k:
            return name
    return f"param[{k}]"
