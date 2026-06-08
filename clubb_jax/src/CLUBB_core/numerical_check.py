"""Pure-Python port of CLUBB_core/numerical_check.F90 and related validation.

Provides:
  check_nan / check_negative — the elemental NaN / negativity field checks (numerical_check.F90 generic check_nan + check_negative)
  length_check            — NaN-check the mixing-length outputs (numerical_check.F90:length_check)
  pdf_closure_check       — NaN-check the pdf_closure outputs + pdf_params (numerical_check.F90:pdf_closure_check)
  sfc_varnce_check        — NaN-check the surface-variance outputs (numerical_check.F90:sfc_varnce_check)
  rad_check               — negativity-check the radiation inputs (numerical_check.F90:rad_check)
  invalid_model_arrays    — NaN/Inf-check the prognostic arrays, True if invalid (numerical_check.F90:invalid_model_arrays)
  parameterization_check  — replaces clubb_api.parameterization_check
  check_clubb_settings    — replaces clubb_api.check_clubb_settings

(check_parameters lives in parameters_tunable.py, mirroring its Fortran home parameters_tunable.F90.)
"""
from __future__ import annotations

import sys

import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg  # REFACTOR B5: detect jax.grad trace
# order_*/ipdf_*/saturation_* are model_flags.F90 enum parameters — import from their Fortran-home model_flags.py
from clubb_jax.src.CLUBB_core.model_flags import (
    order_xm_wpxp as _order_xm_wpxp, order_xp2_xpyp as _order_xp2_xpyp,
    order_wp2_wp3 as _order_wp2_wp3, order_windm as _order_windm,
    ipdf_pre_advance_fields as _ipdf_pre_advance, ipdf_pre_post_advance_fields as _ipdf_pre_post_advance,
    saturation_bolton as _sat_bolton, saturation_gfdl as _sat_gfdl,
    saturation_flatau as _sat_flatau, saturation_lookup as _sat_lookup,
    iiPDF_ADG1 as _iiPDF_ADG1, iiPDF_ADG2 as _iiPDF_ADG2, iiPDF_3D_Luhar as _iiPDF_3D_Luhar,
    iiPDF_new as _iiPDF_new, iiPDF_TSDADG as _iiPDF_TSDADG, iiPDF_LY93 as _iiPDF_LY93,
    iiPDF_new_hybrid as _iiPDF_new_hybrid,
)

CLUBB_NO_ERROR    = 0
CLUBB_FATAL_ERROR = 99

# (iiPDF_*/saturation_*/order_*/ipdf_* enums all imported from model_flags.py above — their Fortran home.)
_l_explicit_turb_adv   = False  # model_flags.F90 line 127

# Parameter name → 0-based index map (single source of truth: parameters_tunable.PARAM_NAMES). Used by
# check_clubb_settings; check_parameters itself lives in parameters_tunable.py (its Fortran home).
from clubb_jax.src.CLUBB_core.parameters_tunable import PNAME_IDX as _PNAME_IDX

_EPS64 = np.finfo(np.float64).eps  # machine epsilon for float64 ≈ 2.22e-16


def calculate_spurious_source(integral_after, integral_before,
                               flux_top, flux_sfc, integral_forcing, dt):
    """Spurious source/sink of a conserved column integral (numerical_check.F90:calculate_spurious_source).

    The residual = d(column integral)/dt + top flux - surface flux - integrated forcing; a nonzero value
    flags non-conservation. Pure arithmetic → differentiable. f2py bit-match (tests/test_spurious_source.py)."""
    return ((integral_after - integral_before) / dt
            + flux_top - flux_sfc - integral_forcing)


def check_nan(arr: np.ndarray, name: str, location: str, err_code: np.ndarray) -> None:
    """Set err_code to CLUBB_FATAL_ERROR if any element of arr is NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        print(f"{name} is NaN in {location}", file=sys.stderr)
        err_code[:] = CLUBB_FATAL_ERROR


def check_negative(arr: np.ndarray, name: str, location: str, err_code: np.ndarray) -> None:
    """Set err_code to CLUBB_FATAL_ERROR if any element of arr is < 0."""
    if np.any(arr < 0.0):
        print(f"{name} < 0 in {location}", file=sys.stderr)
        err_code[:] = CLUBB_FATAL_ERROR


def sfc_varnce_check(sclr_dim, wp2_sfc, up2_sfc, vp2_sfc, thlp2_sfc, rtp2_sfc, rtpthlp_sfc,
                     sclrp2_sfc=None, sclrprtp_sfc=None, sclrpthlp_sfc=None):
    """NaN/Inf-check the calc_surface_varnce outputs (numerical_check.F90:sfc_varnce_check).

    The Fortran NaN-checks each surface variance/covariance and sets err_code=fatal on any non-finite value
    (then returns; the f2py wrapper does not expose the err_code). The JAX path never error-stops, so this
    returns True iff every checked field is finite. Concrete validation (not in any gradient path).
    """
    fields = [(wp2_sfc, "wp2_sfc"), (up2_sfc, "up2_sfc"), (vp2_sfc, "vp2_sfc"),
              (thlp2_sfc, "thlp2_sfc"), (rtp2_sfc, "rtp2_sfc"), (rtpthlp_sfc, "rtpthlp_sfc")]
    if sclr_dim > 0:
        fields += [(sclrp2_sfc, "sclrp2_sfc"), (sclrprtp_sfc, "sclrprtp_sfc"), (sclrpthlp_sfc, "sclrpthlp_sfc")]
    valid = True
    for arr, name in fields:
        if arr is None:
            continue
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            print(f"{name} is NaN in calc_surface_varnce", file=sys.stderr)
            valid = False
    return valid


def length_check(Lscale, Lscale_up, Lscale_down):
    """NaN/Inf-check the mixing-length outputs (numerical_check.F90:length_check).

    The Fortran NaN-checks Lscale/Lscale_up/Lscale_down (proc_name "compute_mixing_length") and sets
    err_code=fatal on any non-finite value. The JAX path never error-stops, so this returns True iff every
    checked field is finite. Concrete validation (not in any gradient path).
    """
    valid = True
    for arr, name in [(Lscale, "Lscale"), (Lscale_up, "Lscale_up"), (Lscale_down, "Lscale_down")]:
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            print(f"{name} is NaN in compute_mixing_length", file=sys.stderr)
            valid = False
    return valid


# pdf_params components NaN-checked by pdf_closure_check, in Fortran order (numerical_check.F90:215-333).
_PDF_CLOSURE_CHECK_FIELDS = (
    "w_1", "w_2", "varnce_w_1", "varnce_w_2", "rt_1", "rt_2", "varnce_rt_1", "varnce_rt_2",
    "thl_1", "thl_2", "varnce_thl_1", "varnce_thl_2", "mixt_frac",
    "corr_w_rt_1", "corr_w_rt_2", "corr_w_thl_1", "corr_w_thl_2", "corr_rt_thl_1", "corr_rt_thl_2",
    "rc_1", "rc_2", "rsatl_1", "rsatl_2", "cloud_frac_1", "cloud_frac_2", "chi_1", "chi_2",
    "stdev_chi_1", "stdev_chi_2", "stdev_eta_1", "stdev_eta_2", "covar_chi_eta_1", "covar_chi_eta_2",
    "corr_w_chi_1", "corr_w_chi_2", "corr_w_eta_1", "corr_w_eta_2", "corr_chi_eta_1", "corr_chi_eta_2",
    "alpha_thl", "alpha_rt", "ice_supersat_frac_1", "ice_supersat_frac_2",
)


def pdf_closure_check(closure_fields, pdf_params, sclr_dim=0, sclr_fields=None):
    """NaN/Inf-check the pdf_closure outputs + every pdf_params component (numerical_check.F90:pdf_closure_check).

    `closure_fields` is a name→array dict of the higher-order-moment outputs (wp4, wprtp2, wp2rtp, …, crt_1,
    crt_2, cthl_1, cthl_2); `pdf_params` is the pdf_parameter NamedTuple (each component checked at the first
    grid column, mirroring the Fortran's `%field(1,:)`); `sclr_fields` is the optional name→array dict of the
    sclr_dim>0 scalar arrays. The Fortran sets err_code=fatal on any non-finite value (proc_name "pdf_closure");
    the JAX path never error-stops, so this returns True iff every checked field is finite. Concrete validation
    (not in any gradient path). The Fortran gates a few checks (wp4/wprtp2/wpthlp2/rcp2/wprtpthlp) on the stats
    list; here every supplied field is checked unconditionally.
    """
    proc = "pdf_closure"
    valid = True

    def _chk(arr, name):
        nonlocal valid
        if arr is None:
            return
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            print(f"{name} is NaN in {proc}", file=sys.stderr)
            valid = False

    for name, arr in closure_fields.items():
        _chk(arr, name)
    for field in _PDF_CLOSURE_CHECK_FIELDS:
        comp = getattr(pdf_params, field, None)
        if comp is not None:
            _chk(np.asarray(comp)[0], f"pdf_params%{field}(1,:)")
    if sclr_dim > 0 and sclr_fields:
        for name, arr in sclr_fields.items():
            _chk(arr, name)
    return valid


def rad_check(thlm, rcm, rtm, rim, cloud_frac, p_in_Pa, exner, rho_zm):
    """Negativity-check the radiation input variables (numerical_check.F90:rad_check).

    The Fortran `check_negative`-checks thlm/rcm/rtm/rvm(=rtm-rcm)/rim/cloud_frac/p_in_Pa/exner/rho_zm (proc_name
    "Before BUGSrad.") and sets err_code=fatal on any value < 0. The JAX path never error-stops, so this returns
    True iff every checked field is >= 0 everywhere. Concrete validation (not in any gradient path).
    """
    rvm = np.asarray(rtm, dtype=np.float64) - np.asarray(rcm, dtype=np.float64)
    fields = [(thlm, "thlm"), (rcm, "rcm"), (rtm, "rtm"), (rvm, "rvm"), (rim, "rim"),
              (cloud_frac, "cloud_frac"), (p_in_Pa, "p_in_Pa"), (exner, "exner"), (rho_zm, "rho_zm")]
    valid = True
    for arr, name in fields:
        if np.any(np.asarray(arr, dtype=np.float64) < 0.0):
            print(f"{name} < 0 in Before BUGSrad.", file=sys.stderr)
            valid = False
    return valid


def invalid_model_arrays(um, vm, rtm, wprtp, thlm, wpthlp, rtp2, thlp2, rtpthlp,
                         wp2, wp3, wp2thvp, wp2up, rtpthvp, thlpthvp,
                         hydromet=None, hydromet_list=None, sclrm=None, edsclrm=None):
    """Check select prognostic model arrays for non-finite (NaN/Inf) values
    (numerical_check.F90:invalid_model_arrays — called from clubb_driver each step).

    Returns **True if ANY checked array is invalid** (non-finite), matching the Fortran function's name and
    return semantics (the driver does `if invalid_model_arrays(...) -> fatal`); this is the inverse polarity of
    the *_check routines above (which return True iff valid). hydromet (shape (..., hydromet_dim)) / sclrm /
    edsclrm (shape (..., dim)) are optional — checked per component, with names from hydromet_list when given.
    Concrete validation (not in any gradient path)."""
    invalid = False
    base = [(um, "um"), (vm, "vm"), (wp2, "wp2"), (wp3, "wp3"), (rtm, "rtm"), (thlm, "thlm"),
            (rtp2, "rtp2"), (thlp2, "thlp2"), (wprtp, "wprtp"), (wpthlp, "wpthlp"),
            (rtpthlp, "rtpthlp"), (wp2thvp, "wp2thvp"), (wp2up, "wp2up"),
            (rtpthvp, "rtpthvp"), (thlpthvp, "thlpthvp")]
    for arr, name in base:
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            print(f"NaN in {name} model array", file=sys.stderr)
            invalid = True
    if hydromet is not None:
        hm = np.asarray(hydromet, dtype=np.float64)
        for i in range(hm.shape[-1]):
            if not np.all(np.isfinite(hm[..., i])):
                nm = hydromet_list[i] if hydromet_list is not None and i < len(hydromet_list) else str(i)
                print(f"NaN in a hydrometeor model array {nm}", file=sys.stderr)
                invalid = True
    for label, block in (("sclrm", sclrm), ("edsclrm", edsclrm)):
        if block is not None:
            b = np.asarray(block, dtype=np.float64)
            for i in range(b.shape[-1]):
                if not np.all(np.isfinite(b[..., i])):
                    print(f"NaN in {label} {i} model array", file=sys.stderr)
                    invalid = True
    return invalid


def parameterization_check(
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
    check_nan(np.asarray(thlm_forcing),    "thlm_forcing",    location, err_code)
    check_nan(np.asarray(rtm_forcing),     "rtm_forcing",     location, err_code)
    check_nan(np.asarray(um_forcing),      "um_forcing",      location, err_code)
    check_nan(np.asarray(vm_forcing),      "vm_forcing",      location, err_code)

    check_nan(np.asarray(wm_zm),           "wm_zm",           location, err_code)
    check_nan(np.asarray(wm_zt),           "wm_zt",           location, err_code)
    check_nan(np.asarray(p_in_Pa),         "p_in_Pa",         location, err_code)
    check_nan(np.asarray(rho_zm),          "rho_zm",          location, err_code)
    check_nan(np.asarray(rho),             "rho",             location, err_code)
    check_nan(np.asarray(exner),           "exner",           location, err_code)
    check_nan(np.asarray(rho_ds_zm),       "rho_ds_zm",       location, err_code)
    check_nan(np.asarray(rho_ds_zt),       "rho_ds_zt",       location, err_code)
    check_nan(np.asarray(invrs_rho_ds_zm), "invrs_rho_ds_zm", location, err_code)
    check_nan(np.asarray(invrs_rho_ds_zt), "invrs_rho_ds_zt", location, err_code)
    check_nan(np.asarray(thv_ds_zm),       "thv_ds_zm",       location, err_code)
    check_nan(np.asarray(thv_ds_zt),       "thv_ds_zt",       location, err_code)

    check_nan(np.asarray(um),      "um",      location, err_code)
    check_nan(np.asarray(upwp),    "upwp",    location, err_code)
    check_nan(np.asarray(vm),      "vm",      location, err_code)
    check_nan(np.asarray(vpwp),    "vpwp",    location, err_code)
    check_nan(np.asarray(up2),     "up2",     location, err_code)
    check_nan(np.asarray(vp2),     "vp2",     location, err_code)
    check_nan(np.asarray(rtm),     "rtm",     location, err_code)
    check_nan(np.asarray(wprtp),   "wprtp",   location, err_code)
    check_nan(np.asarray(thlm),    "thlm",    location, err_code)
    check_nan(np.asarray(wpthlp),  "wpthlp",  location, err_code)
    check_nan(np.asarray(wp2),     "wp2",     location, err_code)
    check_nan(np.asarray(wp3),     "wp3",     location, err_code)
    check_nan(np.asarray(rtp2),    "rtp2",    location, err_code)
    check_nan(np.asarray(thlp2),   "thlp2",   location, err_code)
    check_nan(np.asarray(rtpthlp), "rtpthlp", location, err_code)

    check_nan(np.asarray(wpthlp_sfc), "wpthlp_sfc", location, err_code)
    check_nan(np.asarray(wprtp_sfc),  "wprtp_sfc",  location, err_code)
    check_nan(np.asarray(upwp_sfc),   "upwp_sfc",   location, err_code)
    check_nan(np.asarray(vpwp_sfc),   "vpwp_sfc",   location, err_code)
    check_nan(np.asarray(p_sfc),      "p_sfc",      location, err_code)

    sclrm_arr         = np.asarray(sclrm)
    wpsclrp_sfc_arr   = np.asarray(wpsclrp_sfc)
    wpsclrp_arr       = np.asarray(wpsclrp)
    sclrp2_arr        = np.asarray(sclrp2)
    sclrprtp_arr      = np.asarray(sclrprtp)
    sclrpthlp_arr     = np.asarray(sclrpthlp)
    sclrm_forcing_arr = np.asarray(sclrm_forcing)
    for s in range(sclr_dim):
        check_nan(sclrm_forcing_arr[..., s], "sclrm_forcing", location, err_code)
        check_nan(wpsclrp_sfc_arr[:, s],     "wpsclrp_sfc",   location, err_code)
        check_nan(sclrm_arr[..., s],          "sclrm",         location, err_code)
        check_nan(wpsclrp_arr[..., s],        "wpsclrp",       location, err_code)
        check_nan(sclrp2_arr[..., s],         "sclrp2",        location, err_code)
        check_nan(sclrprtp_arr[..., s],       "sclrprtp",      location, err_code)
        check_nan(sclrpthlp_arr[..., s],      "sclrpthlp",     location, err_code)

    edsclrm_arr         = np.asarray(edsclrm)
    edsclrm_forcing_arr = np.asarray(edsclrm_forcing)
    wpedsclrp_sfc_arr   = np.asarray(wpedsclrp_sfc)
    for e in range(edsclr_dim):
        check_nan(edsclrm_forcing_arr[..., e], "edsclrm_forcing", location, err_code)
        check_nan(wpedsclrp_sfc_arr[:, e],     "wpedsclrp_sfc",   location, err_code)
        check_nan(edsclrm_arr[..., e],          "edsclrm",         location, err_code)

    # ── 2. Return early if NaN fatal error found ──────────────────────────────
    if np.any(err_code == CLUBB_FATAL_ERROR):
        return err_info._replace(err_code=err_code)

    # ── 3. Negativity checks ──────────────────────────────────────────────────
    check_negative(np.asarray(rtm),           "rtm",           location, err_code)
    check_negative(np.asarray(p_in_Pa),       "p_in_Pa",       location, err_code)
    check_negative(np.asarray(rho),           "rho",           location, err_code)
    check_negative(np.asarray(rho_zm),        "rho_zm",        location, err_code)
    check_negative(np.asarray(exner),         "exner",         location, err_code)
    check_negative(np.asarray(rho_ds_zm),     "rho_ds_zm",     location, err_code)
    check_negative(np.asarray(rho_ds_zt),     "rho_ds_zt",     location, err_code)
    check_negative(np.asarray(invrs_rho_ds_zm), "invrs_rho_ds_zm", location, err_code)
    check_negative(np.asarray(invrs_rho_ds_zt), "invrs_rho_ds_zt", location, err_code)
    check_negative(np.asarray(thv_ds_zm),     "thv_ds_zm",     location, err_code)
    check_negative(np.asarray(thv_ds_zt),     "thv_ds_zt",     location, err_code)
    check_negative(np.asarray(up2),           "up2",           location, err_code)
    check_negative(np.asarray(vp2),           "vp2",           location, err_code)
    check_negative(np.asarray(wp2),           "wp2",           location, err_code)
    check_negative(np.asarray(thlm),          "thlm",          location, err_code)
    check_negative(np.asarray(rtp2),          "rtp2",          location, err_code)
    check_negative(np.asarray(thlp2),         "thlp2",         location, err_code)

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


# ── check_clubb_settings ─────────────────────────────────────────────────

def check_clubb_settings(
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
