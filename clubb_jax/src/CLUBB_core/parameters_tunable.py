"""Pure-Python implementation of selected ``parameters_tunable.F90`` APIs.

Description:
  This module contains tunable model parameters.  The purpose of the module is to make it
  easier for the clubb_tuner code to use the clubb_params vector without "knowing" any
  information about the individual parameters contained in the vector itself.
  It makes it easier to add
  new parameters to be tuned for, but does not make the CLUBB_core code itself any simpler.
  The parameters within the vector do not need to be the same variables used in the rest of
  CLUBB_core (see for e.g. nu1_vert_res_dep or lmin_coef).
  The parameters in the clubb_params vector only need to be those parameters for which we're not
  sure the correct value and we'd like to tune for.

References:
  None

Porting deviations:
The JAX driver needs ``get_param_names``, ``init_clubb_params``,
``calc_derrived_params``, and ``check_parameters``.  The Fortran tuner-only
``read_param_minmax`` and ``read_param_constraints`` routines are omitted.
Fortran ``pack_parameters`` and ``unpack_parameters`` are represented by
``PARAM_NAMES``/``PNAME_IDX`` plus a Python dictionary of defaults.  Fortran
mutates derived-type allocations in place; Python returns a ``NuVertResDep``
named tuple with NumPy arrays.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from clubb_jax.src.Input_fields.namelist import read_namelist
from clubb_jax.src.CLUBB_core.nu_vert_res_dep import NuVertResDep


# ---------------------------------------------------------------------------
# These are referenced together often enough that it made sense to
# make a list of them.  Note that lmin_coef is the input parameter,
# while the actual lmin model constant is computed from this.
#***************************************************************
#                    ***** IMPORTANT *****
# If you change the order of the parameters in the parameter_indices,
# you will need to change the order of this list as well or the
# tuner will break!
#                    ***** IMPORTANT *****
#***************************************************************
# ---------------------------------------------------------------------------

PARAM_NAMES: list[str] = [
    "C1", "C1b", "C1c",
    "C2rt", "C2thl", "C2rtthl",
    "C4", "C_uu_shr", "C_uu_buoy",
    "C6rt", "C6rtb", "C6rtc",
    "C6thl", "C6thlb", "C6thlc",
    "C7", "C7b", "C7c",
    "C8", "C8b",
    "C10",
    "C11", "C11b", "C11c",
    "C12", "C13", "C14",
    "C_wp2_pr_dfsn", "C_wp3_pr_tp", "C_wp3_pr_turb", "C_wp3_pr_dfsn",
    "C_wp2_splat",
    "C6rt_Lscale0", "C6thl_Lscale0", "C7_Lscale0", "wpxp_L_thresh",
    "c_K", "c_K1", "nu1",
    "c_K2", "nu2",
    "c_K6", "nu6",
    "c_K8", "nu8",
    "c_K9", "nu9", "nu10",
    "c_K_hm", "c_K_hmb", "K_hm_min_coef", "nu_hm",
    "slope_coef_spread_DG_means_w", "pdf_component_stdev_factor_w",
    "coef_spread_DG_means_rt", "coef_spread_DG_means_thl",
    "gamma_coef", "gamma_coefb", "gamma_coefc",
    "mu", "beta", "lmin_coef",
    "omicron", "zeta_vrnce_rat", "upsilon_precip_frac_rat",
    "lambda0_stability_coef", "mult_coef",
    "taumin", "taumax",
    "Lscale_mu_coef", "Lscale_pert_coef",
    "alpha_corr", "Skw_denom_coef",
    "c_K10", "c_K10h",
    "thlp2_rad_coef", "thlp2_rad_cloud_frac_thresh",
    "up2_sfc_coef",
    "Skw_max_mag",
    "C_invrs_tau_bkgnd", "C_invrs_tau_sfc", "C_invrs_tau_shear",
    "C_invrs_tau_N2", "C_invrs_tau_N2_wp2", "C_invrs_tau_N2_xp2",
    "C_invrs_tau_N2_wpxp", "C_invrs_tau_N2_clear_wp3",
    "C_invrs_tau_wpxp_Ri", "C_invrs_tau_wpxp_N2_thresh",
    "xp3_coef_base", "xp3_coef_slope",
    "altitude_threshold", "rtp2_clip_coef",
    "Cx_min", "Cx_max",
    "Richardson_num_min", "Richardson_num_max",
    "a3_coef_min", "a_const", "bv_efold", "wpxp_Ri_exp", "z_displace",
]
assert len(PARAM_NAMES) == 102, f"Expected 102 params, got {len(PARAM_NAMES)}"

# Case-insensitive lookup: lower-case name → 0-based index
_NAME_TO_IDX: dict[str, int] = {n.lower(): i for i, n in enumerate(PARAM_NAMES)}

# Case-sensitive name → 0-based index (the single source of truth for parameter ordering; numerical_check
# imports this for check_clubb_settings).
PNAME_IDX: dict[str, int] = {n: i for i, n in enumerate(PARAM_NAMES)}

# Sentinel for a fatal validation error (error_code.F90); used by check_parameters below.
CLUBB_FATAL_ERROR = 99
_EPS64 = np.finfo(np.float64).eps

# ---------------------------------------------------------------------------
# Default parameter values (mirrors set_default_parameters in Fortran)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, float] = {
    "C1": 1.0, "C1b": 1.0, "C1c": 1.0,
    "C2rt": 2.0, "C2thl": 2.0, "C2rtthl": 2.0,
    "C4": 2.0, "C_uu_shr": 0.4, "C_uu_buoy": 0.3,
    "C6rt": 2.0, "C6rtb": 2.0, "C6rtc": 1.0,
    "C6thl": 2.0, "C6thlb": 2.0, "C6thlc": 1.0,
    "C7": 0.5, "C7b": 0.5, "C7c": 0.5,
    "C8": 0.5, "C8b": 0.02,
    "C10": 3.3,
    "C11": 0.4, "C11b": 0.4, "C11c": 0.5,
    "C12": 1.0, "C13": 0.1, "C14": 1.0,
    "C_wp2_pr_dfsn": 0.0, "C_wp3_pr_tp": 0.0, "C_wp3_pr_turb": 0.0,
    "C_wp3_pr_dfsn": 0.0, "C_wp2_splat": 2.0,
    "C6rt_Lscale0": 14.0, "C6thl_Lscale0": 14.0, "C7_Lscale0": 0.85,
    "wpxp_L_thresh": 60.0,
    "c_K": 0.2, "c_K1": 0.2,
    "nu1": 20.0, "c_K2": 0.025, "nu2": 1.0,
    "c_K6": 0.375, "nu6": 5.0,
    "c_K8": 5.0, "nu8": 20.0,
    "c_K9": 0.1, "nu9": 10.0, "nu10": 0.0,
    "c_K_hm": 0.75, "c_K_hmb": 0.75, "K_hm_min_coef": 0.1, "nu_hm": 1.5,
    "slope_coef_spread_DG_means_w": 21.0, "pdf_component_stdev_factor_w": 1.0,
    "coef_spread_DG_means_rt": 0.8, "coef_spread_DG_means_thl": 0.8,
    "gamma_coef": 0.25, "gamma_coefb": 0.25, "gamma_coefc": 5.0,
    "mu": 1.0e-3, "beta": 1.0, "lmin_coef": 0.5,
    "omicron": 0.5, "zeta_vrnce_rat": 0.0, "upsilon_precip_frac_rat": 0.55,
    "lambda0_stability_coef": 0.03, "mult_coef": 0.5,
    "taumin": 90.0, "taumax": 3600.0,
    "Lscale_mu_coef": 2.0, "Lscale_pert_coef": 0.1,
    "alpha_corr": 0.15, "Skw_denom_coef": 4.0,
    "c_K10": 1.0, "c_K10h": 1.0,
    "thlp2_rad_coef": 1.0, "thlp2_rad_cloud_frac_thresh": 0.1,
    "up2_sfc_coef": 4.0,
    "Skw_max_mag": 10.0,
    "C_invrs_tau_bkgnd": 1.1, "C_invrs_tau_sfc": 0.1, "C_invrs_tau_shear": 0.15,
    "C_invrs_tau_N2": 0.4, "C_invrs_tau_N2_wp2": 0.2, "C_invrs_tau_N2_xp2": 0.05,
    "C_invrs_tau_N2_wpxp": 0.0, "C_invrs_tau_N2_clear_wp3": 1.0,
    "C_invrs_tau_wpxp_Ri": 0.35, "C_invrs_tau_wpxp_N2_thresh": 3.3e-4,
    "xp3_coef_base": 0.25, "xp3_coef_slope": 0.01,
    "altitude_threshold": 100.0, "rtp2_clip_coef": 0.5,
    "Cx_min": 0.33, "Cx_max": 0.95,
    "Richardson_num_min": 0.25, "Richardson_num_max": 400.0,
    "a3_coef_min": 1.0, "a_const": 1.8, "bv_efold": 5.0,
    "wpxp_Ri_exp": 0.5, "z_displace": 25.0,
}
assert set(_DEFAULTS.keys()) == set(PARAM_NAMES), "Defaults must cover all 102 parameters"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_param_names() -> list[str]:
    """Return the 102 parameter names in canonical Fortran ordering."""
    return list(PARAM_NAMES)


def init_clubb_params(ngrdcol: int, filename: str) -> np.ndarray:
    """Read a namelist containing the model parameters.

    Description:
    Read a namelist containing the model parameters

    References:
    None
    """
    nparams = 102

    # Set the default tunable parameter values
    values = np.tile(
        np.array([_DEFAULTS[n] for n in PARAM_NAMES], dtype=np.float64),
        (ngrdcol, 1),
    )

    # If the filename is empty, assume we're using a `working' set of
    # parameters that are set statically here (handy for host models).
    # Read the namelist
    nml = read_namelist(filename)
    for raw_name, val in nml.items():
        key = raw_name.lower().strip()
        if key in _NAME_TO_IDX:
            idx = _NAME_TO_IDX[key]
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 0:
                values[:, idx] = float(arr)
            else:
                arr = arr.ravel()
                if arr.size == 1:
                    values[:, idx] = float(arr[0])
                elif arr.size == ngrdcol:
                    values[:, idx] = arr
                else:
                    raise ValueError(
                        f"{raw_name} must be scalar or have ngrdcol={ngrdcol} values; "
                        f"got {arr.size}."
                    )

    return values


def calc_derrived_params(
    gr,
    ngrdcol: int,
    grid_type: int,
    deltaz: np.ndarray,
    clubb_params: np.ndarray,
    l_prescribed_avg_deltaz: bool,
) -> tuple[NuVertResDep, float, float]:
    """Calculate parameters that should be derrived from other quantities.

    Description:
      Calculates clubb parameters that should be derrived from other quantities.

      Adjusts the values of background eddy diffusivity based on
      vertical grid spacing.
      This code was made into a public subroutine so that it may be
      called multiple times per model run in scenarios where grid
      altitudes, and hence average grid spacing, change through space
      and/or time.  This occurs, for example, when CLUBB is
      implemented in WRF.  --ldgrant Jul 2010
    """
    #------------------------------ Constant Parameters ------------------------------
    # Fixed value for minimum value for the length scale.
    lmin_deltaz = 40.0

    # It was decided after some experimentation, that the best
    # way to produce grid independent results is to set lmin to be
    # some fixed value. -dschanen 21 May 2007
    # TODO: using "clubb_params(ngrdcol,ilmin_coef)", but lmin should really be
    # changed to dimension(ngrdcol) to avoid this
    lmin = float(clubb_params[ngrdcol - 1, 61]) * lmin_deltaz

    # Using ngrdcol here as well for temporary backward compatibility, same as above
    Skw_max = float(clubb_params[ngrdcol - 1, 78])
    inner = 4.0 * (1.0 - 0.4) ** 3 + Skw_max ** 2
    mixt_frac_max_mag = 1.0 - 0.5 * (1.0 - Skw_max / math.sqrt(inner))
    # Known magic number

    #------------------------------ Local Variables ------------------------------
    # Average grid box height   [m]
    deltaz = np.asarray(deltaz, dtype=np.float64).ravel()
    avg_deltaz = np.empty(ngrdcol, dtype=np.float64)

    if l_prescribed_avg_deltaz or grid_type == 1:
        avg_deltaz[:] = deltaz
    elif grid_type == 2:
        # Stretched (unevenly-spaced) grid:  stretched thermodynamic level
        # input.
        # Find the average deltaz over the stretched grid based on
        # thermodynamic level inputs.
        zt = np.asarray(gr.zt)  # (ngrdcol, nzt)
        for i in range(ngrdcol):
            avg_deltaz[i] = (zt[i, -1] - zt[i, 0]) / max(1, zt.shape[1] - 1)
    elif grid_type == 3:
        # CLUBB is implemented in a host model, or is using grid_type = 3
        # Find the average deltaz over the grid based on momentum level
        # inputs.
        zm = np.asarray(gr.zm)  # (ngrdcol, nzm)
        for i in range(ngrdcol):
            avg_deltaz[i] = (zm[i, -1] - zm[i, 0]) / max(1, zm.shape[1] - 1)
    else:
        avg_deltaz[:] = deltaz

    # Flag for adjusting the values of the constant background eddy diffusivity
    # coefficients based on the average vertical grid spacing.  If this flag is
    # turned off, the values of the various nu coefficients will remain as they
    # are declared in the tunable_parameters.in file.

    # The size of the average vertical grid spacing that serves as a threshold
    # for when to increase the size of the background eddy diffusivity
    # coefficients (nus) by a certain factor above what the background
    # coefficients are specified to be in tunable_parameters.in.  At any average
    # grid spacing at or below this value, the values of the background
    # diffusivities remain the same.  However, at any average vertical grid
    # spacing above this value, the values of the background eddy diffusivities
    # are increased.  Traditionally, the threshold grid spacing has been set to
    # 40.0 meters.  This is only relevant if l_adj_low_res_nu is turned on.
    grid_spacing_thresh = 40.0

    # The factor by which to multiply the coefficients of background eddy
    # diffusivity if the grid spacing threshold is exceeded and l_adj_low_res_nu
    # is turned on.
    mult_factor = np.ones(ngrdcol, dtype=np.float64)
    for i in range(ngrdcol):
        if avg_deltaz[i] > grid_spacing_thresh:
            mult_factor[i] = 1.0 + float(clubb_params[i, 66]) * math.log(
                avg_deltaz[i] / grid_spacing_thresh
            )

    # The nu's are chosen for deltaz <= 40 m. Looks like they must
    # be adjusted for larger grid spacings (Vince Larson)

    # Use a constant mult_factor so nu does not depend on grid spacing
    nu1   = clubb_params[:, 38] * mult_factor
    nu2   = clubb_params[:, 40] * mult_factor
    nu6   = clubb_params[:, 42] * mult_factor
    nu8   = clubb_params[:, 44] * mult_factor   # zt-level
    nu9   = clubb_params[:, 46] * mult_factor
    nu10  = clubb_params[:, 47] * mult_factor   # zt-level (disabled in ARM)
    nu_hm = clubb_params[:, 51] * mult_factor   # zt-level

    nu_vert_res_dep = NuVertResDep(
        nzm=int(gr.nzm),
        nu1=nu1.copy(),
        nu2=nu2.copy(),
        nu6=nu6.copy(),
        nu8=nu8.copy(),
        nu9=nu9.copy(),
        nu10=nu10.copy(),
        nu_hm=nu_hm.copy(),
    )

    return nu_vert_res_dep, lmin, mixt_frac_max_mag


def _idx_to_name(k: int) -> str:
    """Reverse lookup: 0-based parameter index → parameter name."""
    for name, idx in PNAME_IDX.items():
        if idx == k:
            return name
    return f"param[{k}]"


def check_parameters(
    ngrdcol: int,
    clubb_params: np.ndarray,
    lmin: float,
    err_info,
):
    """Validate tunable parameter constraints.

    Description:
    Subroutine to setup model parameters

    References:
    None
    """
    err_code = np.zeros(ngrdcol, dtype=np.int32)

    def _err(msg: str, col: int) -> None:
        print(msg, file=sys.stderr)
        err_code[col] = CLUBB_FATAL_ERROR

    #-------------------- Begin code --------------------

    # This should have ngrdcol dimensions, but doesn't yet
    if lmin < 1.0:
        print(f"lmin = {lmin}", file=sys.stderr)
        print("lmin is < 1.0", file=sys.stderr)
        err_code[ngrdcol - 1] = CLUBB_FATAL_ERROR

    izeta = PNAME_IDX["zeta_vrnce_rat"]
    for i in range(ngrdcol):
        p = clubb_params[i, :]

        # Ensure all variables are greater than 0, and zeta_vrnce_rat is greater than -1
        for k in range(len(p)):
            if k != izeta and p[k] < 0.0:
                pname = _idx_to_name(k)
                _err(f"{pname} = {p[k]}  ({pname} must satisfy 0.0 <= {pname})", i)
            elif k == izeta and p[k] < -1.0:
                _err(f"zeta_vrnce_rat = {p[k]}  (must satisfy -1.0 <= zeta_vrnce_rat)", i)

        C1                          = p[PNAME_IDX["C1"]]
        C6rt                        = p[PNAME_IDX["C6rt"]]
        C6rtb                       = p[PNAME_IDX["C6rtb"]]
        C6rtc                       = p[PNAME_IDX["C6rtc"]]
        C6thl                       = p[PNAME_IDX["C6thl"]]
        C6thlb                      = p[PNAME_IDX["C6thlb"]]
        C6thlc                      = p[PNAME_IDX["C6thlc"]]
        C6rt_Lscale0                = p[PNAME_IDX["C6rt_Lscale0"]]
        C6thl_Lscale0               = p[PNAME_IDX["C6thl_Lscale0"]]
        C7                          = p[PNAME_IDX["C7"]]
        C7b                         = p[PNAME_IDX["C7b"]]
        C11                         = p[PNAME_IDX["C11"]]
        C11b                        = p[PNAME_IDX["C11b"]]
        C_wp2_splat                 = p[PNAME_IDX["C_wp2_splat"]]
        slope_coef_spread_DG_means_w = p[PNAME_IDX["slope_coef_spread_DG_means_w"]]
        pdf_component_stdev_factor_w = p[PNAME_IDX["pdf_component_stdev_factor_w"]]
        coef_spread_DG_means_rt     = p[PNAME_IDX["coef_spread_DG_means_rt"]]
        coef_spread_DG_means_thl    = p[PNAME_IDX["coef_spread_DG_means_thl"]]
        omicron                     = p[PNAME_IDX["omicron"]]
        zeta_vrnce_rat              = p[PNAME_IDX["zeta_vrnce_rat"]]
        upsilon_precip_frac_rat     = p[PNAME_IDX["upsilon_precip_frac_rat"]]
        mu                          = p[PNAME_IDX["mu"]]
        beta                        = p[PNAME_IDX["beta"]]

        if beta < 0.0 or beta > 3.0:
            # Constraints on beta
            _err(f"beta = {beta}  (beta cannot be < 0 or > 3)", i)
        if slope_coef_spread_DG_means_w <= 0.0:
            # Constraint on slope_coef_spread_DG_means_w
            _err(f"slope_coef_spread_DG_means_w = {slope_coef_spread_DG_means_w} (must be > 0)", i)
        if pdf_component_stdev_factor_w <= 0.0:
            # Constraint on pdf_component_stdev_factor_w
            _err(f"pdf_component_stdev_factor_w = {pdf_component_stdev_factor_w} (must be > 0)", i)
        if coef_spread_DG_means_rt < 0.0 or coef_spread_DG_means_rt >= 1.0:
            # Constraint on coef_spread_DG_means_rt
            _err(f"coef_spread_DG_means_rt = {coef_spread_DG_means_rt} (must be 0 <= x < 1)", i)
        if coef_spread_DG_means_thl < 0.0 or coef_spread_DG_means_thl >= 1.0:
            # Constraint on coef_spread_DG_means_thl
            _err(f"coef_spread_DG_means_thl = {coef_spread_DG_means_thl} (must be 0 <= x < 1)", i)
        if omicron <= 0.0 or omicron > 1.0:
            # Constraints on omicron
            _err(f"omicron = {omicron}  (omicron cannot be <= 0 or > 1)", i)
        if zeta_vrnce_rat <= -1.0:
            # Constraints on zeta_vrnce_rat
            _err(f"zeta_vrnce_rat = {zeta_vrnce_rat}  (cannot be <= -1)", i)
        if upsilon_precip_frac_rat < 0.0 or upsilon_precip_frac_rat > 1.0:
            # Constraints on upsilon_precip_frac_rat
            _err(f"upsilon_precip_frac_rat = {upsilon_precip_frac_rat} (must be 0 <= x <= 1)", i)
        if mu < 0.0:
            # Constraints on entrainment rate, mu.
            _err(f"mu = {mu}  (mu cannot be < 0)", i)

        def _check_equal(a, b, aname, bname):
            denom = abs(a + b) / 2.0
            if abs(a - b) > (denom * _EPS64 if denom > 0 else _EPS64):
                _err(f"{aname} = {a}, {bname} = {b}  ({aname} and {bname} must be equal)", i)

        _check_equal(C6rt, C6thl, "C6rt", "C6thl")
        _check_equal(C6rtb, C6thlb, "C6rtb", "C6thlb")
        _check_equal(C6rtc, C6thlc, "C6rtc", "C6thlc")
        _check_equal(C6rt_Lscale0, C6thl_Lscale0, "C6rt_Lscale0", "C6thl_Lscale0")

        # The C6rt parameters must be set equal to the C6thl parameters.
        # Otherwise, the wpthlp pr1 term will be calculated inconsistently.

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
