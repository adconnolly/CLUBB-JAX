"""Pure-Python port of CLUBB_core/parameters_tunable.F90.

Provides init_clubb_params_jax and calc_derrived_params_jax to replace the
corresponding Fortran API calls in clubb_standalone.py.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from clubb_jax.src.Input_fields.namelist import read_namelist

# ---------------------------------------------------------------------------
# NuVertResDep: mirrors clubb_python.derived_types.nu_vert_res_dep.NuVertResDep
# ---------------------------------------------------------------------------

class NuVertResDep(NamedTuple):
    """Vertical-resolution-dependent nu coefficient arrays (one value per column)."""
    nzm: int          # stored as ngrdcol to match the Fortran Python-API convention
    nu1:   np.ndarray  # shape (ngrdcol,) — background eddy diff for wp2    [m²/s]
    nu2:   np.ndarray  # shape (ngrdcol,) — background eddy diff for xp2    [m²/s]
    nu6:   np.ndarray  # shape (ngrdcol,) — background eddy diff for wpxp   [m²/s]
    nu8:   np.ndarray  # shape (ngrdcol,) — background eddy diff for wp3    [m²/s]
    nu9:   np.ndarray  # shape (ngrdcol,) — background eddy diff for up2/vp2 [m²/s]
    nu10:  np.ndarray  # shape (ngrdcol,) — background eddy diff for edsclrm [m²/s]
    nu_hm: np.ndarray  # shape (ngrdcol,) — background eddy diff for hmm    [m²/s]


# ---------------------------------------------------------------------------
# Canonical 102-element parameter name list (matches Fortran params_list order)
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

def get_param_names_jax() -> list[str]:
    """Return the 102 parameter names in canonical Fortran ordering."""
    return list(PARAM_NAMES)


def init_clubb_params_jax(ngrdcol: int, filename: str) -> np.ndarray:
    """Read tunable_parameters.in and return clubb_params array.

    Parameters
    ----------
    ngrdcol:  number of grid columns
    filename: path to tunable_parameters.in (Fortran namelist, group clubb_params_nl)

    Returns
    -------
    clubb_params: ndarray of shape (ngrdcol, 102), dtype float64
    """
    nparams = 102

    # Start with defaults (broadcast to all columns)
    values = np.array([_DEFAULTS[n] for n in PARAM_NAMES], dtype=np.float64)

    # Read and apply namelist overrides
    nml = read_namelist(filename)
    for raw_name, val in nml.items():
        key = raw_name.lower().strip()
        if key in _NAME_TO_IDX:
            values[_NAME_TO_IDX[key]] = float(val)

    # Broadcast to (ngrdcol, nparams) — all columns get the same values
    clubb_params = np.tile(values, (ngrdcol, 1))
    return clubb_params


def calc_derrived_params_jax(
    gr,
    ngrdcol: int,
    grid_type: int,
    deltaz: np.ndarray,
    clubb_params: np.ndarray,
    l_prescribed_avg_deltaz: bool,
) -> tuple[NuVertResDep, float, float]:
    """Port of parameters_tunable.F90 calc_Derrived_Params_api.

    Computes lmin, mixt_frac_max_mag, and nu_vert_res_dep from clubb_params
    and the vertical grid.

    Parameters
    ----------
    gr:       grid object with attributes zt (ngrdcol, nzt), zm (ngrdcol, nzm),
              nzt, nzm, k_ub_zt, k_lb_zt, k_ub_zm, k_lb_zm
    ngrdcol:  number of grid columns
    grid_type: 1=evenly-spaced, 2=stretched zt-input, 3=stretched zm-input
    deltaz:   shape (ngrdcol,) — evenly-spaced grid spacing or prescribed avg deltaz
    clubb_params: shape (ngrdcol, 102)
    l_prescribed_avg_deltaz: if True, avg_deltaz = deltaz directly

    Returns
    -------
    nu_vert_res_dep, lmin, mixt_frac_max_mag
    """
    # ── 1. lmin (fixed formula, not grid-dependent) ────────────────────────
    # Fortran: lmin = clubb_params(ngrdcol, ilmin_coef) * lmin_deltaz
    # Python 0-based: clubb_params[ngrdcol-1, ilmin_coef-1]
    # ilmin_coef = 62 → index 61
    lmin_deltaz = 40.0
    lmin = float(clubb_params[ngrdcol - 1, 61]) * lmin_deltaz

    # ── 2. mixt_frac_max_mag ──────────────────────────────────────────────
    # Fortran: 1 - 0.5*(1 - Skw_max/sqrt(4*(1-0.4)^3 + Skw_max^2))
    # iSkw_max_mag = 79 → index 78
    Skw_max = float(clubb_params[ngrdcol - 1, 78])
    inner = 4.0 * (1.0 - 0.4) ** 3 + Skw_max ** 2
    mixt_frac_max_mag = 1.0 - 0.5 * (1.0 - Skw_max / math.sqrt(inner))

    # ── 3. avg_deltaz per column ───────────────────────────────────────────
    deltaz = np.asarray(deltaz, dtype=np.float64).ravel()
    avg_deltaz = np.empty(ngrdcol, dtype=np.float64)

    if l_prescribed_avg_deltaz or grid_type == 1:
        avg_deltaz[:] = deltaz
    elif grid_type == 2:
        # Stretched grid on zt levels: average over interior zt levels
        # Fortran: (zt(k_ub_zt) - zt(k_lb_zt)) / (nzt - 1)
        zt = np.asarray(gr.zt)  # (ngrdcol, nzt)
        for i in range(ngrdcol):
            avg_deltaz[i] = (zt[i, -1] - zt[i, 0]) / max(1, zt.shape[1] - 1)
    elif grid_type == 3:
        # Stretched grid on zm levels
        zm = np.asarray(gr.zm)  # (ngrdcol, nzm)
        for i in range(ngrdcol):
            avg_deltaz[i] = (zm[i, -1] - zm[i, 0]) / max(1, zm.shape[1] - 1)
    else:
        avg_deltaz[:] = deltaz

    # ── 4. mult_factor (l_adj_low_res_nu = True always) ───────────────────
    grid_spacing_thresh = 40.0
    # imult_coef = 67 → index 66
    mult_factor = np.ones(ngrdcol, dtype=np.float64)
    for i in range(ngrdcol):
        if avg_deltaz[i] > grid_spacing_thresh:
            mult_factor[i] = 1.0 + float(clubb_params[i, 66]) * math.log(
                avg_deltaz[i] / grid_spacing_thresh
            )

    # ── 5. nu_vert_res_dep (scaled nu coefficients) ───────────────────────
    # Index mapping (0-based): inu1=39→38, inu2=41→40, inu6=43→42,
    #   inu8=45→44, inu9=47→46, inu10=48→47, inu_hm=52→51
    nu1   = clubb_params[:, 38] * mult_factor
    nu2   = clubb_params[:, 40] * mult_factor
    nu6   = clubb_params[:, 42] * mult_factor
    nu8   = clubb_params[:, 44] * mult_factor   # zt-level
    nu9   = clubb_params[:, 46] * mult_factor
    nu10  = clubb_params[:, 47] * mult_factor   # zt-level (disabled in ARM)
    nu_hm = clubb_params[:, 51] * mult_factor   # zt-level

    nu_vert_res_dep = NuVertResDep(
        nzm=ngrdcol,
        nu1=nu1.copy(),
        nu2=nu2.copy(),
        nu6=nu6.copy(),
        nu8=nu8.copy(),
        nu9=nu9.copy(),
        nu10=nu10.copy(),
        nu_hm=nu_hm.copy(),
    )

    return nu_vert_res_dep, lmin, mixt_frac_max_mag
