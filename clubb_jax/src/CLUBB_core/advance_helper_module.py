"""JAX implementations of advance_helper_module.F90 routines.

Merged from: advance_helper.py, splat.py, brunt_vaisala.py
Fortran source: src/CLUBB_core/advance_helper_module.F90
"""

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import (
    iCx_min,
    iCx_max,
    iRichardson_num_min,
    iRichardson_num_max,
    ilambda0_stability_coef,
    ithlp2_rad_coef,
    Cp,
    Lv,
    Rd,
    ep,
    grav,
    rc_tol,
    zero_threshold,
    min_max_smth_mag,
)
from clubb_jax.src.CLUBB_core.grid_class import (
    zm2zt_jax,
    zm2zt2zm_jax,
    zt2zm_jax,
    ddzt_jax,
)
from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt  # REFACTOR B5: finite grad for sqrt(max(0,·))

# ── advance_helper_module constants ─────────────────────────────────────────
_RICHARDSON_DIV_THRESH = 1.0e-6  # Richardson_num_divisor_threshold

# ── splat constants ──────────────────────────────────────────────────────────
_SMTH_TYPE2_HALF_WIDTH = 60.0  # [m] — from Fortran local parameter, smth_type=2

# ── brunt_vaisala constants ──────────────────────────────────────────────────
_T_FREEZE_K = 273.15   # Freezing point of water [K]
_MIN_T_IN_C = -85.0    # Minimum T used in Flatau polynomial [°C]


# ============================================================================
# Richardson / stability helpers  (advance_helper_module.F90)
# ============================================================================

def calc_ri_zm_jax(bv_freq_sqd, shear, lim_bv, lim_shear):
    """Richardson number (advance_helper_module.F90:calc_Ri_zm)."""
    return jnp.maximum(bv_freq_sqd, lim_bv) / jnp.maximum(shear, lim_shear)


def compute_cx_fnc_richardson_jax(
    brunt_vaisala_freq_sqd,
    brunt_vaisala_freq_sqd_mixed,
    ddzt_umvm_sqd,
    clubb_params,
    l_use_shear_Richardson: bool,
    l_modify_limiters_for_cnvg_test: bool,
):
    """Cx as a function of Richardson number (advance_helper_module.F90:compute_Cx_fnc_Richardson)."""
    Cx_min = clubb_params[:, iCx_min - 1:iCx_min]
    Cx_max = clubb_params[:, iCx_max - 1:iCx_max]
    Ri_min = clubb_params[:, iRichardson_num_min - 1:iRichardson_num_min]
    Ri_max = clubb_params[:, iRichardson_num_max - 1:iRichardson_num_max]

    if l_use_shear_Richardson:
        Ri_zm_Cx = calc_ri_zm_jax(brunt_vaisala_freq_sqd_mixed, ddzt_umvm_sqd, 1.0e-7, 1.0e-7)
    else:
        Ri_zm_Cx = brunt_vaisala_freq_sqd * (1.0 / _RICHARDSON_DIV_THRESH)

    if l_modify_limiters_for_cnvg_test:
        invrs_min_max_diff = 1.0 / (Ri_max - Ri_min)
        fnc_Richardson = (Ri_zm_Cx - Ri_min) * invrs_min_max_diff
        fnc_clipped = _smooth_min_scalar_array(1.0, fnc_Richardson)
        fnc_smooth  = _smooth_max_scalar_array(0.0, fnc_clipped)
        Cx_fnc = fnc_smooth * (Cx_max - Cx_min) + Cx_min
    else:
        invrs_min_max_diff = 1.0 / (Ri_max - Ri_min)
        Ri_clipped = jnp.maximum(jnp.minimum(Ri_max, Ri_zm_Cx), Ri_min)
        Cx_fnc = (Ri_clipped - Ri_min) * invrs_min_max_diff * (Cx_max - Cx_min) + Cx_min

    return jnp.maximum(0.0, jnp.minimum(1.0, Cx_fnc))


def smooth_max_jax(a, b, smth_coef):
    """Smooth approximation to max(a, b) (advance_helper_module.F90:smooth_max)."""
    return 0.5 * ((a + b) + jnp.sqrt((a - b) ** 2 + smth_coef ** 2))


def smooth_min_jax(a, b, smth_coef):
    """Smooth approximation to min(a, b) (advance_helper_module.F90:smooth_min).
    Complement of smooth_max: 0.5*((a+b) - sqrt((a-b)^2 + smth_coef^2))."""
    return 0.5 * ((a + b) - jnp.sqrt((a - b) ** 2 + smth_coef ** 2))


def pvertinterp(p_mid, p_out, input_var):
    """Interpolate input_var (on pressure levels p_mid) to a single output pressure p_out, per grid column
    (advance_helper_module.F90:pvertinterp). Pressure decreases with height (p_mid descending in k): below the
    lowest level (p_out >= surface pressure) it clamps to the bottom value, above the highest level (p_out <=
    top pressure) to the top value, and linearly interpolates in pressure between the bracketing levels — i.e.
    a pressure-weighted linear interpolation with constant-endpoint extrapolation.

    p_mid and input_var are (ngrdcol, nzt); p_out is a scalar. Returns (ngrdcol,). Pure-jnp → differentiable."""
    p_mid = jnp.asarray(p_mid, dtype=jnp.float64)
    input_var = jnp.asarray(input_var, dtype=jnp.float64)
    p_out = jnp.asarray(p_out, dtype=jnp.float64)
    # Reverse k so the pressure axis is ascending for jnp.interp (constant left/right extrapolation
    # reproduces the Fortran p_out>=p_mid[bottom] / p_out<=p_mid[top] clamps).
    xp = p_mid[:, ::-1]
    fp = input_var[:, ::-1]
    return jax.vmap(lambda x, f: jnp.interp(p_out, x, f))(xp, fp)


def calc_xpwp(Km_zm, xm, invrs_dzm):
    """Down-gradient eddy flux x'w' on momentum levels (advance_helper_module.F90:calc_xpwp_2D):
      xpwp[k] = Km_zm[k] * invrs_dzm[k] * (xm[k] - xm[k-1])  for the interior momentum levels k=1..nzm-2
      (0-based); the top and bottom boundary levels are left at zero.

    Km_zm and invrs_dzm are (ngrdcol, nzm); xm is (ngrdcol, nzt) with nzt = nzm-1. Pure-jnp → differentiable."""
    Km_zm = jnp.asarray(Km_zm, dtype=jnp.float64)
    xm = jnp.asarray(xm, dtype=jnp.float64)
    invrs_dzm = jnp.asarray(invrs_dzm, dtype=jnp.float64)
    ng, nzm = Km_zm.shape
    # Interior k=1..nzm-2 use thermo levels k and k-1: xm[:,1:nzm-1] - xm[:,0:nzm-2].
    interior = Km_zm[:, 1:nzm - 1] * invrs_dzm[:, 1:nzm - 1] * (xm[:, 1:] - xm[:, :-1])
    xpwp = jnp.zeros((ng, nzm))
    return xpwp.at[:, 1:nzm - 1].set(interior)


def calc_stability_correction_jax(
    brunt_vaisala_freq_sqd,
    Lscale_zm,
    em,
    clubb_params,
):
    """Stability correction factor (advance_helper_module.F90:calc_stability_correction)."""
    lambda0 = clubb_params[:, ilambda0_stability_coef - 1:ilambda0_stability_coef]
    lambda0_eff = jnp.where(brunt_vaisala_freq_sqd > 0.0, lambda0, 0.0)
    return 1.0 + jnp.minimum(lambda0_eff * brunt_vaisala_freq_sqd * Lscale_zm ** 2 / em, 3.0)


def _smooth_max_scalar_array(scalar, array):
    coef = min_max_smth_mag
    return 0.5 * ((scalar + array) + jnp.sqrt((scalar - array) ** 2 + coef ** 2))


def _smooth_min_scalar_array(scalar, array):
    coef = min_max_smth_mag
    return 0.5 * ((scalar + array) - jnp.sqrt((scalar - array) ** 2 + coef ** 2))


# ============================================================================
# Splat term  (advance_helper_module.F90:wp23_term_splat_lhs)
# ============================================================================

def _lscale_width_vert_avg_jax(var_profile, rho_ds_zm, below_grnd_val, gr):
    """Lscale_width_vert_avg with smth_type=2 (fixed half-width=60m)."""
    zm  = gr.zm
    dzm = gr.dzm
    half_width = _SMTH_TYPE2_HALF_WIDTH

    numer_terms = rho_ds_zm * gr.grid_dir * dzm * var_profile
    denom_terms = rho_ds_zm * gr.grid_dir * dzm

    zm_k = zm[:, :, None]
    zm_j = zm[:, None, :]
    dist = jnp.abs(zm_k - zm_j)
    # Include all levels j where |zm[k]-zm[j]| <= half_width.
    # Fortran Lscale_width_vert_avg (smth_type=2): the lower-bound loop ends
    # with k_avg_lower = k_lb_zm when zm[k]-zm[k_lb_zm] <= half_width, so the
    # lowest level (j=0) is included in the window for those k (no exclude_j0).
    in_window = dist <= half_width

    numer_sum = jnp.sum(numer_terms[:, None, :] * in_window, axis=2)
    denom_sum = jnp.sum(denom_terms[:, None, :] * in_window, axis=2)

    # Below-ground ghost levels: Fortran adds n_below ghost levels with value
    # below_grnd_val when zm[k] - zm[k_lb_zm] < half_width (window extends below ground).
    # n_below = int((half_width - (zm[k] - zm[0])) / dz_lb) for all k where > 0.
    dz_lb = zm[:, 1] - zm[:, 0]
    zm0   = zm[:, 0]
    below_dist = half_width - (zm - zm0[:, None])
    n_below = jnp.where(
        below_dist > 0.0,
        jnp.floor(below_dist / dz_lb[:, None]),
        0.0,
    )

    denom_at_lb = denom_terms[:, 0:1]
    numer_below = n_below * denom_at_lb * below_grnd_val
    denom_below = n_below * denom_at_lb

    # Guard the divisor (REFACTOR B5): a window always contains level k itself so the denominator is
    # physically nonzero, but a zero lane (or its reverse-mode 0/0) would poison the gradient — the
    # var_profile (Brunt-Vaisala freq sq) numerator is thlm-dependent. Forward-identical for denom!=0.
    denom = denom_sum + denom_below
    denom_safe = jnp.where(denom != 0.0, denom, 1.0)
    return (numer_sum + numer_below) / denom_safe


def wp23_term_splat_lhs_jax(
    brunt_vaisala_freq_sqd_mixed,
    Lscale_zm,
    rho_ds_zm,
    C_wp2_splat,
    below_grnd_val,
    gr,
):
    """wp23_term_splat_lhs: LHS coefficients for wp2 and wp3 splatting terms."""
    bv_sqd_splat = _lscale_width_vert_avg_jax(
        var_profile=brunt_vaisala_freq_sqd_mixed,
        rho_ds_zm=rho_ds_zm,
        below_grnd_val=below_grnd_val,
        gr=gr,
    )

    # _safe_sqrt (REFACTOR B5): bv_sqd_splat goes negative in unstable (convective) layers; the bare
    # jnp.sqrt(jnp.maximum(0,·)) has an inf reverse-mode gradient at the clip (it poisoned the bomex thlm
    # whole-driver grad). Forward-identical, finite (0) gradient where bv_sqd_splat<=0.
    brunt_freq_splat = _safe_sqrt(bv_sqd_splat)
    brunt_freq_splat_smooth = zm2zt2zm_jax(brunt_freq_splat, gr)
    brunt_freq_splat_zt = zm2zt_jax(brunt_freq_splat, gr)

    lhs_splat_wp2 = C_wp2_splat[:, None] * brunt_freq_splat_smooth
    lhs_splat_wp3 = 1.5 * C_wp2_splat[:, None] * brunt_freq_splat_zt

    return lhs_splat_wp2, lhs_splat_wp3, bv_sqd_splat


# ============================================================================
# Brunt-Väisälä frequency  (advance_helper_module.F90:calc_brunt_vaisala_freq_sqd)
# ============================================================================

def _sat_mixrat_liq_flatau_jax(p_in_Pa, T_in_K):
    """Saturation mixing ratio via Flatau polynomial (saturation_formula=3)."""
    T_in_C = jnp.maximum(T_in_K - _T_FREEZE_K, _MIN_T_IN_C)
    T_in_C_sqd = T_in_C ** 2

    esat = (
        -3.21582393e-14
        * (T_in_C - 646.5835252598777)
        * (T_in_C + 90.72381630364440)
        * (T_in_C_sqd + 111.0976961559954 * T_in_C + 6459.629194243118)
        * (T_in_C_sqd + 152.3131930092453 * T_in_C + 6499.774954705265)
        * (T_in_C_sqd + 174.4279584934021 * T_in_C + 7721.679732114084)
    )

    return ep * esat / (p_in_Pa - esat)


def calc_brunt_vaisala_freq_sqd_jax(
    thlm,
    exner,
    rtm,
    rcm,
    p_in_Pa,
    thvm,
    ice_supersat_frac,
    bv_efold,
    T0,
    l_use_thvm_in_bv_freq: bool,
    l_brunt_vaisala_freq_moist: bool,
    l_modify_limiters_for_cnvg_test: bool,
    gr,
):
    """Brunt-Vaisala frequency squared (advance_helper_module.F90:calc_brunt_vaisala_freq_sqd)."""
    ddzt_thlm = ddzt_jax(thlm, gr)

    if l_use_thvm_in_bv_freq:
        thvm_zm = zt2zm_jax(thvm, gr, zm_min=zero_threshold)
        ddzt_thvm = ddzt_jax(thvm, gr)
        bv_dry_main = (grav / thvm_zm) * ddzt_thvm
    else:
        bv_dry_main = (grav / T0) * ddzt_thlm

    T_in_K = thlm * exner + (Lv / Cp) * rcm
    T_in_K_zm = zt2zm_jax(T_in_K, gr, zm_min=zero_threshold)

    rsat     = _sat_mixrat_liq_flatau_jax(p_in_Pa, T_in_K)
    rsat_zm  = zt2zm_jax(rsat, gr, zm_min=zero_threshold)
    ddzt_rsat = ddzt_jax(rsat, gr)

    thm      = thlm + (Lv / (Cp * exner)) * rcm
    thm_zm   = zt2zm_jax(thm, gr, zm_min=zero_threshold)
    ddzt_thm = ddzt_jax(thm, gr)
    ddzt_rtm = ddzt_jax(rtm, gr)

    bv_dry = (grav / thm_zm) * ddzt_thm

    num_fac  = 1.0 + Lv * rsat_zm / (Rd * T_in_K_zm)
    den_fac  = 1.0 + ep * Lv ** 2 * rsat_zm / (Cp * Rd * T_in_K_zm ** 2)
    bv_moist = grav * (
        (num_fac / den_fac) * (ddzt_thm / thm_zm + (Lv / (Cp * T_in_K_zm)) * ddzt_rsat)
        - ddzt_rtm
    )

    ice_supersat_frac_zm = zt2zm_jax(ice_supersat_frac, gr, zm_min=zero_threshold)
    bv_mixed = bv_moist + jnp.exp(-bv_efold[:, None] * ice_supersat_frac_zm) * (bv_dry - bv_moist)

    if l_modify_limiters_for_cnvg_test:
        bv_smth = zm2zt2zm_jax(bv_mixed, gr)
    else:
        # Fortran zm2zt2zm does NOT apply a minimum clamp (oracle: advance_helper_module.F90 line 528).
        bv_clipped = jnp.minimum(bv_mixed, 1.0e8 * jnp.abs(bv_mixed) ** 3)
        bv_smth = zm2zt2zm_jax(bv_clipped, gr)

    if l_brunt_vaisala_freq_moist:
        brunt_vaisala_freq_sqd = bv_moist
    else:
        brunt_vaisala_freq_sqd = bv_dry_main

    return brunt_vaisala_freq_sqd, bv_mixed, bv_smth, bv_dry, bv_moist


def calculate_thlp2_rad_jax(rcm, thlprcp, radht, clubb_params, gr):
    """Compute radiative cooling contribution to thlp2 forcing.

    Faithful port of advance_helper_module.F90:calculate_thlp2_rad (lines 2170-2255).
    Only updates levels where rcm_zm > rc_tol.

    Args:
        rcm:          cloud water mixing ratio on zt levels  (ngrdcol, nzt)
        thlprcp:      thl'rc' on zm levels                   (ngrdcol, nzm)
        radht:        SW+LW heating rate on zt levels        (ngrdcol, nzt)
        clubb_params: tunable parameters                     (ngrdcol, 102)
        gr:           grid object (provides zt2zm weights)
    Returns:
        thlp2_rad increment (ngrdcol, nzm) — caller adds to thlp2_forcing
    """
    from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
    rcm_zm = zt2zm_jax(jnp.asarray(rcm), gr)
    radht_zm = zt2zm_jax(jnp.asarray(radht), gr)
    thlp2_rad_coef = clubb_params[:, ithlp2_rad_coef - 1][:, jnp.newaxis]
    # Double-where (REFACTOR B5): guard the divisor so the masked (rcm_zm<=rc_tol) lanes can't poison the
    # reverse-mode gradient with 0/0 — forward-identical since those lanes are discarded by the outer where.
    rcm_safe = jnp.where(rcm_zm > rc_tol, rcm_zm, 1.0)
    increment = jnp.where(
        rcm_zm > rc_tol,
        thlp2_rad_coef * 2.0 * radht_zm / rcm_safe * jnp.asarray(thlprcp),
        0.0,
    )
    return increment
