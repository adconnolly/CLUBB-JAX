"""JAX implementations of saturation.F90 routines."""

import jax
import jax.numpy as jnp

# Physical constants for rcm_sat_adj
_Cp = 1004.67
_Lv = 2.5e6

# Constants matching CLUBB Fortran values
_T_FREEZE_K = 273.15    # Freezing point of water [K]
_EP = 287.04 / 461.5    # Rd / Rv ≈ 0.622

# Saturation formula integer codes (model_flags.F90)
SATURATION_BOLTON = 1
SATURATION_FLATAU = 3

# Flatau polynomial minimum temperature [C]
_FLATAU_MIN_T_C = -85.0

# Flatau ice polynomial coefficients (Table 4, Flatau et al. 1992, pp.1511)
# multiplied by 100 as in Fortran saturation.F90
_FLATAU_ICE_MIN_T_C = -90.0
_FLATAU_ICE_A = [
    100.0 * 6.09868993,
    100.0 * 0.499320233,
    100.0 * 0.184672631e-1,
    100.0 * 0.402737184e-3,
    100.0 * 0.565392987e-5,
    100.0 * 0.521693933e-7,
    100.0 * 0.307839583e-9,
    100.0 * 0.105785160e-11,
    100.0 * 0.161444444e-14,
]


def _sat_vapor_press_flatau_jax(T_in_K):
    """Flatau et al. 1992 8th-order polynomial SVP over liquid water.

    Valid range: -85 to ~50 deg_C.
    Returns esat [Pa].
    """
    T_in_C = jnp.clip(T_in_K - _T_FREEZE_K, a_min=_FLATAU_MIN_T_C)
    T_sqd = T_in_C ** 2
    esat = (
        -3.21582393e-14
        * (T_in_C - 646.5835252598777)
        * (T_in_C + 90.72381630364440)
        * (T_sqd + 111.0976961559954 * T_in_C + 6459.629194243118)
        * (T_sqd + 152.3131930092453 * T_in_C + 6499.774954705265)
        * (T_sqd + 174.4279584934021 * T_in_C + 7721.679732114084)
    )
    return esat


def _sat_vapor_press_bolton_jax(T_in_K):
    """Bolton 1980 SVP over liquid water. Returns esat [Pa]."""
    return 611.2 * jnp.exp(17.67 * (T_in_K - _T_FREEZE_K) / (T_in_K - 29.65))


def sat_mixrat_liq_jax(p_in_Pa, T_in_K, saturation_formula: int):
    """Saturation mixing ratio over liquid water (saturation.F90:sat_mixrat_liq_api).

    Uses I_sat_sphum = .false. (mixing ratio, not specific humidity).

    rsat = ep * esat / (p_in_Pa - esat)   when p_in_Pa - esat >= 1.0
         = ep                              otherwise (esat ≈ pressure)

    Args:
        p_in_Pa:            Pressure [Pa], any shape.
        T_in_K:             Temperature [K], same shape.
        saturation_formula: SATURATION_FLATAU=3 or SATURATION_BOLTON=1.

    Returns:
        rsat: saturation mixing ratio [kg/kg], same shape as inputs.
    """
    if saturation_formula == SATURATION_FLATAU:
        esat = _sat_vapor_press_flatau_jax(T_in_K)
    elif saturation_formula == SATURATION_BOLTON:
        esat = _sat_vapor_press_bolton_jax(T_in_K)
    else:
        raise ValueError(f"Unsupported saturation_formula={saturation_formula}")

    # Protect against p - esat < 1 (esat ≥ pressure — near pure vapor)
    safe = (p_in_Pa - esat) >= 1.0
    rsat = jnp.where(safe, _EP * esat / (p_in_Pa - esat), _EP)
    return rsat


def sat_mixrat_ice_jax(p_in_Pa, T_in_K):
    """Saturation mixing ratio over ice via Flatau polynomial (saturation_formula=3).

    Matches Fortran saturation.F90:sat_mixrat_ice_2D with saturation_flatau.

    Args:
        p_in_Pa: Pressure [Pa], any shape.
        T_in_K:  Temperature [K], same shape.

    Returns:
        rsat_ice: saturation mixing ratio over ice [kg/kg].
    """
    T_in_C = jnp.clip(T_in_K - _T_FREEZE_K, a_min=_FLATAU_ICE_MIN_T_C)
    a = _FLATAU_ICE_A
    esat_ice = (a[0] + T_in_C * (a[1] + T_in_C * (a[2] + T_in_C * (
                a[3] + T_in_C * (a[4] + T_in_C * (a[5] + T_in_C * (
                a[6] + T_in_C * (a[7] + T_in_C * a[8]))))))))
    safe = (p_in_Pa - esat_ice) >= 1.0
    return jnp.where(safe, _EP * esat_ice / (p_in_Pa - esat_ice), _EP)


def rcm_sat_adj_jax(thlm, rtm, p_in_Pa, exner, saturation_formula: int):
    """Bisection saturation adjustment for cloud water (saturation.F90:rcm_sat_adj).

    Finds theta satisfying:
      theta - (Lv/(Cp*exner)) * max(rtm - rsat(p, theta*exner), 0) = thlm
    then returns rcm = max(rtm - rsat(p, theta*exner), 0).

    Works for any array shape. Uses 100 bisection iterations (enough for 0.001 K
    tolerance from an initial bracket of ~20 K).

    Args:
        thlm:               liquid potential temperature [K]
        rtm:                total water mixing ratio [kg/kg]
        p_in_Pa:            pressure [Pa]
        exner:              Exner function [-]
        saturation_formula: SATURATION_FLATAU=3 or SATURATION_BOLTON=1

    Returns:
        rcm: cloud water mixing ratio [kg/kg], same shape as inputs.
    """
    _tol = 0.001  # Fortran tolerance [K]
    _Lv_Cp_exner = _Lv / (_Cp * exner)

    def _answer(theta):
        T = theta * exner
        rsat = sat_mixrat_liq_jax(p_in_Pa, T, saturation_formula)
        rc = jnp.maximum(rtm - rsat, 0.0)
        return theta - _Lv_Cp_exner * rc

    # --- First iteration (Fortran special case: always sets too_high = theta + 20) ---
    theta0 = thlm
    answer0 = _answer(theta0)
    diff0 = answer0 - thlm

    too_low0 = jnp.where(diff0 < -_tol, theta0, jnp.zeros_like(thlm))
    # Fortran: if iteration==1 then too_high = theta + 20 (overrides any too_high update)
    too_high0 = theta0 + 20.0
    # But also apply normal update if answer > thlm+tol
    too_high0 = jnp.where(diff0 > _tol, theta0, too_high0)

    converged0 = jnp.abs(diff0) <= _tol
    theta1 = jnp.where(converged0, theta0, (too_low0 + too_high0) / 2.0)

    # --- Subsequent bisection steps ---
    def _bisect_step(carry, _):
        theta, too_low, too_high, converged = carry
        answer = _answer(theta)
        diff = answer - thlm

        new_too_high = jnp.where(diff > _tol, theta, too_high)
        new_too_low  = jnp.where(diff < -_tol, theta, too_low)
        new_conv = converged | (jnp.abs(diff) <= _tol)
        new_theta_next = (new_too_low + new_too_high) / 2.0

        # Freeze converged elements
        new_theta    = jnp.where(new_conv, theta, new_theta_next)
        new_too_low  = jnp.where(converged, too_low,  new_too_low)
        new_too_high = jnp.where(converged, too_high, new_too_high)
        return (new_theta, new_too_low, new_too_high, new_conv), None

    state0 = (theta1, too_low0, too_high0, converged0)
    (theta_final, _, _, _), _ = jax.lax.scan(_bisect_step, state0, None, length=100)

    T_final = theta_final * exner
    rsat_final = sat_mixrat_liq_jax(p_in_Pa, T_final, saturation_formula)
    return jnp.maximum(rtm - rsat_final, 0.0)
