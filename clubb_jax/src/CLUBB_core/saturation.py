"""JAX implementations of selected routines from `saturation.F90`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.clubb_constants import (
    Cp,
    Lv,
    T_freeze_K,
    ep,
    saturation_bolton,
    saturation_flatau,
    saturation_gfdl,
    saturation_lookup,
)

SATURATION_FLATAU = saturation_flatau
SATURATION_BOLTON = saturation_bolton
SATURATION_GFDL = saturation_gfdl
SATURATION_LOOKUP = saturation_lookup

_FLATAU_ICE_MIN_T_C = -90.0
_FLATAU_ICE_A = (
    100.0 * 6.09868993,
    100.0 * 0.499320233,
    100.0 * 0.184672631e-1,
    100.0 * 0.402737184e-3,
    100.0 * 0.565392987e-5,
    100.0 * 0.521693933e-7,
    100.0 * 0.307839583e-9,
    100.0 * 0.105785160e-11,
    100.0 * 0.161444444e-14,
)


def sat_vapor_press_liq_flatau(T_in_K):
    """Flatau et al. polynomial approximation for liquid saturation pressure.

    This mirrors the factored polynomial in `sat_mixrat_liq_k` for the
    `saturation_flatau` case.  The temperature is clipped at -85 C, matching
    the valid range limiter in the Fortran routine.
    """
    T_in_C = jnp.maximum(T_in_K - T_freeze_K, -85.0)
    T_in_C_sqd = T_in_C ** 2
    return (
        -3.21582393e-14
        * (T_in_C - 646.5835252598777)
        * (T_in_C + 90.72381630364440)
        * (T_in_C_sqd + 111.0976961559954 * T_in_C + 6459.629194243118)
        * (T_in_C_sqd + 152.3131930092453 * T_in_C + 6499.774954705265)
        * (T_in_C_sqd + 174.4279584934021 * T_in_C + 7721.679732114084)
    )


def sat_vapor_press_liq_bolton(T_in_K):
    """Bolton 1980 approximation for liquid saturation pressure."""
    return 611.2 * jnp.exp(17.67 * (T_in_K - T_freeze_K) / (T_in_K - 29.65))


def sat_vapor_press_liq_gfdl(T_in_K):
    """GFDL/Goff-Gratch approximation for liquid saturation pressure."""
    T_in_K_clipped = jnp.maximum(173.15, T_in_K)
    return (
        10.0 ** (
            -7.90298 * (373.16 / T_in_K_clipped - 1.0)
            + 5.02808 * jnp.log10(373.16 / T_in_K_clipped)
            - 1.3816e-7
            * (10.0 ** (11.344 * (1.0 - T_in_K_clipped / 373.16)) - 1.0)
            + 8.1328e-3
            * (10.0 ** (-3.49149 * (373.16 / T_in_K_clipped - 1.0)) - 1.0)
            + jnp.log10(1013.246)
        )
        * 100.0
    )


def sat_vapor_press_liq(T_in_K, saturation_formula: int):
    """Liquid saturation vapor pressure selected by `saturation_formula`.

    The Flatau, Bolton, and GFDL branches mirror `saturation.F90`.
    `saturation_lookup` depends on the Fortran lookup table and is left
    explicit here rather than silently using a different formula.
    """
    T_in_K = jnp.asarray(T_in_K, dtype=jnp.float64)
    if saturation_formula == saturation_flatau:
        return sat_vapor_press_liq_flatau(T_in_K)
    if saturation_formula == saturation_bolton:
        return sat_vapor_press_liq_bolton(T_in_K)
    if saturation_formula == saturation_gfdl:
        return sat_vapor_press_liq_gfdl(T_in_K)
    if saturation_formula == saturation_lookup:
        raise ValueError("saturation_lookup is not ported to JAX yet.")
    raise ValueError(f"Unsupported saturation_formula={saturation_formula}")


def sat_mixrat_liq(p_in_Pa, T_in_K, saturation_formula: int):
    """Saturation mixing ratio over liquid water.

    Mirrors `sat_mixrat_liq_api` with `I_sat_sphum = .false.`:

      rs = ep * esat / (p - esat)

    If `p - esat < 1`, the Fortran routine returns `ep`.
    """
    p_in_Pa = jnp.asarray(p_in_Pa, dtype=jnp.float64)
    T_in_K = jnp.asarray(T_in_K, dtype=jnp.float64)
    esat = sat_vapor_press_liq(T_in_K, saturation_formula)

    denom = p_in_Pa - esat
    safe = denom >= 1.0
    denom_safe = jnp.where(safe, denom, 1.0)
    return jnp.where(safe, ep * esat / denom_safe, ep)


def sat_mixrat_ice(p_in_Pa, T_in_K):
    """Saturation mixing ratio over ice using the Flatau polynomial."""
    p_in_Pa = jnp.asarray(p_in_Pa, dtype=jnp.float64)
    T_in_K = jnp.asarray(T_in_K, dtype=jnp.float64)
    T_in_C = jnp.maximum(T_in_K - T_freeze_K, _FLATAU_ICE_MIN_T_C)
    a = _FLATAU_ICE_A
    esat_ice = (
        a[0]
        + T_in_C
        * (
            a[1]
            + T_in_C
            * (
                a[2]
                + T_in_C
                * (
                    a[3]
                    + T_in_C
                    * (
                        a[4]
                        + T_in_C
                        * (
                            a[5]
                            + T_in_C
                            * (
                                a[6]
                                + T_in_C * (a[7] + T_in_C * a[8])
                            )
                        )
                    )
                )
            )
        )
    )
    denom = p_in_Pa - esat_ice
    safe = denom >= 1.0
    denom_safe = jnp.where(safe, denom, 1.0)
    return jnp.where(safe, ep * esat_ice / denom_safe, ep)


def rcm_sat_adj(thlm, rtm, p_in_Pa, exner, saturation_formula: int):
    """Diagnose liquid water from total water and liquid potential temperature."""
    thlm = jnp.asarray(thlm, dtype=jnp.float64)
    rtm = jnp.asarray(rtm, dtype=jnp.float64)
    p_in_Pa = jnp.asarray(p_in_Pa, dtype=jnp.float64)
    exner = jnp.asarray(exner, dtype=jnp.float64)

    theta_lo = thlm
    theta_hi = thlm + Lv / (Cp * exner) * jnp.maximum(rtm, 0.0)

    def residual(theta):
        rsat = sat_mixrat_liq(p_in_Pa, theta * exner, saturation_formula)
        rcm = jnp.maximum(rtm - rsat, 0.0)
        return theta - Lv / (Cp * exner) * rcm - thlm

    def step(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        use_upper = residual(mid) < 0.0
        lo = jnp.where(use_upper, mid, lo)
        hi = jnp.where(use_upper, hi, mid)
        return (lo, hi), None

    (theta_lo, theta_hi), _ = jax.lax.scan(
        step, (theta_lo, theta_hi), None, length=64
    )
    theta = 0.5 * (theta_lo + theta_hi)
    rsat = sat_mixrat_liq(p_in_Pa, theta * exner, saturation_formula)
    return jnp.maximum(rtm - rsat, 0.0)
