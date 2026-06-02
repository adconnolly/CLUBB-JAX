"""JAX port of KK_utilities.F90 — thermodynamic helpers for the KK microphysics.

So far: G_T_p, the drop-radius-growth coefficient G(T,p) [m^2/s] used in the KK
evaporation coefficient (KK_evap_coef = 3 C_evap G_T_p). From Rogers & Yau (1989)
Eqs. 7.17-7.18 / Khairoutdinov & Kogan (2000) Eq. 22. Oracle: KK_utilities.F90:169.

(The parabolic cylinder function Dv_fnc, also in KK_utilities.F90, lives in
parabolic_cylinder.py.)
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.saturation import (
    _sat_vapor_press_flatau_jax, _sat_vapor_press_bolton_jax,
    sat_mixrat_liq_jax, SATURATION_FLATAU, SATURATION_BOLTON,
)
from clubb_jax.src.CLUBB_core.constants_clubb import (
    Lv, Rv, Rd, Cp, T_freeze_K, rho_lw,
)


def G_T_p(T_in_K, p_in_Pa, saturation_formula=SATURATION_FLATAU):
    """G(T,p) [m^2/s], the drop-radius growth coefficient. KK_utilities.F90:169.

    Ka = thermal conductivity of air; Dv = vapor diffusion coefficient;
    Fk (heat-conduction term) and Fd (vapor-diffusion term) per Rogers & Yau 7.17-7.18;
    G = 1 / (Fk + Fd)."""
    T_in_K = jnp.asarray(T_in_K, dtype=jnp.float64)
    p_in_Pa = jnp.asarray(p_in_Pa, dtype=jnp.float64)
    Celsius = T_in_K - T_freeze_K

    # Thermal conductivity of air: cal/(cm s C) -> J/(m s K).
    Ka = (5.69 + 0.017 * Celsius) * 0.00001
    Ka = 4.1868 * 100.0 * Ka

    # Vapor diffusion coefficient: cm^2/s -> m^2/s.
    Dv = 0.221 * (T_in_K / 273.16) ** 1.94 * (101325.0 / p_in_Pa)
    Dv = Dv / 10000.0

    if saturation_formula == SATURATION_FLATAU:
        esatv = _sat_vapor_press_flatau_jax(T_in_K)
    elif saturation_formula == SATURATION_BOLTON:
        esatv = _sat_vapor_press_bolton_jax(T_in_K)
    else:
        raise ValueError(f"Unsupported saturation_formula={saturation_formula}")

    Fk = (Lv / (Rv * T_in_K) - 1.0) * (Lv * rho_lw) / (Ka * T_in_K)
    Fd = (rho_lw * Rv * T_in_K) / (Dv * esatv)
    return 1.0 / (Fk + Fd)


def kk_evap_coef(T_liq_in_K, p_in_Pa, C_evap, saturation_formula=SATURATION_FLATAU):
    """KK evaporation coefficient. KK_microphys_module.F90:1177.

    = 3 C_evap G_T_p ((4/3)pi rho_lw)^(2/3) (1 + Beta_Tl r_sl)/r_sl,
    Beta_Tl = (Rd/Rv)(Lv/(Rd T))(Lv/(Cp T)), r_sl = sat_mixrat_liq(p,T)."""
    T = jnp.asarray(T_liq_in_K, dtype=jnp.float64)
    p = jnp.asarray(p_in_Pa, dtype=jnp.float64)
    r_sl = sat_mixrat_liq_jax(p, T, saturation_formula)
    Beta_Tl = (Rd / Rv) * (Lv / (Rd * T)) * (Lv / (Cp * T))
    return (3.0 * C_evap * G_T_p(T, p, saturation_formula)
            * ((4.0 / 3.0) * jnp.pi * rho_lw) ** (2.0 / 3.0)
            * ((1.0 + Beta_Tl * r_sl) / r_sl))
