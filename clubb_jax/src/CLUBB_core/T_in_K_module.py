"""JAX implementation of T_in_K_module.F90 — absolute temperature from thlm.

(`calculate_thvm` lives in calc_pressure.py, mirroring its Fortran home calc_pressure.F90.)
"""

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv


def thlm2T_in_K(thlm, exner, rcm):
    """Absolute temperature from liquid-water potential temperature (T_in_K_module.F90:thlm2T_in_K).

    T_in_K = thlm * exner + Lv * rcm / Cp
    Elemental in the Fortran — works on any matching shape. Bit-validated vs `f2py_thlm2t_in_k_1d`.
    """
    return thlm * exner + Lv * rcm / Cp


def T_in_K2thlm(T_in_K, exner, rcm):
    """Liquid-water potential temperature from absolute temperature (T_in_K_module.F90:T_in_K2thlm_api).

    thlm = ( T_in_K - Lv/Cp * rcm ) / exner   — the exact inverse of thlm2T_in_K.
    Elemental in the Fortran (public via clubb_api_module; the Fortran name carries the `_api` suffix).
    """
    return (T_in_K - Lv / Cp * rcm) / exner
