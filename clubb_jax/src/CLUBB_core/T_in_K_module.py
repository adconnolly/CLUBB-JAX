"""JAX port of `src/CLUBB_core/T_in_K_module.F90`."""

from __future__ import annotations

from clubb_jax.src.CLUBB_core.clubb_constants import Cp, Lv


def thlm2T_in_K(thlm, exner, rcm):
    """Absolute temperature from liquid-water potential temperature.

    Mirrors `thlm2T_in_K_api`:

      T_in_K = thlm * exner + Lv * rcm / Cp

    The Fortran routine is elemental; this JAX version works on matching
    scalar or array shapes.
    """
    return thlm * exner + Lv * rcm / Cp


def T_in_K2thlm(T_in_K, exner, rcm):
    """Liquid-water potential temperature from absolute temperature.

    Mirrors `T_in_K2thlm_api`:

      thlm = (T_in_K - Lv / Cp * rcm) / exner
    """
    return (T_in_K - Lv / Cp * rcm) / exner
