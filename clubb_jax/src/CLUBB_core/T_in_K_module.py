"""JAX implementations of thermodynamic routines from calc_pressure.F90 and T_in_K_module.F90."""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv, ep1, ep2


def thlm2T_in_K_jax(thlm, exner, rcm):
    """Absolute temperature from liquid-water potential temperature (T_in_K_module.F90:thlm2T_in_K).

    T_in_K = thlm * exner + Lv * rcm / Cp
    Elemental in the Fortran — works on any matching shape. Bit-validated vs `f2py_thlm2t_in_k_1d`.
    """
    return thlm * exner + Lv * rcm / Cp


def calculate_thvm_jax(thlm, rtm, rcm, exner, thv_ds_zt):
    """Compute mean virtual potential temperature (calc_pressure.F90:calculate_thvm).

    thvm = thlm + ep1*thv_ds_zt*rtm + (Lv/(Cp*exner) - ep2*thv_ds_zt)*rcm
    """
    return thlm + ep1 * thv_ds_zt * rtm + (Lv / (Cp * exner) - ep2 * thv_ds_zt) * rcm
