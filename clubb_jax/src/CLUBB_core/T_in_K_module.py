"""JAX implementations of thermodynamic routines from calc_pressure.F90."""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv, ep1, ep2


def calculate_thvm_jax(thlm, rtm, rcm, exner, thv_ds_zt):
    """Compute mean virtual potential temperature (calc_pressure.F90:calculate_thvm).

    thvm = thlm + ep1*thv_ds_zt*rtm + (Lv/(Cp*exner) - ep2*thv_ds_zt)*rcm
    """
    return thlm + ep1 * thv_ds_zt * rtm + (Lv / (Cp * exner) - ep2 * thv_ds_zt) * rcm
