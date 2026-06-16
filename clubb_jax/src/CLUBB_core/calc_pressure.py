"""JAX port of selected routines from ``src/CLUBB_core/calc_pressure.F90``."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.clubb_constants import Cp, Lv, ep1, ep2


@partial(jax.jit, static_argnames=("nzt", "ngrdcol"))
def calculate_thvm(
    nzt: int,
    ngrdcol: int,
    thlm,
    rtm,
    rcm,
    exner,
    thv_ds_zt,
):
    """Calculates mean theta_v using the source linearized approximation."""
    del nzt, ngrdcol

    # Calculate mean theta_v
    thvm = (
        thlm
        + ep1 * thv_ds_zt * rtm
        + (Lv / (Cp * exner) - ep2 * thv_ds_zt) * rcm
    )

    return thvm
