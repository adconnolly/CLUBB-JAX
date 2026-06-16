"""JAX port of selected routines from ``src/CLUBB_core/calc_pressure.F90``."""

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.clubb_constants import (
    Cp,
    Lv,
    ep1,
    ep2,
    grav,
    kappa,
    p0,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.grid_class import zt2zm


def _exner_step(exner_1, z_1, z_2, thvm_1, thvm_2):
    g_ov_cp = grav / Cp
    diff = thvm_2 - thvm_1
    tol = jnp.finfo(jnp.asarray(thvm_2).dtype).eps * thvm_2
    use_log = jnp.abs(diff) > tol
    safe_diff = jnp.where(use_log, diff, 1.0)
    safe_ratio = jnp.where(use_log, thvm_2 / thvm_1, 1.0)
    log_step = (
        exner_1
        - g_ov_cp * (z_2 - z_1) / safe_diff * jnp.log(safe_ratio)
    )
    constant_step = exner_1 - g_ov_cp * (z_2 - z_1) / thvm_2
    return jnp.where(use_log, log_step, constant_step)


def init_pressure(thvm, p_sfc, gr):
    """Calculate hydrostatic pressure and Exner on thermodynamic and momentum levels."""
    thvm = jnp.asarray(thvm, dtype=jnp.float64)
    p_sfc = jnp.asarray(p_sfc, dtype=jnp.float64)

    thvm_zm = zt2zm(
        gr.nzm,
        gr.nzt,
        gr.ngrdcol,
        gr,
        thvm,
        zero_threshold,
    )

    exner_sfc = (p_sfc / p0) ** kappa
    exner_first = _exner_step(
        exner_sfc,
        gr.zm[:, 0],
        gr.zt[:, 0],
        thvm_zm[:, 0],
        thvm[:, 0],
    )

    def zt_step(exner_prev, k):
        exner_next = _exner_step(
            exner_prev,
            gr.zt[:, k - 1],
            gr.zt[:, k],
            thvm[:, k - 1],
            thvm[:, k],
        )
        return exner_next, exner_next

    _, exner_rest = jax.lax.scan(
        zt_step,
        exner_first,
        jnp.arange(1, gr.nzt),
    )
    exner = jnp.concatenate([exner_first[None, :,], exner_rest], axis=0).T
    p_in_Pa = p0 * exner ** (1.0 / kappa)

    exner_zm_first = exner_sfc[:, None]
    exner_zm_rest = _exner_step(
        exner,
        gr.zt,
        gr.zm[:, 1:],
        thvm,
        thvm_zm[:, 1:],
    )
    exner_zm = jnp.concatenate([exner_zm_first, exner_zm_rest], axis=1)
    p_in_Pa_zm = p0 * exner_zm ** (1.0 / kappa)

    return p_in_Pa, exner, p_in_Pa_zm, exner_zm


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
