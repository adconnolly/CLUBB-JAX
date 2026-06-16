"""JAX implementations of selected `grid_class.F90` helpers."""

from __future__ import annotations

import jax.numpy as jnp

T_ABOVE = 0
T_BELOW = 1
M_ABOVE = 0
M_BELOW = 1


def zt2zm(nzm: int, nzt: int, ngrdcol: int, gr, azt, zm_min=None):
    """Interpolate a thermo-level field to momentum levels."""
    azt = jnp.asarray(azt, dtype=jnp.float64)

    interior = (
        gr.weights_zt2zm[:, 1:nzm - 1, T_ABOVE] * azt[:, 1:nzt]
        + gr.weights_zt2zm[:, 1:nzm - 1, T_BELOW] * azt[:, :nzt - 1]
    )

    lower_ascending = azt[:, :1]
    upper_ascending = (
        gr.weights_zt2zm[:, nzm - 1:nzm, T_ABOVE] * azt[:, nzt - 1:nzt]
        + gr.weights_zt2zm[:, nzm - 1:nzm, T_BELOW] * azt[:, nzt - 2:nzt - 1]
    )
    lower_descending = (
        gr.weights_zt2zm[:, :1, T_ABOVE] * azt[:, 1:2]
        + gr.weights_zt2zm[:, :1, T_BELOW] * azt[:, :1]
    )
    upper_descending = azt[:, nzt - 1:nzt]

    is_ascending = gr.grid_dir_indx == 1
    lower = jnp.where(is_ascending, lower_ascending, lower_descending)
    upper = jnp.where(is_ascending, upper_ascending, upper_descending)

    azm = jnp.concatenate([lower, interior, upper], axis=1)
    if zm_min is not None:
        azm = jnp.maximum(azm, zm_min)
    return azm


def zm2zt(nzm: int, nzt: int, ngrdcol: int, gr, azm, zt_min=None):
    """Interpolate a momentum-level field to thermo levels."""
    azm = jnp.asarray(azm, dtype=jnp.float64)
    azt = (
        gr.weights_zm2zt[:, :, M_ABOVE] * azm[:, 1:nzm]
        + gr.weights_zm2zt[:, :, M_BELOW] * azm[:, :nzt]
    )
    if zt_min is not None:
        azt = jnp.maximum(azt, zt_min)
    return azt


def zt2zm2zt(nzm: int, nzt: int, ngrdcol: int, gr, azt, zt_min=None):
    """Smooth a thermo-level field by mapping zt -> zm -> zt."""
    return zm2zt(nzm, nzt, ngrdcol, gr, zt2zm(nzm, nzt, ngrdcol, gr, azt), zt_min)


def zm2zt2zm(nzm: int, nzt: int, ngrdcol: int, gr, azm, zm_min=None):
    """Smooth a momentum-level field by mapping zm -> zt -> zm."""
    return zt2zm(nzm, nzt, ngrdcol, gr, zm2zt(nzm, nzt, ngrdcol, gr, azm), zm_min)


def ddzm(nzm: int, nzt: int, ngrdcol: int, gr, azm):
    """Differentiate a momentum-level field across thermo levels."""
    azm = jnp.asarray(azm, dtype=jnp.float64)
    return (azm[:, 1:nzm] - azm[:, :nzt]) * gr.invrs_dzt


def ddzt(nzm: int, nzt: int, ngrdcol: int, gr, azt):
    """Differentiate a thermo-level field across momentum levels."""
    azt = jnp.asarray(azt, dtype=jnp.float64)
    interior = (azt[:, 1:nzt] - azt[:, :nzt - 1]) * gr.invrs_dzm[:, 1:nzm - 1]
    return jnp.concatenate([interior[:, :1], interior, interior[:, -1:]], axis=1)


__all__ = [
    "zt2zm",
    "zm2zt",
    "zt2zm2zt",
    "zm2zt2zm",
    "ddzm",
    "ddzt",
]
