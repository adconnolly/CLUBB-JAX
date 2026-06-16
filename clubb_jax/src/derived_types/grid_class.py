"""JAX-side representation of the CLUBB vertical grid derived type."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

core_rknd = np.float64

T_ABOVE = 0
T_BELOW = 1
M_ABOVE = 0
M_BELOW = 1

GRID_TYPE_EVEN = 1
GRID_TYPE_STRETCHED_ZT = 2
GRID_TYPE_STRETCHED_ZM = 3


_GRID_ARRAY_FIELDS = (
    "zm",
    "zt",
    "dzm",
    "dzt",
    "invrs_dzm",
    "invrs_dzt",
    "weights_zt2zm",
    "weights_zm2zt",
)

_GRID_STATIC_FIELDS = (
    "nzm",
    "nzt",
    "ngrdcol",
    "k_lb_zm",
    "k_ub_zm",
    "k_lb_zt",
    "k_ub_zt",
    "grid_dir_indx",
    "grid_dir",
)


@jax.tree_util.register_pytree_node_class
class Grid(NamedTuple):
    """Grid structure for CLUBB vertical discretization."""

    nzm: int
    nzt: int
    ngrdcol: int

    zm: jnp.ndarray
    zt: jnp.ndarray

    dzm: jnp.ndarray
    dzt: jnp.ndarray

    invrs_dzm: jnp.ndarray
    invrs_dzt: jnp.ndarray

    weights_zt2zm: jnp.ndarray
    weights_zm2zt: jnp.ndarray

    k_lb_zm: int
    k_ub_zm: int
    k_lb_zt: int
    k_ub_zt: int

    grid_dir_indx: int
    grid_dir: float

    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in _GRID_ARRAY_FIELDS)
        aux_data = tuple(getattr(self, name) for name in _GRID_STATIC_FIELDS)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data = dict(zip(_GRID_STATIC_FIELDS, aux_data))
        data.update(dict(zip(_GRID_ARRAY_FIELDS, children)))
        return cls(**data)


def setup_grid(
    ngrdcol: int,
    deltaz,
    zm_init,
    zm_top,
    l_ascending_grid: bool = True,
    grid_type: int = GRID_TYPE_EVEN,
    momentum_heights=None,
    thermodynamic_heights=None,
) -> Grid:
    """Construct a JAX `Grid` with CLUBB-style heights, spacings, and weights."""
    deltaz_arr = _as_column_input(deltaz, ngrdcol, "deltaz")
    zm_init_arr = _as_column_input(zm_init, ngrdcol, "zm_init")
    zm_top_arr = _as_column_input(zm_top, ngrdcol, "zm_top")

    if grid_type not in (
        GRID_TYPE_EVEN,
        GRID_TYPE_STRETCHED_ZT,
        GRID_TYPE_STRETCHED_ZM,
    ):
        raise ValueError(f"Unsupported grid_type={grid_type}.")
    if not l_ascending_grid:
        raise ValueError(
            "l_ascending_grid=False is not supported by the JAX grid operators."
        )

    if grid_type == GRID_TYPE_EVEN:
        nzm = int(
            np.floor(
                (zm_top_arr[0] - zm_init_arr[0] + deltaz_arr[0])
                / deltaz_arr[0]
            )
        )
        nzt = nzm - 1
        if nzm < 2 or nzt < 1:
            raise ValueError(
                f"Invalid derived grid dimensions: nzm={nzm}, nzt={nzt}."
            )
        zm, zt = _setup_even_grid(nzm, ngrdcol, deltaz_arr, zm_init_arr)
    elif grid_type == GRID_TYPE_STRETCHED_ZT:
        zt_in = _prepare_height_array(
            "thermodynamic_heights", thermodynamic_heights, ngrdcol
        )
        _validate_monotonic_increasing("thermodynamic_heights", zt_in)
        if np.any(zt_in[:, 0] <= zm_init_arr):
            raise ValueError(
                "Stretched zt grid lowest thermodynamic level must be above zm_init."
            )
        end_idx = int(np.searchsorted(zt_in[0], zm_top_arr[0], side="right") - 1)
        if end_idx < 0:
            raise ValueError("Stretched zt grid cannot fulfill zm_top requirement.")
        zt = zt_in[:, : end_idx + 1]
        nzt = zt.shape[1]
        nzm = nzt + 1
        zm = _calc_zm_from_zt(nzm, nzt, ngrdcol, zt, zm_init_arr)
    else:
        zm_in = _prepare_height_array("momentum_heights", momentum_heights, ngrdcol)
        _validate_monotonic_increasing("momentum_heights", zm_in)
        begin_idx = int(np.searchsorted(zm_in[0], zm_init_arr[0], side="left"))
        end_idx = int(np.searchsorted(zm_in[0], zm_top_arr[0], side="right") - 1)
        if begin_idx >= zm_in.shape[1]:
            raise ValueError("Stretched zm grid cannot fulfill zm_init requirement.")
        if end_idx < begin_idx:
            raise ValueError("Stretched zm grid cannot fulfill zm_top requirement.")
        zm = zm_in[:, begin_idx : end_idx + 1]
        nzm = zm.shape[1]
        nzt = nzm - 1
        if nzm < 2 or nzt < 1:
            raise ValueError(
                f"Stretched zm grid produced invalid dimensions: nzm={nzm}, nzt={nzt}."
            )
        zt = 0.5 * (zm[:, :-1] + zm[:, 1:])

    dzm, dzt, invrs_dzm, invrs_dzt = _calc_grid_spacings(
        nzm, ngrdcol, zm, zt
    )
    weights_zt2zm = _calc_zt2zm_weights(nzm, nzt, ngrdcol, zm, zt)
    weights_zm2zt = _calc_zm2zt_weights(nzt, ngrdcol, zm, zt, dzt)

    return Grid(
        nzm=nzm,
        nzt=nzt,
        ngrdcol=ngrdcol,
        zm=jnp.asarray(zm),
        zt=jnp.asarray(zt),
        dzm=jnp.asarray(dzm),
        dzt=jnp.asarray(dzt),
        invrs_dzm=jnp.asarray(invrs_dzm),
        invrs_dzt=jnp.asarray(invrs_dzt),
        weights_zt2zm=jnp.asarray(weights_zt2zm),
        weights_zm2zt=jnp.asarray(weights_zm2zt),
        k_lb_zm=0,
        k_ub_zm=nzm - 1,
        k_lb_zt=0,
        k_ub_zt=nzt - 1,
        grid_dir_indx=1,
        grid_dir=1.0,
    )


def _as_column_input(value, ngrdcol: int, name: str):
    if isinstance(value, (int, float)):
        return np.full(ngrdcol, value, dtype=core_rknd)
    arr = np.asarray(value, dtype=core_rknd)
    if arr.shape != (ngrdcol,):
        raise ValueError(f"{name} must have shape ({ngrdcol},), got {arr.shape}.")
    return arr


def _prepare_height_array(name: str, heights, ngrdcol: int):
    if heights is None:
        raise ValueError(f"{name} must be provided for stretched grid setup.")
    arr = np.asarray(heights, dtype=core_rknd)
    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (ngrdcol, 1))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")
    if arr.shape[0] != ngrdcol:
        raise ValueError(
            f"{name} first dimension must match ngrdcol={ngrdcol}, got {arr.shape[0]}."
        )
    return arr


def _validate_monotonic_increasing(name: str, arr) -> None:
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one level.")
    if np.any(np.diff(arr, axis=1) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing with level index.")


def _setup_even_grid(nzm: int, ngrdcol: int, deltaz, zm_init):
    k_indices = np.arange(nzm, dtype=core_rknd)
    zm = zm_init[:, None] + deltaz[:, None] * k_indices[None, :]
    zt = 0.5 * (zm[:, :-1] + zm[:, 1:])
    return zm, zt


def _calc_zm_from_zt(nzm: int, nzt: int, ngrdcol: int, zt, zm_init):
    zm = np.zeros((ngrdcol, nzm), dtype=core_rknd)
    zm[:, 1:-1] = 0.5 * (zt[:, :-1] + zt[:, 1:])
    zm[:, 0] = np.asarray(zm_init, dtype=core_rknd)
    if nzt > 1:
        zm[:, -1] = zt[:, -1] + 0.5 * (zt[:, -1] - zt[:, -2])
    else:
        zm[:, -1] = zt[:, -1] + (zt[:, -1] - zm[:, 0])
    return zm


def _calc_grid_spacings(nzm: int, ngrdcol: int, zm, zt):
    dzm = np.zeros((ngrdcol, nzm), dtype=core_rknd)
    dzt = zm[:, 1:] - zm[:, :-1]

    dzm[:, 1:-1] = zt[:, 1:] - zt[:, :-1]
    dzm[:, 0] = 2.0 * (zt[:, 0] - zm[:, 0])
    dzm[:, -1] = dzm[:, -2]

    invrs_dzm = np.where(np.abs(dzm) > 1e-30, 1.0 / dzm, 0.0)
    invrs_dzt = np.where(np.abs(dzt) > 1e-30, 1.0 / dzt, 0.0)
    return dzm, dzt, invrs_dzm, invrs_dzt


def _calc_zt2zm_weights(nzm: int, nzt: int, ngrdcol: int, zm, zt):
    weights = np.zeros((ngrdcol, nzm, 2), dtype=core_rknd)

    for k in range(1, nzm - 1):
        denom = zt[:, k] - zt[:, k - 1]
        weights[:, k, T_ABOVE] = (zm[:, k] - zt[:, k - 1]) / (denom + 1e-30)
        weights[:, k, T_BELOW] = (zt[:, k] - zm[:, k]) / (denom + 1e-30)

    if nzt >= 2:
        denom0 = zt[:, 1] - zt[:, 0]
        weights[:, 0, T_ABOVE] = (zm[:, 0] - zt[:, 0]) / (denom0 + 1e-30)
        weights[:, 0, T_BELOW] = (zt[:, 1] - zm[:, 0]) / (denom0 + 1e-30)

        denomn = zt[:, -1] - zt[:, -2]
        weights[:, -1, T_ABOVE] = (zm[:, -1] - zt[:, -2]) / (denomn + 1e-30)
        weights[:, -1, T_BELOW] = (zt[:, -1] - zm[:, -1]) / (denomn + 1e-30)
    else:
        weights[:, 0, T_ABOVE] = 1.0
        weights[:, -1, T_ABOVE] = 1.0

    return weights


def _calc_zm2zt_weights(nzt: int, ngrdcol: int, zm, zt, dzt):
    weights = np.zeros((ngrdcol, nzt, 2), dtype=core_rknd)
    for k in range(nzt):
        total_dist = dzt[:, k] + 1e-30
        weights[:, k, M_ABOVE] = (zt[:, k] - zm[:, k]) / total_dist
        weights[:, k, M_BELOW] = (zm[:, k + 1] - zt[:, k]) / total_dist
    return weights


__all__ = [
    "Grid",
    "setup_grid",
    "GRID_TYPE_EVEN",
    "GRID_TYPE_STRETCHED_ZT",
    "GRID_TYPE_STRETCHED_ZM",
]
