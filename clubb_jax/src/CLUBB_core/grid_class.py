"""Pure JAX implementations of grid interpolation and derivative operators.

Faithful port of CLUBB_core/grid_class.F90 routines:
  - zm2zt_jax:   momentum-level field → thermodynamic-level field
  - zt2zm_jax:   thermodynamic-level field → momentum-level field
  - ddzm_jax:    d/dz of momentum-level field evaluated at thermo levels
  - ddzt_jax:    d/dz of thermo-level field evaluated at momentum levels
  - zm2zt2zm_jax: zm→zt→zm smoother
  - zt2zm2zt_jax: zt→zm→zt smoother

Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level).

References:
  Golaz et al. (2002) JAS 59:3540–3551, Section 3c.
  src/CLUBB_core/grid_class.F90, linear_interpolated_azt_2D,
  linear_interpolated_azm_2D, gradzm_2D, gradzt_2D.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


def zm2zt_jax(azm: jnp.ndarray, gr) -> jnp.ndarray:
    """Interpolate a field from momentum (zm) levels to thermodynamic (zt) levels.

    Faithfully implements Fortran linear_interpolated_azt_2D.

    Formula for each thermo level k=0..nzt-1 (ascending grid):
        azt[k] = w_above * azm[k+1] + w_below * azm[k]
    where:
        w_above = (zt[k] - zm[k]) / (zm[k+1] - zm[k])   [weight for upper zm]
        w_below = (zm[k+1] - zt[k]) / (zm[k+1] - zm[k]) [weight for lower zm]

    Args:
        azm: Field on momentum levels, shape (ngrdcol, nzm).
        gr:  Grid object with .zm shape (ngrdcol, nzm) and .zt (ngrdcol, nzt).

    Returns:
        azt: Field on thermodynamic levels, shape (ngrdcol, nzt).
    """
    zm = gr.zm      # (ngrdcol, nzm)
    zt = gr.zt      # (ngrdcol, nzt)

    # dzt[k] = zm[k+1] - zm[k], shape (ngrdcol, nzt)
    dzt = zm[:, 1:] - zm[:, :-1]

    # Both weights computed DIRECTLY, matching Fortran calc_zm2zt_weights
    # (grid_class.F90:2621/2625) — NOT w_below = 1 - w_above. The two are
    # identical (0.5) on an evenly-spaced grid but differ by ~1 ULP on a
    # stretched grid (grid_type=2, e.g. rico), so the direct form is the
    # faithful one. azt[k] = w_above*azm[k+1] + w_below*azm[k].
    w_above = (zt - zm[:, :-1]) / dzt          # weights_zm2zt(m_above)
    w_below = (zm[:, 1:] - zt) / dzt           # weights_zm2zt(m_below)
    return w_above * azm[:, 1:] + w_below * azm[:, :-1]


def zt2zm_jax(azt: jnp.ndarray, gr, zm_min: float | None = None) -> jnp.ndarray:
    """Interpolate a field from thermodynamic (zt) levels to momentum (zm) levels.

    Faithfully implements Fortran linear_interpolated_azm_2D (ascending grid).

    Boundary conditions (ascending grid):
      k=0 (lower): linear extension using zt[0], zt[1] -- NOT a simple copy,
                   matches Fortran weights_zt2zm(1,*) which uses linear extension.
      k=nzm-1 (upper): linear extension using zt[nzt-2], zt[nzt-1].

    Interior k=1..nzm-2:
      azm[k] = w_above * azt[k] + (1-w_above) * azt[k-1]
    where w_above = (zm[k] - zt[k-1]) / (zt[k] - zt[k-1]).

    Args:
        azt: Field on thermodynamic levels, shape (ngrdcol, nzt).
        gr:  Grid with .zm (ngrdcol, nzm) and .zt (ngrdcol, nzt).
        zm_min: Optional lower clamp applied after interpolation.

    Returns:
        azm: Field on momentum levels, shape (ngrdcol, nzm).
    """
    zm = gr.zm   # (ngrdcol, nzm)
    zt = gr.zt   # (ngrdcol, nzt)
    nzm = zm.shape[1]

    # --- Interior levels k=1..nzm-2 ---
    # zt[k-1] < zm[k] < zt[k] for ascending grids
    # denom = zt[k] - zt[k-1], shape (ngrdcol, nzm-2)
    denom_int = zt[:, 1:] - zt[:, :-1]          # (ngrdcol, nzt-1) = (ngrdcol, nzm-2)
    zm_int = zm[:, 1:-1]                          # (ngrdcol, nzm-2)
    # Both weights DIRECT, matching Fortran calc_zt2zm_weights (grid_class.F90:
    # 2265/2269) — NOT w_below = 1 - w_above (identical on uniform grids, ~1 ULP
    # apart on stretched grids like rico). azm[k] = w_above*azt[k] + w_below*azt[k-1].
    w_above_int = (zm_int - zt[:, :-1]) / denom_int      # weights_zt2zm(t_above)
    w_below_int = (zt[:, 1:] - zm_int) / denom_int       # weights_zt2zm(t_below)
    azm_int = w_above_int * azt[:, 1:] + w_below_int * azt[:, :-1]

    # --- Lower boundary k=0 (ascending grid): linear extension below zt[0] ---
    # Fortran: azm(1) = azt(1) for ascending grid
    # The Fortran code sets azm(1) = azt(1) directly (not using weights).
    # Python 0-indexed: azm[0] = azt[0]
    azm_bot = azt[:, :1]   # shape (ngrdcol, 1)

    # --- Upper boundary k=nzm-1: linear extension above zt[nzt-1] ---
    # Fortran: azm(nzm) = w_zt2zm(nzm,t_above)*azt(nzt) + w_zt2zm(nzm,t_below)*azt(nzt-1)
    # where t_above weight = (zm(nzm) - zt(nzt-1)) / (zt(nzt) - zt(nzt-1))
    # Python 0-indexed:
    denom_top = zt[:, -1:] - zt[:, -2:-1]        # (ngrdcol, 1)
    w_above_top = (zm[:, -1:] - zt[:, -2:-1]) / denom_top   # weights_zt2zm(nzm,t_above)
    w_below_top = (zt[:, -1:] - zm[:, -1:]) / denom_top     # weights_zt2zm(nzm,t_below), direct
    azm_top = w_above_top * azt[:, -1:] + w_below_top * azt[:, -2:-1]

    azm = jnp.concatenate([azm_bot, azm_int, azm_top], axis=1)

    if zm_min is not None:
        azm = jnp.maximum(azm, zm_min)

    return azm


def ddzm_jax(azm: jnp.ndarray, gr) -> jnp.ndarray:
    """Vertical derivative of a momentum-level field, evaluated at thermo levels.

    Implements Fortran gradzm_2D:
        dazm_dz[k] = (azm[k+1] - azm[k]) * invrs_dzt[k]  for k=0..nzt-1

    Args:
        azm: Field on momentum levels, shape (ngrdcol, nzm).
        gr:  Grid with .invrs_dzt (ngrdcol, nzt).

    Returns:
        Result on thermodynamic levels, shape (ngrdcol, nzt).
    """
    return (azm[:, 1:] - azm[:, :-1]) * gr.invrs_dzt


def ddzt_jax(azt: jnp.ndarray, gr) -> jnp.ndarray:
    """Vertical derivative of a thermo-level field, evaluated at momentum levels.

    Implements Fortran gradzt_2D:
        Interior k=1..nzm-2: dazt_dz[k] = (azt[k] - azt[k-1]) * invrs_dzm[k]
        Boundary: dazt_dz[0] = dazt_dz[1]  (Fortran boundary condition)
                  dazt_dz[nzm-1] = dazt_dz[nzm-2]

    Args:
        azt: Field on thermodynamic levels, shape (ngrdcol, nzt).
        gr:  Grid with .invrs_dzm (ngrdcol, nzm).

    Returns:
        Result on momentum levels, shape (ngrdcol, nzm).
    """
    # Interior: (azt[k] - azt[k-1]) * invrs_dzm[k] for k=1..nzm-2
    interior = (azt[:, 1:] - azt[:, :-1]) * gr.invrs_dzm[:, 1:-1]  # (ngrdcol, nzm-2)

    # Boundary: replicate adjacent interior value
    bottom = interior[:, :1]   # same as k=1
    top = interior[:, -1:]     # same as k=nzm-2

    return jnp.concatenate([bottom, interior, top], axis=1)


def zm2zt2zm_jax(azm: jnp.ndarray, gr, zm_min: float | None = None) -> jnp.ndarray:
    """Smooth azm by mapping zm→zt→zm.

    Implements Fortran zm2zt2zm.
    """
    azt = zm2zt_jax(azm, gr)
    return zt2zm_jax(azt, gr, zm_min=zm_min)


def zt2zm2zt_jax(azt: jnp.ndarray, gr, zt_min: float | None = None) -> jnp.ndarray:
    """Smooth azt by mapping zt→zm→zt.

    Implements Fortran zt2zm2zt.
    """
    azm = zt2zm_jax(azt, gr)
    result = zm2zt_jax(azm, gr)
    if zt_min is not None:
        result = jnp.maximum(result, zt_min)
    return result


# JIT-compiled versions for production use
zm2zt = jit(zm2zt_jax)
zt2zm = jit(zt2zm_jax)
ddzm = jit(ddzm_jax)
ddzt = jit(ddzt_jax)
zm2zt2zm = jit(zm2zt2zm_jax)
zt2zm2zt = jit(zt2zm2zt_jax)


__all__ = [
    "zm2zt_jax",
    "zt2zm_jax",
    "ddzm_jax",
    "ddzt_jax",
    "zm2zt2zm_jax",
    "zt2zm2zt_jax",
    "zm2zt",
    "zt2zm",
    "ddzm",
    "ddzt",
    "zm2zt2zm",
    "zt2zm2zt",
]
