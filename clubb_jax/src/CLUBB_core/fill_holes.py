"""JAX port of exposed routines from `src/CLUBB_core/fill_holes.F90`.

This module ports the vertical hole-filling wrapper used by the active JAX
`advance_xm_wpxp` and `advance_xp2_xpyp` paths, plus the wp2/TKE helper exposed
by the Python API. Python callers pass zero-based `lower_hf_level` and
`upper_hf_level`, matching `clubb_python.CLUBB_core.fill_holes`.

The source Fortran notes that the lowest level should not be included because
hole filling should not alter the set surface value for momentum-level fields
or consider a below-surface thermodynamic level. Momentum-level calls should
also exclude the upper boundary level.

TODO(JAX port):
  Only `global_fill` and `sliding_window` are currently implemented. The
  widening, smart-window, smart-window-smooth, and parallel-fill methods remain
  unported.
"""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.clubb_constants import (
    eps,
    global_fill,
    num_hf_draw_points,
    one,
    sliding_window,
    zero,
)

_F64_EPS = jnp.finfo(jnp.float64).eps


@partial(
    jax.jit,
    static_argnames=("nz", "ngrdcol"),
)
def fill_holes_global(
    nz: int,
    ngrdcol: int,
    threshold: float,
    lower_hf_level: int,
    upper_hf_level: int,
    dz,
    rho_ds,
    field,
):
    """Fill holes using the whole range as the fill window.

    This is maximally effective and computationally cheap, but minimally local.
    Mass is conserved by reducing the clipped field everywhere by a constant
    multiplicative coefficient. This routine does not guarantee that the clipped
    field will exceed threshold everywhere; blunt clipping is needed for that.
    """
    del ngrdcol
    k_idx = jnp.arange(nz)[None, :]
    in_range = (k_idx >= lower_hf_level) & (k_idx <= upper_hf_level)
    rho_ds_dz = rho_ds * dz

    numer_integral_global = jnp.sum(jnp.where(in_range, rho_ds_dz * field, 0.0), axis=1, keepdims=True)
    denom_integral_global = jnp.sum(jnp.where(in_range, rho_ds_dz, 0.0), axis=1, keepdims=True)

    # Find the vertical average of field, using the precomputed numerator and
    # denominator. See the description of vertical_avg in advance_helper_module.
    field_avg_global = numer_integral_global / denom_integral_global

    # Clip small or negative values from field.
    field_clipped = jnp.where(
        field_avg_global >= threshold,
        jnp.maximum(threshold, field),
        jnp.minimum(threshold, field),
    )

    numer_integral_clipped = jnp.sum(
        jnp.where(in_range, rho_ds_dz * field_clipped, 0.0),
        axis=1,
        keepdims=True,
    )
    field_clipped_avg = numer_integral_clipped / denom_integral_global

    safe_to_scale = (
        jnp.abs(field_clipped_avg - threshold)
        > jnp.abs(field_clipped_avg + threshold) * eps / 2.0
    )
    mass_fraction_global = (field_avg_global - threshold) / (
        field_clipped_avg - threshold
    )
    field_filled = threshold + mass_fraction_global * (field_clipped - threshold)

    # Do not update this column if there are no holes in the fill range, or if
    # field_clipped_avg ~= threshold.
    any_hole = jnp.any(jnp.where(in_range, field < threshold, False), axis=1, keepdims=True)
    apply_fill = in_range & any_hole & safe_to_scale
    return jnp.where(apply_fill, field_filled, field)


@partial(
    jax.jit,
    static_argnames=("nz", "ngrdcol"),
)
def fill_holes_sliding_window(
    nz: int,
    ngrdcol: int,
    threshold: float,
    lower_hf_level: int,
    upper_hf_level: int,
    dz,
    rho_ds,
    field,
):
    """Fill holes with a sliding-window technique and global fallback.

    This modifies consecutive vertical ranges in serial. It is computationally
    expensive but highly local. The locality has a tradeoff with effectiveness,
    so this falls back to a global fill if the first pass leaves holes.

    References:
      "Numerical Methods for Wave Equations in Geophysical Fluid Dynamics",
      Durran (1999), p. 292.
    """
    del nz
    rho_ds_dz = rho_ds * dz
    window_len = 2 * num_hf_draw_points + 1
    start_indx = lower_hf_level + num_hf_draw_points
    stop_indx = upper_hf_level - num_hf_draw_points + 1

    def fill_one_window(k, field_carry):
        k_start = k - num_hf_draw_points
        field_window = jax.lax.dynamic_slice(
            field_carry,
            (0, k_start),
            (ngrdcol, window_len),
        )
        rho_window = jax.lax.dynamic_slice(
            rho_ds_dz,
            (0, k_start),
            (ngrdcol, window_len),
        )

        invrs_denom_integral = one / jnp.sum(rho_window, axis=1, keepdims=True)
        field_avg = jnp.sum(rho_window * field_window, axis=1, keepdims=True) * invrs_denom_integral

        field_clipped = jnp.where(
            field_avg >= threshold,
            jnp.maximum(threshold, field_window),
            jnp.minimum(threshold, field_window),
        )
        field_clipped_avg = (
            jnp.sum(rho_window * field_clipped, axis=1, keepdims=True)
            * invrs_denom_integral
        )

        safe_to_scale = (
            jnp.abs(field_clipped_avg - threshold)
            > jnp.abs(field_clipped_avg + threshold) * eps / 2.0
        )
        mass_fraction = (field_avg - threshold) / (field_clipped_avg - threshold)
        field_window_filled = threshold + mass_fraction * (field_clipped - threshold)

        any_hole = jnp.any(field_window < threshold, axis=1, keepdims=True)
        field_window_out = jnp.where(
            any_hole & safe_to_scale,
            field_window_filled,
            field_window,
        )
        return jax.lax.dynamic_update_slice(field_carry, field_window_out, (0, k_start))

    field = jax.lax.fori_loop(start_indx, stop_indx, fill_one_window, field)

    # If the first sliding-window pass did not work, fall back to global fill.
    return jax.lax.cond(
        jnp.any(field < threshold),
        lambda f: fill_holes_global(
            nz=field.shape[1],
            ngrdcol=ngrdcol,
            threshold=threshold,
            lower_hf_level=lower_hf_level,
            upper_hf_level=upper_hf_level,
            dz=dz,
            rho_ds=rho_ds,
            field=f,
        ),
        lambda f: f,
        field,
    )


@partial(
    jax.jit,
    static_argnames=(
        "nz",
        "ngrdcol",
        "grid_dir_indx",
        "fill_holes_type",
    ),
)
def fill_holes_vertical(
    nz: int,
    ngrdcol: int,
    threshold: float,
    lower_hf_level: int,
    upper_hf_level: int,
    dz,
    rho_ds,
    grid_dir_indx: int,
    fill_holes_type: int,
    field,
):
    """Call a vertical hole-filling method selected by `fill_holes_type`.

    The lowest level should not be included because hole filling should not
    alter the set surface value for momentum-level variables or consider a
    below-surface thermodynamic level. Momentum-level calls should also exclude
    the upper boundary level.
    """
    if grid_dir_indx not in (1, -1):
        raise ValueError(f"Unsupported grid_dir_indx={grid_dir_indx}")

    if grid_dir_indx == -1:
        field_reversed = jnp.flip(field, axis=1)
        dz_reversed = jnp.flip(dz, axis=1)
        rho_ds_reversed = jnp.flip(rho_ds, axis=1)
        lower_reversed = nz - 1 - lower_hf_level
        upper_reversed = nz - 1 - upper_hf_level
        filled = fill_holes_vertical(
            nz,
            ngrdcol,
            threshold,
            lower_reversed,
            upper_reversed,
            dz_reversed,
            rho_ds_reversed,
            1,
            fill_holes_type,
            field_reversed,
        )
        return jnp.flip(filled, axis=1)

    # Only bother with a fill call if there are values below threshold.
    if fill_holes_type == global_fill:
        filled = fill_holes_global(
            nz,
            ngrdcol,
            threshold,
            lower_hf_level,
            upper_hf_level,
            dz,
            rho_ds,
            field,
        )
    elif fill_holes_type == sliding_window:
        filled = fill_holes_sliding_window(
            nz,
            ngrdcol,
            threshold,
            lower_hf_level,
            upper_hf_level,
            dz,
            rho_ds,
            field,
        )
    else:
        # TODO(JAX port): port the remaining Fortran fill_holes_type options
        # rather than routing them through a Python/API fallback.
        raise NotImplementedError(
            "JAX fill_holes_vertical currently supports global_fill and "
            f"sliding_window; got fill_holes_type={fill_holes_type}."
        )

    return jnp.where(jnp.any(field < threshold), filled, field)


@partial(
    jax.jit,
    static_argnames=("nz", "ngrdcol"),
)
def fill_holes_wp2_from_horz_tke(
    nz: int,
    ngrdcol: int,
    threshold: float,
    lower_hf_level: int,
    upper_hf_level: int,
    wp2,
    up2,
    vp2,
):
    """Fill wp2 holes by borrowing TKE from up2 and vp2.

    This clips wp2 values below threshold as much as possible while conserving
    turbulent kinetic energy up2+vp2+wp2 at each height level. It does not
    guarantee that wp2 will exceed threshold everywhere; blunt clipping is
    needed for that.
    """
    del ngrdcol
    k_idx = jnp.arange(nz)[None, :]
    in_range = (k_idx >= lower_hf_level) & (k_idx <= upper_hf_level)

    missing_wp2 = threshold - wp2
    up2_avail = jnp.maximum(up2 - threshold, zero)
    vp2_avail = jnp.maximum(vp2 - threshold, zero)
    up2_vp2_avail = up2_avail + vp2_avail
    do_fill = in_range & (wp2 < threshold) & ((up2 > threshold) | (vp2 > threshold))

    # Not enough TKE available to fill the hole.
    case_not_enough = do_fill & (missing_wp2 >= up2_vp2_avail)
    wp2_not_enough = wp2 + up2_vp2_avail
    up2_not_enough = jnp.minimum(up2, threshold)
    vp2_not_enough = jnp.minimum(vp2, threshold)

    # Enough TKE is available to fill the hole.
    case_enough = do_fill & (missing_wp2 < up2_vp2_avail)
    no_up2_avail = jnp.abs(up2_avail) < _F64_EPS * 1000.0
    no_vp2_avail = jnp.abs(vp2_avail) < _F64_EPS * 1000.0
    ratio = jnp.where(up2_vp2_avail > zero, missing_wp2 / up2_vp2_avail, zero)

    up2_enough = jnp.where(
        no_up2_avail,
        up2,
        jnp.where(
            no_vp2_avail,
            up2 - missing_wp2,
            threshold + up2_avail * (one - ratio),
        ),
    )
    vp2_enough = jnp.where(
        no_up2_avail,
        vp2 - missing_wp2,
        jnp.where(
            no_vp2_avail,
            vp2,
            threshold + vp2_avail * (one - ratio),
        ),
    )

    wp2_out = jnp.where(case_not_enough, wp2_not_enough, jnp.where(case_enough, threshold, wp2))
    up2_out = jnp.where(case_not_enough, up2_not_enough, jnp.where(case_enough, up2_enough, up2))
    vp2_out = jnp.where(case_not_enough, vp2_not_enough, jnp.where(case_enough, vp2_enough, vp2))
    return wp2_out, up2_out, vp2_out


__all__ = [
    "fill_holes_global",
    "fill_holes_sliding_window",
    "fill_holes_vertical",
    "fill_holes_wp2_from_horz_tke",
]
