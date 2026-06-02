"""JAX implementation of fill_holes.F90:fill_holes_vertical_api.

ARM uses fill_holes_type=2 (sliding_window with global fallback).
num_hf_draw_points=2 (from constants_clubb.F90).
"""

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import eps

_NUM_HF_DRAW = 2   # num_hf_draw_points=2 — window half-width for sliding fill


def _fill_holes_global_jax(field, rho_dz, threshold, lower_k, upper_k):
    """fill_holes_global: mass-conserving global fill over [lower_k, upper_k].

    Args:
        field:    (ngrdcol, nz) — field to fill (modified in-place-style)
        rho_dz:   (ngrdcol, nz) — rho_ds * dz (precomputed)
        threshold: scalar
        lower_k, upper_k: Python int — 0-based inclusive index range

    Returns:
        field: (ngrdcol, nz) with holes filled
    """
    nz = field.shape[1]
    k_idx = jnp.arange(nz)[None, :]   # (1, nz)
    mask = (k_idx >= lower_k) & (k_idx <= upper_k)

    rho_dz_m = jnp.where(mask, rho_dz, 0.0)
    denom = jnp.sum(rho_dz_m, axis=1, keepdims=True)          # (ngrdcol, 1)
    field_avg = jnp.sum(rho_dz_m * field, axis=1, keepdims=True) / denom  # (ngrdcol, 1)

    # Clip field
    field_clipped = jnp.where(
        field_avg >= threshold,
        jnp.maximum(threshold, field),
        jnp.minimum(threshold, field),
    )
    field_clipped_avg = jnp.sum(rho_dz_m * field_clipped, axis=1, keepdims=True) / denom

    # Mass-conservation coefficient
    safe = (jnp.abs(field_clipped_avg - threshold)
            > jnp.abs(field_clipped_avg + threshold) * eps / 2.0)
    mass_frac = jnp.where(
        safe,
        (field_avg - threshold) / (field_clipped_avg - threshold),
        1.0,
    )
    field_new = jnp.where(mask,
                          threshold + mass_frac * (field_clipped - threshold),
                          field)

    # Only apply if any hole exists in this column
    any_hole = jnp.any(jnp.where(mask, field < threshold, False),
                       axis=1, keepdims=True)  # (ngrdcol, 1)
    return jnp.where(any_hole, field_new, field)


def _fill_holes_sliding_window_jax(field, rho_dz, threshold, lower_k, upper_k,
                                   num_draw=_NUM_HF_DRAW):
    """fill_holes_sliding_window + global fallback (fill_holes_type=2).

    Args:
        field:    (ngrdcol, nz)
        rho_dz:   (ngrdcol, nz) — rho_ds * dz
        threshold: scalar
        lower_k, upper_k: Python int — 0-based inclusive bounds
        num_draw: half-window size (default 2)

    Returns:
        field: (ngrdcol, nz) with holes filled
    """
    # Fast path: no holes — skip all work
    # (Fortran checks this first; in JAX we always run the loop but it's a noop)
    wlen = 2 * num_draw + 1  # static window length = 5

    def body(k, field_carry):
        """Process one level k of the sliding window."""
        # Dynamic slice window of width wlen centered at k
        start = k - num_draw   # dynamic integer
        field_win = jax.lax.dynamic_slice(
            field_carry, (0, start), (field_carry.shape[0], wlen))   # (ngrdcol, wlen)
        rho_dz_win = jax.lax.dynamic_slice(
            rho_dz, (0, start), (rho_dz.shape[0], wlen))             # (ngrdcol, wlen)

        denom = jnp.sum(rho_dz_win, axis=1, keepdims=True)           # (ngrdcol, 1)
        field_avg = jnp.sum(rho_dz_win * field_win, axis=1, keepdims=True) / denom

        # Only modify if any value in window is below threshold
        any_hole = jnp.any(field_win < threshold, axis=1, keepdims=True)  # (ngrdcol,1)

        # Clip
        field_clipped = jnp.where(
            field_avg >= threshold,
            jnp.maximum(threshold, field_win),
            jnp.minimum(threshold, field_win),
        )
        field_clipped_avg = jnp.sum(rho_dz_win * field_clipped, axis=1, keepdims=True) / denom

        # Mass-conservation coefficient (avoid divide-by-zero)
        safe = (jnp.abs(field_clipped_avg - threshold)
                > jnp.abs(field_clipped_avg + threshold) * eps / 2.0)
        mass_frac = jnp.where(
            safe,
            (field_avg - threshold) / (field_clipped_avg - threshold),
            1.0,
        )
        field_win_new = threshold + mass_frac * (field_clipped - threshold)
        # Only write back if any hole existed in window
        field_win_out = jnp.where(any_hole, field_win_new, field_win)

        field_out = jax.lax.dynamic_update_slice(field_carry, field_win_out, (0, start))
        return field_out

    # Serial loop: k from lower_k+num_draw to upper_k-num_draw (inclusive)
    start_k = lower_k + num_draw
    end_k = upper_k - num_draw + 1   # exclusive for fori_loop

    field_sw = jax.lax.fori_loop(start_k, end_k, body, field)

    # Global fallback if holes remain
    field_out = jax.lax.cond(
        jnp.any(field_sw < threshold),
        lambda f: _fill_holes_global_jax(f, rho_dz, threshold, lower_k, upper_k),
        lambda f: f,
        field_sw,
    )
    return field_out


def fill_holes_vertical_jax(field, rho_ds, dz, threshold, lower_k, upper_k,
                             fill_holes_type, grid_dir_indx=1):
    """fill_holes_vertical_api in JAX.

    Args:
        field:           (ngrdcol, nz) — field to fill (NOT mutated)
        rho_ds:          (ngrdcol, nz)
        dz:              (ngrdcol, nz)
        threshold:       scalar (e.g. w_tol_sqd)
        lower_k, upper_k: Python int — 0-based inclusive index range
        fill_holes_type: Python int — 1=global, 2=sliding_window+global_fallback
        grid_dir_indx:   Python int — +1 (standard ascending grid)

    Returns:
        field: (ngrdcol, nz) with holes filled (copy, not in-place)
    """
    rho_dz = rho_ds * dz   # precomputed for both fill methods

    # Skip entirely if no values below threshold (fast path, no JIT-traced branch)
    # In JIT, we always trace both paths but only apply the relevant one.
    if fill_holes_type == 1:
        return _fill_holes_global_jax(field, rho_dz, threshold, lower_k, upper_k)
    elif fill_holes_type == 2:
        return _fill_holes_sliding_window_jax(field, rho_dz, threshold, lower_k, upper_k)
    else:
        raise NotImplementedError(f"fill_holes_type={fill_holes_type} not implemented")


# Called eagerly ~7-9× per timestep (rtm, thlm, rtp2, thlp2, up2, vp2, wp2), the inner sliding-window
# `fori_loop` / global-fill bodies CLOSE OVER `rho_dz`/`threshold`, so eager use bakes those values into
# the loop jaxpr → XLA recompiles every step → unbounded compile-cache growth (the Iter290 residual ~9
# scan-recompiles/step; Iter291). Jitting the entry makes the arrays tracers (hoisted to operands), so
# each (grid-size, fill_type) variant compiles ONCE and cache-hits. The int control args
# (lower_k/upper_k/fill_holes_type/grid_dir_indx) drive Python branching/shaping → static; `threshold`
# is only used arithmetically → traced. Value-preserving + differentiable; all callers import this name.
fill_holes_vertical_jax = jax.jit(
    fill_holes_vertical_jax,
    static_argnames=("lower_k", "upper_k", "fill_holes_type", "grid_dir_indx"),
)
