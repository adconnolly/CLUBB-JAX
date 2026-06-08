"""JAX implementation of fill_holes.F90:fill_holes_vertical_api.

ARM uses fill_holes_type=2 (sliding_window with global fallback).
num_hf_draw_points=2 (from constants_clubb.F90).
"""

import jax
import jax.numpy as jnp
import numpy as np

from clubb_jax.src.CLUBB_core.constants_clubb import eps, Lv, Cp

_NUM_HF_DRAW = 2   # num_hf_draw_points=2 — window half-width for sliding fill


def fill_holes_global(field, rho_dz, threshold, lower_k, upper_k):
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


def fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k,
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
        lambda f: fill_holes_global(f, rho_dz, threshold, lower_k, upper_k),
        lambda f: f,
        field_sw,
    )
    return field_out


def fill_holes_vertical(field, rho_ds, dz, threshold, lower_k, upper_k,
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
        return fill_holes_global(field, rho_dz, threshold, lower_k, upper_k)
    elif fill_holes_type == 2:
        return fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k)
    else:
        raise NotImplementedError(f"fill_holes_type={fill_holes_type} not implemented")


# Called eagerly ~7-9× per timestep (rtm, thlm, rtp2, thlp2, up2, vp2, wp2), the inner sliding-window
# `fori_loop` / global-fill bodies CLOSE OVER `rho_dz`/`threshold`, so eager use bakes those values into
# the loop jaxpr → XLA recompiles every step → unbounded compile-cache growth (the Iter290 residual ~9
# scan-recompiles/step; Iter291). Jitting the entry makes the arrays tracers (hoisted to operands), so
# each (grid-size, fill_type) variant compiles ONCE and cache-hits. The int control args
# (lower_k/upper_k/fill_holes_type/grid_dir_indx) drive Python branching/shaping → static; `threshold`
# is only used arithmetically → traced. Value-preserving + differentiable; all callers import this name.
fill_holes_vertical = jax.jit(
    fill_holes_vertical,
    static_argnames=("lower_k", "upper_k", "fill_holes_type", "grid_dir_indx"),
)


_F64_EPS = jnp.finfo(jnp.float64).eps


def fill_holes_wp2_from_horz_tke(wp2, up2, vp2, threshold, lower_k, upper_k):
    """fill_holes.F90:fill_holes_wp2_from_horz_tke — TKE-conserving wp2 hole-fill.

    Where wp2 < threshold AND (up2 > threshold OR vp2 > threshold), borrows TKE
    from up2 and vp2 proportionally to fill holes in wp2.

    ARM call: lower_hf_level=1 (Fortran 1-based), upper_hf_level=nzm-2 (Fortran 1-based)
              → Python lower_k=0..1? Let's accept 0-based Python range.

    Args:
        wp2, up2, vp2: (ngrdcol, nzm)
        threshold:     scalar
        lower_k, upper_k: Python 0-based inclusive index range

    Returns:
        wp2, up2, vp2: (ngrdcol, nzm) modified within [lower_k, upper_k]
    """
    nzm = wp2.shape[1]
    k_idx = jnp.arange(nzm)[None, :]   # (1, nzm)
    in_range = (k_idx >= lower_k) & (k_idx <= upper_k)

    # Operate only where a hole exists AND there is available TKE
    has_hole    = wp2 < threshold
    has_tke_up  = up2 > threshold
    has_tke_vp  = vp2 > threshold
    do_fill     = in_range & has_hole & (has_tke_up | has_tke_vp)

    missing     = threshold - wp2            # > 0 where has_hole
    up2_avail   = jnp.maximum(up2 - threshold, 0.0)
    vp2_avail   = jnp.maximum(vp2 - threshold, 0.0)
    total_avail = up2_avail + vp2_avail      # > 0 where (has_tke_up | has_tke_vp)

    # --- Case 1: not enough TKE ---
    case1 = do_fill & (missing >= total_avail)
    wp2_c1  = wp2 + total_avail
    up2_c1  = jnp.minimum(up2, threshold)
    vp2_c1  = jnp.minimum(vp2, threshold)

    # --- Case 2: enough TKE (missing < total_avail) ---
    case2 = do_fill & (missing < total_avail)

    # Fortran epsilon threshold for deciding if a component is "zero"
    eps_thr = _F64_EPS * 1000.0

    case2a = case2 & (jnp.abs(up2_avail) < eps_thr)   # no up2 avail → take all from vp2
    case2b = case2 & (~case2a) & (jnp.abs(vp2_avail) < eps_thr)  # no vp2 avail → take all from up2
    case2c = case2 & (~case2a) & (~case2b)              # both available → proportional

    ratio  = jnp.where(total_avail > 0.0, missing / total_avail, 0.0)

    # case2a: up2 unchanged, vp2 -= missing
    up2_2a = up2
    vp2_2a = vp2 - missing

    # case2b: up2 -= missing, vp2 unchanged
    up2_2b = up2 - missing
    vp2_2b = vp2

    # case2c: proportional
    up2_2c = threshold + up2_avail * (1.0 - ratio)
    vp2_2c = threshold + vp2_avail * (1.0 - ratio)

    up2_c2 = jnp.where(case2a, up2_2a, jnp.where(case2b, up2_2b, up2_2c))
    vp2_c2 = jnp.where(case2a, vp2_2a, jnp.where(case2b, vp2_2b, vp2_2c))

    # --- Assemble ---
    wp2_new = jnp.where(case1, wp2_c1, jnp.where(case2, threshold, wp2))
    up2_new = jnp.where(case1, up2_c1, jnp.where(case2, up2_c2, up2))
    vp2_new = jnp.where(case1, vp2_c1, jnp.where(case2, vp2_c2, vp2))

    return wp2_new, up2_new, vp2_new


def fill_holes_hydromet_clip_jax(hm, num, hm_tol, rvm_mc, thlm_mc, exner, dt):
    """Clip a non-frozen (rain) precipitating hydrometeor mass `hm` <= `hm_tol` to 0, returning the removed mass
    to vapor (rvm_mc) with a latent cooling on thlm (mirrors fill_holes.F90:fill_holes_driver_api, lines
    2444-2476); the partner number `num` is zeroed (clip_hydromet_conc_mvr, <rx>=0). Concrete-numpy path
    (the KK step runs concrete). Returns (hm, num, rvm_mc, thlm_mc)."""
    below = hm <= hm_tol
    removed = np.where(below, hm, 0.0)                      # removed rr mass [kg/kg]
    hm = np.where(below, 0.0, hm)
    num = np.where(below, 0.0, num)
    exner = np.asarray(exner, np.float64)
    rvm_mc = np.asarray(rvm_mc, np.float64) + removed / dt
    thlm_mc = np.asarray(thlm_mc, np.float64) - (Lv / Cp) * removed / (exner * dt)
    return hm, num, rvm_mc, thlm_mc
