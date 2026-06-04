"""Sponge-layer damping — port of sponge_layer_damping.F90.

Relaxes a mean field toward a reference profile in the top region of the domain
(absorbs gravity waves / reduces reflection off the model top). Active per field
when the corresponding `*_sponge_damp_settings%l_sponge_damping` flag is set
(e.g. ekman, coriolis_test, cloud_feedback, cgils). It is a no-op for cases with
all sponge flags off (which is all 9 current bit-faithful cases).

Ascending-grid standalone path (grid_dir_indx = +1). Per-column (ngrdcol=1) arrays.

Reference: clubb_release/src/CLUBB_core/sponge_layer_damping.F90
  initialize_tau_sponge_damp_api / sponge_damp_xm
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def initialize_tau_sponge_damp(z, dt, zm_top, tau_min, tau_max, sponge_damp_depth):
    """Build the damping-timescale profile (initialize_tau_sponge_damp_api).

    Args:
        z:          (nz,) heights of the levels being damped [m] (zt for xm, zm for xp2/xp3).
        dt:         timestep [s].
        zm_top:     height of the model top, gr.zm[k_ub_zm] [m].
        tau_min:    damping timescale at the model top [s].
        tau_max:    damping timescale at the base of the damping layer [s].
        sponge_damp_depth: damping depth as a fraction of the domain height [-].

    Returns:
        (tau, sponge_layer_depth): tau is (nz,) with np.inf below the sponge layer
        (no damping); sponge_layer_depth [m].
    """
    nz = len(z)
    sponge_layer_depth = sponge_damp_depth * zm_top
    if tau_min < 2.0 * dt:
        raise ValueError("tau_sponge_damp_min is too small (< 2*dt)")
    tau = np.full(nz, np.inf, dtype=np.float64)
    ratio = tau_max / tau_min
    for k in range(nz - 1, -1, -1):          # top down (ascending grid)
        d = zm_top - z[k]
        if d < sponge_layer_depth:
            exponent = d / sponge_layer_depth
            tau[k] = tau_min * ratio ** exponent
        else:
            break                            # below the sponge layer
    return tau, sponge_layer_depth


def sponge_damp_xm(xm, xm_ref, z, zm_top, tau, sponge_layer_depth, dt):
    """Damp xm toward xm_ref in the top sponge layer (sponge_damp_xm).

    Implicit discretization: xm_p[k] = (xm[k] + (dt/tau[k])*xm_ref[k]) / (1 + dt/tau[k]).

    Vectorized + tracer-transparent (REFACTOR B5): ``tau`` is ``inf`` outside the sponge (initialize_tau_
    sponge_damp), so ``dt/tau == 0`` there and the formula collapses to ``xm`` (a no-op) — making this
    pure broadcast arithmetic, **bit-identical** to the original top-down loop (which only touches the top
    contiguous sponge block) yet differentiable. Works for a 1-D column or a batched (ngrdcol, nz) array.
    (z/zm_top/sponge_layer_depth are now implicit in ``tau`` and kept only for signature compatibility.)"""
    dt_on_tau = dt / tau                       # 0.0 where tau=inf (outside the sponge), finite inside
    return (xm + dt_on_tau * xm_ref) / (1.0 + dt_on_tau)


def sponge_damp_xp2(dt, zm, xp2, x_tol_sqd, tau, sponge_layer_depth):
    """Damp a variance <x'^2> in the top sponge layer (sponge_layer_damping.F90:sponge_damp_xp2):
    xp2_damped = max((1 − dt/tau)^2 · xp2, x_tol_sqd) where (zm_top − zm) < sponge_layer_depth, else xp2.
    zm and xp2 are (ngrdcol, nzm); tau is (ngrdcol, nzm). Pure-jnp → differentiable."""
    zm = jnp.asarray(zm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    zm_top = zm[:, -1:]
    in_sponge = (zm_top - zm) < sponge_layer_depth
    damped = jnp.maximum((1.0 - dt / tau) ** 2 * xp2, x_tol_sqd)
    return jnp.where(in_sponge, damped, xp2)


def sponge_damp_xp3(dt, z, zm, xp3, tau, sponge_layer_depth):
    """Damp a third moment <x'^3> in the top sponge layer (sponge_layer_damping.F90:sponge_damp_xp3):
    xp3_damped = (1 − dt/tau)^3 · xp3 where (zm_top − z) < sponge_layer_depth, else xp3.
    z and xp3 are (ngrdcol, nzt) thermodynamic-level fields; zm is (ngrdcol, nzm); tau is (ngrdcol, nzt).
    Pure-jnp → differentiable."""
    z = jnp.asarray(z, dtype=jnp.float64); zm = jnp.asarray(zm, dtype=jnp.float64)
    xp3 = jnp.asarray(xp3, dtype=jnp.float64); tau = jnp.asarray(tau, dtype=jnp.float64)
    zm_top = zm[:, -1:]
    in_sponge = (zm_top - z) < sponge_layer_depth
    damped = (1.0 - dt / tau) ** 3 * xp3
    return jnp.where(in_sponge, damped, xp3)
