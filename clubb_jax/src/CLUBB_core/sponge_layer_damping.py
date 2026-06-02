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
    xm/xm_ref/z are (nz,). Returns the damped (nz,) array.
    """
    xm_p = np.array(xm, dtype=np.float64)
    nz = len(z)
    for k in range(nz - 1, -1, -1):          # top down
        if zm_top - z[k] < sponge_layer_depth:
            dt_on_tau = dt / tau[k]
            xm_p[k] = (xm[k] + dt_on_tau * xm_ref[k]) / (1.0 + dt_on_tau)
        else:
            break
    return xm_p
