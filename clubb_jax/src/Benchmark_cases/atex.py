"""JAX port of atex.F90 — ATEX case large-scale tendencies + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/atex.F90: `atex_tndcy` (rtm-inversion-based subsidence + thlm/rtm
large-scale forcing, both gated off for the first 90 min) and `atex_sfclyr` (bulk-aerodynamic surface fluxes
with a time-interpolated SST, fixed ustar). prescribe_forcings.py imports these, mirroring the Fortran case
dispatch's `use atex`.

Pure-numpy / tracer-transparent → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zt2zm
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc


def atex_tndcy(state: dict, time_current: float) -> None:
    """ATEX analytic large-scale tendencies (atex.F90:atex_tndcy). The WHOLE package — subsidence AND
    the thlm/rtm large-scale forcing — is OFF for the first 90 min (`time >= time_initial + 5400`,
    atex.F90:215); then `z_inversion` = the zt level just below the first level where rtm ≤ 6.5e-3, and:
      wm_zt    = -0.0065·z/z_inv (0<z≤z_inv), a linear ramp to 0 over the next 300 m, else 0;
      thlm_forcing = -1.1575e-5·(3 - z/z_inv) for 0<z<z_inv, -2.315e-5·(1-(z-z_inv)/300) in the next
                     300 m, else 0;  rtm_forcing = -1.58e-8·(1 - z/z_inv) for 0<z<z_inv, else 0
    (atex.F90:calc_forcings). NOTE the forcing uses a STRICT z < z_inv where the subsidence uses z ≤ z_inv.
    `wm_zm = zt2zm(wm_zt)` with zeroed surface/top BCs."""
    gr = state['gr']; ngrdcol = state['ngrdcol']
    if time_current < state['time_initial'] + 5400.0:
        state['thlm_forcing'][:] = 0.0
        state['rtm_forcing'][:] = 0.0
        state['wm_zt'][:] = 0.0
        state['wm_zm'][:] = 0.0
        return
    zt = np.asarray(gr.zt, dtype=np.float64)
    rtm = np.asarray(state['rtm'], dtype=np.float64)
    wm_zt = np.zeros_like(zt)
    z_inversion = np.zeros(ngrdcol, dtype=np.float64)
    for i in range(ngrdcol):
        j = int(np.argmax(rtm[i] <= 6.5e-3))          # first 0-based level with rtm ≤ 6.5e-3
        z_inv = zt[i, j - 1]                           # the level just below (Fortran zt(z_lev-1))
        z_inversion[i] = z_inv
        z = zt[i]
        wm_zt[i] = np.where((z > 0.0) & (z <= z_inv), -0.0065 * z / z_inv,
                            np.where((z > z_inv) & (z <= z_inv + 300.0),
                                     -0.0065 * (1.0 - (z - z_inv) / 300.0), 0.0))
    thlm_f, rtm_f = calc_forcings(ngrdcol, gr, z_inversion, zt)
    state['thlm_forcing'][:] = thlm_f
    state['rtm_forcing'][:] = rtm_f
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(wm_zt)), dtype=np.float64)
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def calc_forcings(ngrdcol: int, gr, z_inversion: np.ndarray, zt: np.ndarray):
    """atex.F90:calc_forcings — the ATEX thlm/rtm large-scale forcing profiles given `z_inversion`
    (the subsidence wm is computed by the caller atex_tndcy). Returns (thlm_forcing, rtm_forcing),
    each (ngrdcol, nzt). (Extracted from the inline block in atex_tndcy to mirror the Fortran
    atex_tndcy→calc_forcings split, mirror-refactor iter 238.)
    """
    thlm_f = np.zeros_like(zt)
    rtm_f = np.zeros_like(zt)
    for i in range(ngrdcol):
        z_inv = z_inversion[i]
        z = zt[i]
        thlm_f[i] = np.where((z > 0.0) & (z < z_inv),
                             -1.1575e-5 * (3.0 - z / z_inv),
                             np.where((z > z_inv) & (z <= z_inv + 300.0),
                                      -2.315e-5 * (1.0 - (z - z_inv) / 300.0), 0.0))
        rtm_f[i] = np.where((z > 0.0) & (z < z_inv), -1.58e-8 * (1.0 - z / z_inv), 0.0)
    return thlm_f, rtm_f


def atex_sfclyr(state: dict, time_current: float, ngrdcol: int,
                    ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """ATEX surface fluxes (atex.F90:atex_sfclyr, sfctype=1).

    C_10=0.0013, adjustment=0.0198293 (fixed), ustar=0.3.
    T_sfc time-interpolated from atex_sfc.in.
    """
    C_10       = 0.0013
    adjustment = 0.0198293

    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))
    T_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = float(np.interp(time_current, times, T_sfc_arr)) if T_sfc_arr is not None else 298.0

    T_sfc   = np.full(ngrdcol, T_sfc_val)
    ustar   = np.full(ngrdcol, 0.3)
    Cd      = np.full(ngrdcol, C_10)
    adj     = np.full(ngrdcol, adjustment)

    wpthlp_sfc = compute_wpthlp_sfc(Cd, ubar, thlm_bot, T_sfc, exner_bot)
    wprtp_sfc  = compute_wprtp_sfc(Cd, ubar, rtm_bot, adj)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc
