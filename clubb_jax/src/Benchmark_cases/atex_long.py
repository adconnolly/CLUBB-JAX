"""JAX port of atex_long.F90 — Long-ATEX case large-scale tendencies + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/atex_long.F90: `atex_long_tndcy` (a FIXED 3-piece subsidence profile +
nonzero thlm/rtm large-scale forcings, all ramped linearly during a 12 h spin-up) and `atex_long_sfclyr`
(bulk-aerodynamic surface fluxes with a time-interpolated SST, fixed ustar). prescribe_forcings.py imports these,
mirroring the Fortran case dispatch's `use atex_long`.

Pure-numpy / tracer-transparent → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zt2zm
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc


def atex_long_tndcy(state: dict, time_current: float) -> None:
    """Long-ATEX analytic large-scale tendencies (atex_long.F90:atex_long_tndcy). Unlike `atex`, the
    subsidence is a FIXED 3-piece profile (no rtm-based inversion, no 90-min gate) and there are nonzero
    thlm/rtm forcings (atex_long.F90:calc_forcings). During the first `spinup = 43200 s` all three
    (thlm_forcing, rtm_forcing, wm_zt) ramp linearly as `· time / spinup`. `wm_zm = zt2zm(wm_zt)`, BCs zeroed.

    wm_zt(z):   -0.00636·z/1050                       0≤z<1050
                -0.00636 - 0.00079·(z-1050)/600       1050≤z<1650
                -0.00715                               z≥1650
    thlm_forcing(z): -3.5805e-5                        0≤z<1400
                     -3.5805e-5 + 1.1935e-5·(z-1400)·0.004      1400≤z<1650
                     -2.3870e-5 - 0.1155e-5·(z-1650)/1350       1650≤z<2990
                     0                                          z≥2990
    rtm_forcing(z):  -1.58e-8·(1 - z/1050)             0≤z<1050 ; else 0
    """
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    # subsidence (3-piece)
    wm_zt = np.where(zt < 1050.0, -0.00636 * zt / 1050.0,
                     np.where(zt < 1650.0, -0.00636 - 0.00079 * (zt - 1050.0) / 600.0,
                              -0.00715))
    thlm_forcing, rtm_forcing = calc_forcings(gr, zt)
    # spin-up ramp (atex_long.F90:155-160)
    spinup = 43200.0
    if time_current < spinup:
        ramp = time_current / spinup
        thlm_forcing = thlm_forcing * ramp
        rtm_forcing = rtm_forcing * ramp
        wm_zt = wm_zt * ramp
    state['thlm_forcing'][:] = thlm_forcing
    state['rtm_forcing'][:] = rtm_forcing
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(wm_zt)), dtype=np.float64)
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def calc_forcings(gr, zt: np.ndarray):
    """atex_long.F90:calc_forcings — the (unramped) ATEX-long thlm/rtm large-scale forcing profiles
    (the subsidence wm + spin-up ramp are applied by the caller atex_long_tndcy). Returns
    (thlm_forcing, rtm_forcing), each (ngrdcol, nzt). (Extracted from the inline block in
    atex_long_tndcy to mirror the Fortran atex_long_tndcy→calc_forcings split, mirror-refactor iter 239.)
    """
    # theta-l tendency (4-piece)
    thlm_forcing = np.where(zt < 1400.0, -3.5805e-5,
                            np.where(zt < 1650.0, -3.5805e-5 + 1.1935e-5 * (zt - 1400.0) * 0.004,
                                     np.where(zt < 2990.0, -2.3870e-5 - 0.1155e-5 * (zt - 1650.0) / 1350.0,
                                              0.0)))
    # moisture tendency (2-piece)
    rtm_forcing = np.where(zt < 1050.0, -1.58e-8 * (1.0 - zt / 1050.0), 0.0)
    return thlm_forcing, rtm_forcing


def atex_long_sfclyr(state: dict, time_current: float, ngrdcol: int,
                         ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """ATEX-Long surface fluxes (atex_long.F90:atex_long_sfclyr, l_compute_flux=True).

    C_10=0.0013, adjustment=0.0194664, ustar=0.3. T_sfc from sfc file.
    """
    C_10       = 0.0013
    adjustment = 0.0194664

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
