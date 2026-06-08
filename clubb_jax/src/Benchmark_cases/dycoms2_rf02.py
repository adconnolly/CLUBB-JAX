"""JAX port of dycoms2_rf02.F90 — DYCOMS-II RF02 surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/dycoms2_rf02.F90: `dycoms2_rf02_sfclyr` — prescribed time-interpolated
sensible/latent heat fluxes converted to kinematic form with a CONSTANT surface density, fixed ustar.
prescribe_forcings.py imports it, mirroring the Fortran case dispatch's `use dycoms2_rf02` (shared by the
rf02_nd / rf02_so / rf02_do / rf02_ds variants).

Pure-numpy → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np


def dycoms2_rf02_tndcy(state: dict) -> None:
    """DYCOMS-II RF02 large-scale tendencies (dycoms2_rf02.F90:dycoms2_rf02_tndcy): zero thlm/rtm forcing and
    zero the subsidence wm ONLY at the top level (the rest of wm is set at init and left unchanged, as in RF01)."""
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0
    state['wm_zt'][:, -1] = 0.0
    state['wm_zm'][:, -1] = 0.0


def dycoms2_rf02_sfclyr(state: dict, time_current: float, ngrdcol: int) -> tuple:
    """DYCOMS-II RF02 surface fluxes (dycoms2_rf02.F90:dycoms2_rf02_sfclyr): time-interpolated
    sens_ht/latent_ht converted with rho_sfc=1.21 (CONSTANT): wpthlp_sfc=sens_ht/(1.21·Cp),
    wprtp_sfc=latent_ht/(1.21·Lv); ustar=0.25. (The sfc file has sens_ht/latent_ht, not wpthlp_sfc.)"""
    from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv
    rho_sfc = 1.21
    sfc = (state['_forcings_data'] or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))
    sh = sfc.get('sens_ht'); lh = sfc.get('latent_ht')
    sens_ht = float(np.interp(time_current, times, sh)) if sh is not None else 0.0
    latent_ht = float(np.interp(time_current, times, lh)) if lh is not None else 0.0
    wpthlp_sfc = np.full(ngrdcol, sens_ht / (rho_sfc * Cp))
    wprtp_sfc  = np.full(ngrdcol, latent_ht / (rho_sfc * Lv))
    ustar      = np.full(ngrdcol, 0.25)
    return wpthlp_sfc, wprtp_sfc, ustar
