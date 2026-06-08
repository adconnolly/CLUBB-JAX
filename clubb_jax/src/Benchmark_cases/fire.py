"""JAX port of fire.F90 — FIRE case surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/fire.F90: `fire_sfclyr` — the bulk-aerodynamic surface layer
(sfctype=1) with a time-interpolated SST and the saturation mixing ratio at the surface. prescribe_forcings.py
imports it, mirroring the Fortran case dispatch's `use fire`.

Tracer-transparent (the saturation mixing ratio uses sat_mixrat_liq) → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc


def fire_sfclyr(state: dict, time_current: float, ngrdcol: int,
                    ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """FIRE surface fluxes — bulk formula, sfctype=1 (fire.F90:fire_sfclyr).

    Cz=0.0013, ustar=0.3, T_sfc from fire_sfc.in; rsat = sat_mixrat_liq(p_sfc, T_sfc).
    wpthlp_sfc = -Cz*ubar*(thlm - T_sfc/exner);  wprtp_sfc = -Cz*ubar*(rtm - rsat).
    """
    sfc = (state.get('_forcings_data') or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))
    t_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = (float(np.interp(time_current, times, t_sfc_arr))
                 if t_sfc_arr is not None else 288.0)
    Cz = 0.0013
    ustar = np.full(ngrdcol, 0.3)
    T_sfc = np.full(ngrdcol, T_sfc_val)
    sat_formula = state['flags'].saturation_formula
    p_sfc = state['p_sfc']
    rsat = np.array([float(sat_mixrat_liq(float(p_sfc[i]), T_sfc_val, sat_formula))
                     for i in range(ngrdcol)])
    wpthlp_sfc = compute_wpthlp_sfc(Cz, ubar, thlm_bot, T_sfc, exner_bot)
    wprtp_sfc  = compute_wprtp_sfc(Cz, ubar, rtm_bot, rsat)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc
