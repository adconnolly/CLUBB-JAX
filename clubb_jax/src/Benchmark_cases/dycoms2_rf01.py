"""JAX port of dycoms2_rf01.F90 — DYCOMS-II RF01 surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/dycoms2_rf01.F90: `dycoms2_rf01_sfclyr` — the GCSS DYCOMS-II RF01
surface layer (fixed ustar=0.25; sfctype=0 prescribed sensible/latent heat, or sfctype=1 fixed-SST bulk
aerodynamic flux). prescribe_forcings.py imports it, mirroring the Fortran case dispatch's `use dycoms2_rf01`.

Tracer-transparent (numpy-in → numpy-out for the prescribed-heat path; the fixed-SST bulk path uses
sat_mixrat_liq) → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _iset
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.Benchmark_cases.sfc_flux import (
    compute_wpthlp_sfc, compute_wprtp_sfc,
    convert_sens_ht_to_km_s, convert_latent_ht_to_m_s)


def dycoms2_rf01_tndcy(state: dict) -> None:
    """DYCOMS-II RF01 large-scale tendencies (dycoms2_rf01.F90:dycoms2_rf01_tndcy): zero thlm/rtm forcing;
    the subsidence wm is set at init and left unchanged (the Fortran tndcy does not touch wm).

    Tracer-transparent (_iset): under a jax.grad trace thlm_forcing/rtm_forcing carry the Morrison
    *_mc tendency (a jnp tracer) from the prior step, so a raw ``[:] =`` in-place set fails; _iset
    reassigns immutably under trace and mutates in place (bit-identical) on the concrete path."""
    state['thlm_forcing'] = _iset(state['thlm_forcing'], np.s_[:], 0.0)
    state['rtm_forcing'] = _iset(state['rtm_forcing'], np.s_[:], 0.0)


def dycoms2_rf01_sfclyr(state: dict, time_current: float, ngrdcol: int,
                            rho_bot, ubar=None, thlm_bot=None, rtm_bot=None, exner_bot=None) -> tuple:
    """DYCOMS-II RF01 surface fluxes (dycoms2_rf01.F90:dycoms2_rf01_sfclyr). ustar=0.25 (GCSS spec).
    sfctype=0: prescribed sens_ht/latent_ht → /(rho·Cp), /(rho·Lv). sfctype=1 (fixed SST): bulk flux
    with Cd=0.0011, T_sfc from the sfc file: wpthlp_sfc=-Cd·ubar·(thlm-T_sfc/exner),
    wprtp_sfc=-Cd·ubar·(rtm-rsat). Reads sens_ht/latent_ht/t_sfc from dycoms2_rf01[_fixed_sst]_sfc.in."""
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))

    def _interp(key, default):
        arr = sfc.get(key)
        if arr is not None and len(arr) > 0:
            return float(np.interp(time_current, times, arr))
        return default

    T_sfc_val = _interp('t_sfc', 292.0)
    ustar = np.full(ngrdcol, 0.25)
    T_sfc = np.full(ngrdcol, T_sfc_val)

    if int(state.get('sfctype', 0)) == 1:                # fixed-SST bulk flux (dycoms2_rf01_fixed_sst)
        Cd = 0.0011
        sat_formula = state['flags'].saturation_formula
        p_sfc = np.asarray(state['p_sfc'], dtype=np.float64)
        rsat = np.array([float(sat_mixrat_liq(float(p_sfc[i]), T_sfc_val, sat_formula))
                         for i in range(ngrdcol)])
        wpthlp_sfc = compute_wpthlp_sfc(Cd, ubar, thlm_bot, T_sfc, exner_bot)
        wprtp_sfc  = compute_wprtp_sfc(Cd, ubar, rtm_bot, rsat)
    else:                                                # sfctype=0: prescribed sens/latent heat
        sens_ht   = _interp('sens_ht',   16.0)   # W/m^2 — DYCOMS RF01 spec
        latent_ht = _interp('latent_ht', 93.0)
        wpthlp_sfc = convert_sens_ht_to_km_s(sens_ht, rho_bot)
        wprtp_sfc  = convert_latent_ht_to_m_s(latent_ht, rho_bot)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc
