"""JAX port of rico.F90 — RICO case large-scale tendencies + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/rico.F90: `rico_tndcy` (analytic, time-independent thlm/rtm large-scale
forcings; subsidence is init-set and untouched) and `rico_sfclyr` (RICO-3D bulk-aerodynamic surface fluxes with
height-scaled drag coefficients and a time-interpolated SST). prescribe_forcings.py imports these, mirroring the
Fortran case dispatch's `use rico`.

Pure-numpy / tracer-transparent → bit-identical and differentiable. (The rico CASE is KK-microphysics-limited /
Tier-C, not strictly bit-faithful — that is a property of the case, independent of this verbatim relocation.)
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.CLUBB_core.constants_clubb import g_per_kg  # rico.F90:39 `use constants_clubb, only: g_per_kg`
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.Benchmark_cases.spec_hum_to_mixing_ratio import force_spec_hum_to_mixing_ratio
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc


def rico_sfclyr(state: dict, time_current: float, ngrdcol: int,
                    um_bot, vm_bot, thlm_bot, rtm_bot, rho_bot, exner_bot,
                    z_bot, ubar) -> tuple:
    """RICO surface fluxes — RICO 3D spec (rico.F90:rico_sfclyr, l_use_old_atex=False).

    Uses bulk aerodynamic coefficients C_m_20, C_h_20, C_q_20 scaled to model height.
    T_sfc interpolated from rico_sfc.in.
    """
    C_m_20 = 0.001229
    C_h_20 = 0.001094
    C_q_20 = 0.001133
    z0     = 0.00015
    z_ref  = 20.0

    fd = state['_forcings_data']
    sfc = fd.get('sfc')
    if sfc is not None and 't_sfc' in sfc:
        T_sfc_val = float(np.interp(time_current, sfc['time'], sfc['t_sfc']))
    else:
        T_sfc_val = 299.8
    T_sfc = np.full(ngrdcol, T_sfc_val)

    log_ref_z0 = np.log(z_ref / z0)
    log_z_z0   = np.log(np.maximum(z_bot, z0 * 1.001) / z0)
    scale      = (log_ref_z0 / log_z_z0) ** 2

    Cm = C_m_20 * scale
    Ch = C_h_20 * scale
    Cq = C_q_20 * scale

    ustar = np.full(ngrdcol, 0.3)  # Rico spec constant

    # sat_mixrat_liq expects scalar saturation_formula
    sat_formula = state['flags'].saturation_formula
    p_sfc = state['p_sfc']
    rsat = np.array([float(sat_mixrat_liq(
        float(p_sfc[i]), T_sfc_val, sat_formula)) for i in range(ngrdcol)])

    wpthlp_sfc = compute_wpthlp_sfc(Ch, ubar, thlm_bot, T_sfc, exner_bot)
    wprtp_sfc  = compute_wprtp_sfc(Cq, ubar, rtm_bot, rsat)
    # Momentum flux: -Cm * ubar (not using ustar^2/ubar pattern for RICO)
    upwp_sfc = -um_bot * Cm * ubar
    vpwp_sfc = -vm_bot * Cm * ubar

    return wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc, T_sfc


def rico_tndcy(state: dict) -> None:
    """RICO analytic large-scale tendencies (rico.F90:rico_tndcy). Does NOT touch `wm` (subsidence is
    init-set and left unchanged). Both forcings are time-independent.

    thlm_forcing = t_tendency / exner, where t_tendency [K/s]:
        z < 4000 :  -2.51/86400 + (0.33/(86400·4000))·z
        4000≤z<5000: -2.18/86400 + (2.18/(86400·1000))·(z-4000)
        z ≥ 5000 :  0
    qtm_forcing (specific-humidity tendency [g/kg/s], then /g_per_kg → kg/kg/s):
        z < 3000 :  -1.0/86400 + (1.345/(86400·3000))·z
        3000≤z<4000: 0.345/86400
        4000≤z<5000: 0.345/86400 + (-0.345/(86400·1000))·(z-4000)
        z ≥ 5000 :  0
    rtm_forcing = (1 + rtm)^2 · qtm_forcing   (spec_hum_to_mixing_ratio:force_spec_hum_to_mixing_ratio).
    """
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    exner = np.asarray(state['exner'], dtype=np.float64)
    rtm = np.asarray(state['rtm'], dtype=np.float64)
    day = 86400.0
    # temperature tendency (K/s), 3-piece
    t_tendency = np.where(
        zt < 4000.0, -2.51 / day + ((-2.18 + 2.51) / (day * 4000.0)) * zt,
        np.where(zt < 5000.0, -2.18 / day + (2.18 / (day * (5000.0 - 4000.0))) * (zt - 4000.0),
                 0.0))
    state['thlm_forcing'][:] = t_tendency / exner
    # specific-humidity tendency (g/kg/s), 4-piece, then → kg/kg/s
    qtm_forcing = np.where(
        zt < 3000.0, -1.0 / day + ((0.345 + 1.0) / (day * 3000.0)) * zt,
        np.where(zt < 4000.0, 0.345 / day,
                 np.where(zt < 5000.0, 0.345 / day + (-0.345 / (day * (5000.0 - 4000.0))) * (zt - 4000.0),
                          0.0)))
    qtm_forcing = qtm_forcing / g_per_kg                     # [g/kg/s] → [kg/kg/s] (rico.F90:135)
    # rtm_forcing = (1 + rtm)^2 * qtm_forcing, via the named routine (rico.F90 calls force_spec_hum_to_mixing_ratio)
    state['rtm_forcing'][:] = force_spec_hum_to_mixing_ratio(rtm, qtm_forcing)
