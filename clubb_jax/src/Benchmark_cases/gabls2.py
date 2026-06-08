"""JAX port of gabls2.F90 — GABLS2 case large-scale tendencies + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/gabls2.F90: `gabls2_tndcy` (analytic subsidence, off for the first
26 h) and `gabls2_sfclyr` (analytic diurnal surface temperature → bulk-aerodynamic heat/moisture fluxes,
diag_ustar friction velocity). prescribe_forcings.py imports these, mirroring the Fortran case dispatch's
`use gabls2`.

Tracer-transparent: concrete runs keep the exact numpy path (bit-identical); under a jax.grad trace the
diag_ustar friction velocity routes to its vectorized differentiable mirror.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc
# Physical constants — mirror the Fortran gabls2.F90 `use constants_clubb, only: Cp, Rd, p0, grav, sec_per_hr`
from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Rd, p0, grav, sec_per_hr as _SEC_PER_HR


def gabls2_tndcy(state: dict, time_current: float) -> None:
    """GABLS2 analytic large-scale tendencies (gabls2.F90:gabls2_tndcy). thlm/rtm forcing = 0; subsidence
    is OFF for the first 26 h then `wm_zt = -0.005·min(zt/1000, 1)`; `wm_zm = zt2zm(wm_zt)` with zeroed
    top/surface BCs. (Does not touch ug/vg/um_ref — they keep their sounding/init values, as in Fortran.)"""
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0
    if time_current > state['time_initial'] + 93600.0:
        wm_zt = np.where(zt <= 1000.0, -0.005 * (zt / 1000.0), -0.005)
    else:
        wm_zt = np.zeros_like(zt)
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm_jax(jnp.asarray(wm_zt), gr), dtype=np.float64)   # np.array → writable copy
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def gabls2_sfclyr(state: dict, time_current: float, ngrdcol: int,
                      z_bot, p_sfc_arr, ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """GABLS2 surface fluxes (gabls2.F90:gabls2_sfclyr).

    Analytic T_sfc formula (piecewise cosine/linear in local hours starting at 14h).
    Bulk aerodynamic with C_10 scaled to model height, diag_ustar.
    """
    from clubb_jax.src.Benchmark_cases.diag_ustar_module import _diag_ustar
    C_10 = 0.0013      # case-specific drag coefficient
    z0   = 0.03        # case-specific roughness height [m]
    z_ref = 10.0       # case-specific reference height [m]
    Rd_over_Cp = Rd / Cp   # = kappa (constants_clubb); p0/grav/Rd/Cp now from constants_clubb

    time_in_hours = 14.0 + (time_current - state['time_initial']) / _SEC_PER_HR

    if time_in_hours <= 17.4:
        T_sfc_C = -10.0 - 25.0 * math.cos(time_in_hours * 0.22 + 0.2)
    elif time_in_hours <= 30.0:
        T_sfc_C = -0.54 * time_in_hours + 15.2
    elif time_in_hours <= 41.9:
        T_sfc_C = -7.0 - 25.0 * math.cos(time_in_hours * 0.21 + 1.8)
    elif time_in_hours <= 53.3:
        T_sfc_C = -0.37 * time_in_hours + 18.0
    elif time_in_hours <= 65.6:
        T_sfc_C = -4.0 - 25.0 * math.cos(time_in_hours * 0.22 + 2.5)
    else:
        T_sfc_C = 4.4
    T_sfc_val = T_sfc_C + 273.15

    T_sfc  = np.full(ngrdcol, T_sfc_val)
    sat_formula = state['flags'].saturation_formula
    rsat = np.array([float(sat_mixrat_liq(float(p_sfc_arr[i]), T_sfc_val, sat_formula))
                     for i in range(ngrdcol)])

    log_ref_z0 = math.log(z_ref / z0)
    log_z_z0   = np.log(np.maximum(z_bot, z0 * 1.001) / z0)
    Cz = C_10 * (log_ref_z0 / log_z_z0) ** 2

    wpthlp_sfc = compute_wpthlp_sfc(Cz, ubar, thlm_bot, T_sfc, exner_bot)
    wprtp_sfc  = compute_wprtp_sfc(Cz, ubar, rtm_bot, rsat)
    wprtp_sfc  = wprtp_sfc * 0.025          # gabls2.F90:299: latent heat flux is 2.5% of its potential

    sstheta = T_sfc * ((p0 / p_sfc_arr) ** Rd_over_Cp)
    bflx_arr = wpthlp_sfc * grav / sstheta
    # Tracer dispatch (REFACTOR B5): concrete = exact per-column _diag_ustar loop; trace = jnp mirror.
    if not _is_tracer_arg([bflx_arr, ubar]):
        ustar = np.array([_diag_ustar(float(z_bot[i]), float(bflx_arr[i]),
                                       float(ubar[i]), z0)
                          for i in range(ngrdcol)])
    else:
        from clubb_jax.src.Benchmark_cases.diag_ustar_module import diag_ustar
        ustar = diag_ustar(jnp.asarray(z_bot), bflx_arr, jnp.asarray(ubar), z0)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc
