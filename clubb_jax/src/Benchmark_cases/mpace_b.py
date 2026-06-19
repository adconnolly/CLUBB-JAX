"""JAX port of Benchmark_cases/mpace_b.F90 — the M-PACE B (Arctic mixed-phase) large-scale forcing.

`mpace_b_tndcy` prescribes a divergence-driven subsidence (capped above the inversion) plus an analytic
radiative-cooling thlm tendency and a moisture tendency, all functions of pressure. Self-contained +
differentiable. The M-PACE B *case* is otherwise blocked by interactive SILHS, but this forcing is a faithful,
tested port of the oracle.
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Rd, Cp, grav, Lv
from clubb_jax.src.CLUBB_core.grid_class import zt2zm

_D = 5.8e-6        # large-scale divergence [1/s]
_P_SFC = 101000.0  # reference surface pressure [Pa]
_PINV = 85000.0    # inversion pressure [Pa]
_SEC_PER_DAY = 86400.0
_G_PER_KG = 1000.0


def mpace_b_tndcy(p_in_Pa, thvm, gr):
    """Large-scale subsidence + thlm/rtm forcing for M-PACE B (mpace_b.F90:mpace_b_tndcy).

    Args:
        p_in_Pa: pressure on thermodynamic levels (ngrdcol, nzt) [Pa].
        thvm:    virtual potential temperature (ngrdcol, nzt) [K].
        gr:      JAX grid.
    Returns:
        dict with wm_zt (ngrdcol,nzt), wm_zm (ngrdcol,nzm), thlm_forcing (ngrdcol,nzt), rtm_forcing (ngrdcol,nzt).
    """
    p = jnp.asarray(p_in_Pa)
    thvm = jnp.asarray(thvm)
    # Subsidence: omega = min(D(p_sfc-p), D(p_sfc-pinv)) (capped above the inversion); wm = -omega Rd thvm /(p g).
    velocity_omega = jnp.minimum(_D * (_P_SFC - p), _D * (_P_SFC - _PINV))
    wm_zt = -velocity_omega * Rd * thvm / p / grav
    wm_zm = zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, wm_zt)
    wm_zm = wm_zm.at[:, 0].set(0.0).at[:, -1].set(0.0)   # surface + top BCs

    # Radiative cooling thlm tendency [K/s]: min(-4, -15(1-(p_sfc-p)/21818)) K/day × exner factor.
    t_tendency = jnp.minimum(-4.0, -15.0 * (1.0 - (_P_SFC - p) / 21818.0))   # K/day
    thlm_forcing = (t_tendency * (_P_SFC / p) ** (Rd / Cp)) / _SEC_PER_DAY
    # Moisture tendency [kg/kg/s]: min(0.164, -3(1-(p_sfc-p)/15171)) g/kg/day.
    rtm_forcing = jnp.minimum(0.164, -3.0 * (1.0 - (_P_SFC - p) / 15171.0)) / _G_PER_KG / _SEC_PER_DAY
    return {'wm_zt': wm_zt, 'wm_zm': wm_zm, 'thlm_forcing': thlm_forcing, 'rtm_forcing': rtm_forcing}


def mpace_b_sfclyr(sensible_heat_flx, latent_heat_flx, rho_sfc):
    """M-PACE B surface fluxes (mpace_b.F90:mpace_b_sfclyr) — prescribed (time-interpolated) sensible/latent heat
    fluxes converted to kinematic form, with a FIXED ustar = 0.25 m/s. Differentiable.

    Args:
        sensible_heat_flx: surface sensible heat flux [W/m^2] (time-interpolated by the caller).
        latent_heat_flx:   surface latent heat flux   [W/m^2].
        rho_sfc:           surface air density        [kg/m^3].
    Returns:
        (wpthlp_sfc, wprtp_sfc, ustar).
    """
    wpthlp_sfc = sensible_heat_flx / (rho_sfc * Cp)     # W/m^2 -> K m/s
    wprtp_sfc = latent_heat_flx / (rho_sfc * Lv)        # W/m^2 -> kg/kg m/s
    ustar = jnp.broadcast_to(0.25, jnp.shape(rho_sfc))
    return wpthlp_sfc, wprtp_sfc, ustar
