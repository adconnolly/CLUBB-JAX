"""JAX port of Benchmark_cases/lba.F90 — the LBA (Amazon deep-convection) surface scheme.

`lba_sfclyr` prescribes a diurnal sensible/latent surface heat flux (peaking ~5.25 h after start) and derives the
friction velocity from the resulting buoyancy flux via the Monin-Obukhov `diag_ustar`. Self-contained +
differentiable. The LBA *case* is otherwise blocked by interactive SILHS, but this surface scheme is a faithful,
tested port of the oracle.
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv, grav, sec_per_hr
from clubb_jax.src.Benchmark_cases.diag_ustar_module import diag_ustar

_Z0 = 0.035        # ARM momentum roughness height [m]
_PEAK_HR = 5.25    # diurnal peak time [h] (lba.F90 magic number)


def lba_tndcy(shape):
    """LBA large-scale forcing (lba.F90:lba_tndcy) — identically ZERO (LBA deep convection is surface-driven,
    no large-scale thlm/rtm tendency). `shape` is (ngrdcol, nzt). Returns (thlm_forcing, rtm_forcing)."""
    z = jnp.zeros(shape)
    return z, z


def lba_diurnal_factor(time_elapsed):
    """Diurnal forcing factor ft = max(0, cos(½π (5.25 - t_hr)/5.25)) (lba.F90:lba_sfclyr)."""
    t_hr = time_elapsed / sec_per_hr
    return jnp.maximum(0.0, jnp.cos(0.5 * jnp.pi * (_PEAK_HR - t_hr) / _PEAK_HR))


def lba_sfclyr(time_elapsed, z, rho_sfc, thlm_sfc, ubar):
    """LBA surface heat/moisture fluxes + friction velocity (lba.F90:lba_sfclyr).

    Args (per column, broadcastable):
        time_elapsed: time since model start [s].
        z:            lowest model level height [m].
        rho_sfc:      surface air density [kg/m^3].
        thlm_sfc:     thlm at the lowest level [K].
        ubar:         sqrt(u^2+v^2) at the surface [m/s].
    Returns:
        (wpthlp_sfc, wprtp_sfc, ustar) — heat flux [K m/s], moisture flux [kg/kg m/s], friction velocity [m/s].
    """
    ft = lba_diurnal_factor(time_elapsed)
    ft_safe = jnp.maximum(ft, 0.0)                       # ft >= 0 already; keeps the fractional power finite
    wpthlp_sfc = (270.0 * ft_safe ** 1.5) / (rho_sfc * Cp)     # sensible heat flux 270 W/m² · ft^1.5
    wprtp_sfc = (554.0 * ft_safe ** 1.3) / (rho_sfc * Lv)      # latent heat flux 554 W/m² · ft^1.3
    bflx = grav / thlm_sfc * wpthlp_sfc
    ustar = diag_ustar(z, bflx, ubar, _Z0)
    return wpthlp_sfc, wprtp_sfc, ustar
