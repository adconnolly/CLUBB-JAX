"""JAX port of sfc_flux.F90 — shared surface-flux utilities for the benchmark cases.

Mirrors clubb_release/src/Benchmark_cases/sfc_flux.F90. Holds the surface-flux helper routines shared by the
case `*_sfclyr` routines in Benchmark_cases/prescribe_forcings.py, mirroring the Fortran `use sfc_flux` there:
  compute_ht_mostr_flux  — time-interpolate the prescribed ARM surface sensible/latent heat fluxes [W/m^2]
  compute_ubar           — mean surface wind speed, lower-bounded by ubmin
  compute_momentum_flux  — surface momentum fluxes upwp_sfc/vpwp_sfc from ustar, ubar
  compute_wpthlp_sfc     — bulk-aerodynamic surface heat flux  (-Cd·ubar·(thlm − T_sfc/exner))
  compute_wprtp_sfc      — bulk-aerodynamic surface moisture flux (-Cd·ubar·(rtm − adjustment))
  set_sclr_sfc_rtm_thlm  — copy the thl/rt surface fluxes into the passive-scalar surface-flux array
  convert_sens_ht_to_km_s / convert_latent_ht_to_m_s — heat-flux [W/m^2] → kinematic flux unit conversions

All sfc_flux.F90 routines now have a JAX mirror here (the case `*_sfclyr` paths call them via `use sfc_flux`).

Tracer-transparent: concrete runs keep the exact numpy path (bit-identical); under a jax.grad trace the helpers
route to jnp via the tracer_numpy toolkit so the surface fluxes stay differentiable.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg, _safe_sqrt, _iset
from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv

_UBMIN = 0.25   # m/s (compute_ubar ubmin)


def convert_sens_ht_to_km_s(sens_ht, rho_sfc):
    """sfc_flux.F90:convert_sens_ht_to_km_s — sensible heat flux [W/m^2] → kinematic
    surface w'thl' flux [K m/s]: sens_ht / (rho_sfc * Cp)."""
    return sens_ht / (rho_sfc * Cp)


def convert_latent_ht_to_m_s(latent_ht, rho_sfc):
    """sfc_flux.F90:convert_latent_ht_to_m_s — latent heat flux [W/m^2] → kinematic
    surface w'rt' flux [(kg/kg) m/s]: latent_ht / (rho_sfc * Lv)."""
    return latent_ht / (rho_sfc * Lv)


def compute_ht_mostr_flux(time, time_sfc_given, sens_ht_given, latent_ht_given):
    """Time-interpolate the prescribed surface heat/moisture fluxes from ARM-style data
    (sfc_flux.F90:compute_ht_mostr_flux). Returns (heat_flx, moisture_flx) in W/m^2, with flat
    extrapolation outside the table (linear_interp_factor inside it). time is concrete (the surface
    fluxes depend only on the prescribed schedule, not the prognostic state)."""
    def _interp(values):
        if time <= time_sfc_given[0]:
            return float(values[0])
        if time >= time_sfc_given[-1]:
            return float(values[-1])
        i = int(np.searchsorted(time_sfc_given, time, side='right')) - 1
        i = min(i, len(time_sfc_given) - 2)
        frac = (time - time_sfc_given[i]) / (time_sfc_given[i + 1] - time_sfc_given[i])
        return float(values[i] + frac * (values[i + 1] - values[i]))
    return _interp(sens_ht_given), _interp(latent_ht_given)


def compute_wpthlp_sfc(Cd, ubar, thlm_sfc, T_sfc, exner_sfc):
    """Bulk-aerodynamic surface heat flux (sfc_flux.F90:compute_wpthlp_sfc):
    wpthlp_sfc = -Cd * ubar * (thlm_sfc - T_sfc / exner_sfc). Pure arithmetic → tracer-transparent."""
    return -Cd * ubar * (thlm_sfc - T_sfc / exner_sfc)


def compute_wprtp_sfc(Cd, ubar, rtm_sfc, adjustment):
    """Bulk-aerodynamic surface moisture flux (sfc_flux.F90:compute_wprtp_sfc):
    wprtp_sfc = -Cd * ubar * (rtm_sfc - adjustment). Pure arithmetic → tracer-transparent."""
    return -Cd * ubar * (rtm_sfc - adjustment)


def compute_ubar(um_bot: np.ndarray, vm_bot: np.ndarray) -> np.ndarray:
    """Mean surface wind speed, lower-bounded by ubmin=0.25 m/s (sfc_flux.F90:compute_ubar).
    Tracer-transparent (REFACTOR B5): concrete runs keep the exact numpy path (bit-identical); under a jax.grad
    trace use _safe_sqrt so a near-zero wind doesn't give a nan gradient (the `maximum` would otherwise still
    propagate it)."""
    if _is_tracer_arg([um_bot, vm_bot]):
        return jnp.maximum(_UBMIN, _safe_sqrt(um_bot ** 2 + vm_bot ** 2))
    return np.maximum(_UBMIN, np.sqrt(um_bot ** 2 + vm_bot ** 2))


def compute_momentum_flux(um_bot: np.ndarray, vm_bot: np.ndarray,
                              ubar: np.ndarray, ustar: np.ndarray):
    """upwp_sfc = -um * ustar^2 / ubar;  vpwp_sfc = -vm * ustar^2 / ubar (sfc_flux.F90:compute_momentum_flux)."""
    ustar2 = ustar ** 2
    upwp_sfc = -um_bot * ustar2 / ubar
    vpwp_sfc = -vm_bot * ustar2 / ubar
    return upwp_sfc, vpwp_sfc


def set_sclr_sfc_rtm_thlm(state: dict, wpthlp_sfc: np.ndarray,
                              wprtp_sfc: np.ndarray) -> None:
    """Copy wpthlp/wprtp to scalar surface fluxes (sfc_flux.F90:set_sclr_sfc_rtm_thlm)."""
    sclr_idx = state['sclr_idx']
    nzm = state['nzm']
    ngrdcol = state['ngrdcol']
    if state['sclr_dim'] > 0:
        if sclr_idx.iisclr_thl > 0:
            k = sclr_idx.iisclr_thl - 1
            state['wpsclrp'] = _iset(state['wpsclrp'], np.s_[:, :, k],
                                     wpthlp_sfc[:, np.newaxis] * np.ones((ngrdcol, nzm)))
        if sclr_idx.iisclr_rt > 0:
            k = sclr_idx.iisclr_rt - 1
            state['wpsclrp'] = _iset(state['wpsclrp'], np.s_[:, :, k],
                                     wprtp_sfc[:, np.newaxis] * np.ones((ngrdcol, nzm)))
