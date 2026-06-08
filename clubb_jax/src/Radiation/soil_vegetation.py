"""Interactive land-surface scheme — port of soil_vegetation.F90 (gabls3's lower BC, `l_soil_veg`).

A Deardorff (1978) / Duynkerke (1991) force-restore surface energy budget: given the surface radiative
fluxes (from BUGSrad) and the turbulent heat fluxes, it advances three temperatures — the vegetation
layer, the surface soil layer, and the deep soil layer — each timestep. `veg_T_in_K` is then used as the
surface temperature by the gabls3 surface-flux code (prescribe_forcings). Pure arithmetic ⇒ bit-exact.

Called (radiation_module.F90:152, when l_soil_veg) BEFORE the radiation advance, with the SURFACE slices
of the previous step's fluxes: rho_zm(:,1), Frad_SW_up(:,1), Frad_SW_down(:,1), Frad_LW_down(:,1)
(CLUBB index 1 = surface), plus wpthlp_sfc, wprtp_sfc, p_sfc. dt must be < 60 s.
"""
import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp as _CP, Lv as _LV, kappa as _KAPPA, p0 as _P0,
    stefan_boltzmann as _STEFAN_BOLTZMANN,   # soil_vegetation.F90:84 `use constants_clubb`
)

_PI = math.pi

# soil parameters (soil_vegetation.F90:154-162) — heat capacity, density, diffusivity + force-restore coeffs
_CS = 2.00e3
_RS = 1.00e3
_KS = 2.00e-7
_D1 = math.sqrt(_KS * 3600.0 * 24.0)
_C1 = 2.0 * math.sqrt(_PI) / (_RS * _CS * _D1)
_C2 = 2.0 * _PI / (3600.0 * 24.0)                                    # Ω
_C3 = math.sqrt(_PI * 2.0) / (math.exp(_PI / 4.0) * _RS * _CS *
                              math.sqrt(_KS * 3600.0 * 24.0 * 365.0))


def advance_soil_veg(dt, rho_sfc, Frad_SW_up_sfc, Frad_SW_down_sfc, Frad_LW_down_sfc,
                     wpthlp_sfc, wprtp_sfc, p_sfc, deep_soil_T_in_K, sfc_soil_T_in_K, veg_T_in_K):
    """Advance the soil/vegetation temperatures one step (soil_vegetation.F90:advance_soil_veg). All
    arrays (ngrdcol,). Returns (deep_soil_T_in_K, sfc_soil_T_in_K, veg_T_in_K, soil_heat_flux). The three
    updates all use the OLD temperatures (soil_heat_flux/veg_heat_flux are formed first), so the order is
    immaterial — the functional form is bit-identical to the Fortran in-place loop."""
    a = lambda x: jnp.asarray(x, dtype=jnp.float64)
    rho_sfc, p_sfc = a(rho_sfc), a(p_sfc)
    Frad_SW_up_sfc, Frad_SW_down_sfc, Frad_LW_down_sfc = a(Frad_SW_up_sfc), a(Frad_SW_down_sfc), a(Frad_LW_down_sfc)
    wpthlp_sfc, wprtp_sfc = a(wpthlp_sfc), a(wprtp_sfc)
    deep_soil_T, sfc_soil_T, veg_T = a(deep_soil_T_in_K), a(sfc_soil_T_in_K), a(veg_T_in_K)

    Frad_LW_up_sfc = _STEFAN_BOLTZMANN * veg_T ** 4
    wpthep = wpthlp_sfc + (_LV / _CP) * (_P0 / p_sfc) ** _KAPPA * wprtp_sfc
    veg_heat_flux = (Frad_LW_down_sfc - Frad_LW_up_sfc - wpthep * rho_sfc * _CP
                     + (Frad_SW_down_sfc - Frad_SW_up_sfc))
    soil_heat_flux = 10.0 * (veg_T - sfc_soil_T) + 0.05 * Frad_SW_down_sfc

    veg_T_new = veg_T + dt * 5.0e-5 * (veg_heat_flux - soil_heat_flux)
    sfc_soil_T_new = sfc_soil_T + dt * (_C1 * soil_heat_flux - _C2 * (sfc_soil_T - deep_soil_T))
    deep_soil_T_new = deep_soil_T + dt * _C3 * soil_heat_flux
    return deep_soil_T_new, sfc_soil_T_new, veg_T_new, soil_heat_flux


def initialize_soil_veg(ngrdcol):
    """Default soil/vegetation temperatures (soil_vegetation.F90:initialize_soil_veg).
    Returns (deep_soil_T_in_K, sfc_soil_T_in_K, veg_T_in_K), each (ngrdcol,)."""
    import numpy as np
    return (np.full(ngrdcol, 288.58), np.full(ngrdcol, 300.0), np.full(ngrdcol, 300.0))
