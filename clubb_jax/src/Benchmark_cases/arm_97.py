"""JAX port of Benchmark_cases/arm_97.F90 — the ARM-97 surface scheme.

`arm_97_sfclyr` converts the prescribed (time-interpolated) sensible/latent surface heat fluxes to kinematic
form and derives the friction velocity from the resulting buoyancy flux via the Monin-Obukhov `diag_ustar`.
Self-contained + differentiable. The ARM-97 *case* is otherwise blocked by interactive SILHS + Morrison/bugsrad,
but this surface scheme is a faithful, tested port of the oracle.
"""
from clubb_jax.src.CLUBB_core.constants_clubb import grav
from clubb_jax.src.Benchmark_cases.diag_ustar_module import diag_ustar
from clubb_jax.src.Benchmark_cases.sfc_flux import convert_sens_ht_to_km_s, convert_latent_ht_to_m_s

_Z0 = 0.035   # ARM Cu momentum roughness height [m]


def arm_97_sfclyr(heat_flx, moisture_flx, z, rho_sfc, thlm_sfc, ubar):
    """ARM-97 surface heat/moisture fluxes + friction velocity (arm_97.F90:arm_97_sfclyr).

    Args (per column, broadcastable):
        heat_flx:     surface sensible heat flux [W/m^2] (time-interpolated by the caller).
        moisture_flx: surface latent heat flux   [W/m^2].
        z:            lowest model level height   [m].
        rho_sfc:      surface air density         [kg/m^3].
        thlm_sfc:     thlm at the lowest level    [K].
        ubar:         sqrt(u^2+v^2) at the surface [m/s].
    Returns:
        (wpthlp_sfc, wprtp_sfc, ustar).
    """
    wpthlp_sfc = convert_sens_ht_to_km_s(heat_flx, rho_sfc)     # W/m^2 -> K m/s
    wprtp_sfc = convert_latent_ht_to_m_s(moisture_flx, rho_sfc)  # W/m^2 -> kg/kg m/s
    bflx = grav / thlm_sfc * wpthlp_sfc
    ustar = diag_ustar(z, bflx, ubar, _Z0)
    return wpthlp_sfc, wprtp_sfc, ustar
