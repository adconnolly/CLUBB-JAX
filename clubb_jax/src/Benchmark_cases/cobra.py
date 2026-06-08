"""JAX port of cobra.F90 — the COBRA case surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/cobra.F90: `cobra_sfclyr` — the ARM-variant surface layer (prescribed,
time-interpolated sensible/latent heat fluxes converted to kinematic form via `sfc_flux:convert_*_ht`, friction
velocity from the Monin-Obukhov `diag_ustar`) but with the COBRA momentum roughness height z0 = 1.75 m (not the
ARM 0.035 m). The CO2 passive-scalar surface flux is not ported. prescribe_forcings.py imports it, mirroring the
Fortran case dispatch's `use cobra`.

Tracer-transparent: concrete runs keep the exact per-column diag_ustar loop (bit-identical); under a jax.grad
trace ustar routes to its vectorized differentiable mirror.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg
from clubb_jax.src.Benchmark_cases.diag_ustar_module import _diag_ustar, diag_ustar
from clubb_jax.src.Benchmark_cases.sfc_flux import (
    convert_sens_ht_to_km_s, convert_latent_ht_to_m_s)
from clubb_jax.src.CLUBB_core.constants_clubb import grav as _GRAV  # mirror cobra.F90 `use constants_clubb, only: grav`

_Z0 = 1.75   # COBRA momentum roughness height [m] (cobra.F90:119)


def cobra_sfclyr(state: dict, time_current: float, ngrdcol: int,
                     z_bot, rho_bot, thlm_bot, ubar) -> tuple:
    """COBRA surface fluxes (cobra.F90:cobra_sfclyr).

    Reads sens_ht/latent_ht from the sfc file (W/m^2), converts to kinematic form using rho_sfc, then computes
    ustar via diag_ustar with z0=1.75 m. Returns (wpthlp_sfc, wprtp_sfc, ustar).
    """
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))

    def _interp_sfc(key, default):
        arr = sfc.get(key)
        if arr is not None and len(arr) > 0:
            return float(np.interp(time_current, times, arr))
        return default

    sens_ht   = _interp_sfc('sens_ht',   0.0)
    latent_ht = _interp_sfc('latent_ht', 0.0)

    wpthlp_sfc = convert_sens_ht_to_km_s(sens_ht, rho_bot)
    wprtp_sfc  = convert_latent_ht_to_m_s(latent_ht, rho_bot)

    bflx = _GRAV / thlm_bot * wpthlp_sfc
    # Tracer dispatch (REFACTOR B5): concrete = exact per-column _diag_ustar loop; trace = jnp mirror.
    if not _is_tracer_arg([bflx, ubar]):
        ustar = np.array([_diag_ustar(float(z_bot[i]), float(bflx[i]),
                                       float(ubar[i]), _Z0)
                          for i in range(ngrdcol)])
    else:
        ustar = diag_ustar(jnp.asarray(z_bot), bflx, jnp.asarray(ubar), _Z0)
    return wpthlp_sfc, wprtp_sfc, ustar
