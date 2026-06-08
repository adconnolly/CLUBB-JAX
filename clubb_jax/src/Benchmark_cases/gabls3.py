"""JAX port of gabls3.F90 — GABLS3 (daytime) surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/gabls3.F90: `gabls3_sfclyr` — the daytime GABLS3 surface layer, which
bulk-aerodynamically computes heat/moisture fluxes from the interactive vegetation temperature `veg_T_in_K`
(from the soil_vegetation scheme) as the surface temperature, with diag_ustar friction velocity.
prescribe_forcings.py imports it, mirroring the Fortran case dispatch's `use gabls3`. (The nighttime case lives in
the separate gabls3_night.py / gabls3_night.F90.)

Tracer-transparent: concrete runs keep the exact per-column diag_ustar loop (bit-identical); under a jax.grad
trace ustar routes to its vectorized differentiable mirror.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _is_tracer_arg
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc


def gabls3_sfclyr(state: dict, ngrdcol: int, ubar, thlm_bot, rtm_bot, z_bot, exner_bot) -> tuple:
    """GABLS3 (daytime) surface fluxes (gabls3.F90:gabls3_sfclyr). Bulk-aerodynamic heat/moisture fluxes
    using the interactive vegetation temperature `veg_T_in_K` (from soil_vegetation) as the surface temp:
    `wpthlp_sfc=-C_10·ubar·(thlm-veg_T/exner)`, `wprtp_sfc=-C_10·ubar·(rtm-offset)` then ×10, ustar via
    diag_ustar with the veg-based buoyancy flux. C_10=0.00195, offset=9.9e-3, z0=0.15."""
    from clubb_jax.src.Benchmark_cases.diag_ustar_module import _diag_ustar, diag_ustar
    from clubb_jax.src.CLUBB_core.constants_clubb import grav
    C_10, offset, z0 = 0.00195, 9.9e-3, 0.15
    veg_T = _asarray(state.get('veg_T_in_K', np.full(ngrdcol, 300.0)), dtype=np.float64)

    wpthlp_sfc = compute_wpthlp_sfc(C_10, ubar, thlm_bot, veg_T, exner_bot)  # T_sfc=veg_T
    wprtp_sfc = compute_wprtp_sfc(C_10, ubar, rtm_bot, offset)               # adjustment=offset
    wprtp_sfc = wprtp_sfc * 10.0                                     # gabls3.F90:116

    # ustar via diag_ustar (data-dependent MO branches + float()). Tracer dispatch (REFACTOR B5): concrete
    # runs keep the exact per-column loop (bit-identical); the jax.grad trace uses the vectorized jnp mirror.
    if not _is_tracer_arg([wpthlp_sfc, ubar]):
        ustar = np.zeros(ngrdcol)
        for i in range(ngrdcol):
            veg_theta = veg_T[i] / exner_bot[i]
            bflx = float(wpthlp_sfc[i]) * grav / veg_theta
            ustar[i] = _diag_ustar(float(z_bot[i]), bflx, float(ubar[i]), z0)
    else:
        veg_theta = veg_T / exner_bot
        bflx = wpthlp_sfc * grav / veg_theta
        ustar = diag_ustar(jnp.asarray(z_bot), bflx, jnp.asarray(ubar), z0)
    return wpthlp_sfc, wprtp_sfc, ustar
