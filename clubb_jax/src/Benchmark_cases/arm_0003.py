"""JAX port of Benchmark_cases/arm_0003.F90 — the arm_0003 surface scheme.

`arm_0003_sfclyr` is algebraically IDENTICAL to `arm_97_sfclyr` (prescribed sensible/latent heat fluxes →
kinematic + the Monin-Obukhov `diag_ustar` at z0=0.035). This module reuses that validated implementation.
The arm_0003 case is otherwise unviable in the JAX driver (arm_0003 = COAMPS-fatal in the Fortran / forcings data
removed from the repo), but the surface scheme is a faithful, tested port of the oracle.
"""
from clubb_jax.src.Benchmark_cases.arm_97 import arm_97_sfclyr


def arm_0003_sfclyr(heat_flx, moisture_flx, z, rho_sfc, thlm_sfc, ubar):
    """arm_0003 surface fluxes + friction velocity (arm_0003.F90:arm_0003_sfclyr); identical to arm_97_sfclyr."""
    return arm_97_sfclyr(heat_flx, moisture_flx, z, rho_sfc, thlm_sfc, ubar)
