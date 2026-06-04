"""JAX port of Benchmark_cases/arm_3year.F90 — the arm_3year surface scheme.

`arm_3year_sfclyr` is algebraically IDENTICAL to `arm_97_sfclyr` (prescribed sensible/latent heat fluxes →
kinematic + the Monin-Obukhov `diag_ustar` at z0=0.035). This module reuses that validated implementation.
The arm_3year case is otherwise unviable in the JAX driver (arm_3year = COAMPS-fatal in the Fortran / forcings data
removed from the repo), but the surface scheme is a faithful, tested port of the oracle.
"""
from clubb_jax.src.Benchmark_cases.arm_97 import arm_97_sfclyr


def arm_3year_sfclyr(heat_flx, moisture_flx, z, rho_sfc, thlm_sfc, ubar):
    """arm_3year surface fluxes + friction velocity (arm_3year.F90:arm_3year_sfclyr); identical to arm_97_sfclyr."""
    return arm_97_sfclyr(heat_flx, moisture_flx, z, rho_sfc, thlm_sfc, ubar)
