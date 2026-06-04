"""JAX port of Benchmark_cases/twp_ice.F90 — the TWP-ICE surface scheme.

`twp_ice_sfclyr` is algebraically IDENTICAL to the CGILS `cloud_feedback_sfclyr` bulk drag-law scheme (the same
RICO/ATEX drag coefficients, the same `compute_wpthlp_sfc`/`compute_wprtp_sfc`, ustar = 0.3) — it differs only in
taking `exner_sfc` directly rather than recomputing it from p_sfc (and exner_sfc = (p_sfc/p0)^kappa, so the two
agree). This module reuses the validated implementation. The TWP-ICE case is otherwise blocked by SILHS.
"""
from clubb_jax.src.Benchmark_cases.cloud_feedback import cloud_feedback_sfclyr


def twp_ice_sfclyr(thlm_sfc, rtm_sfc, lowest_level, ubar, p_sfc, T_sfc, saturation_formula):
    """TWP-ICE surface fluxes — the RICO drag law (twp_ice.F90:twp_ice_sfclyr); identical to cloud_feedback_sfclyr.

    Returns (wpthlp_sfc, wprtp_sfc, ustar). T_sfc is the time-interpolated sea-surface temperature.
    """
    return cloud_feedback_sfclyr(thlm_sfc, rtm_sfc, lowest_level, ubar, p_sfc, T_sfc, saturation_formula)
