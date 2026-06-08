"""JAX port of Benchmark_cases/astex_a209.F90 — the ASTEX-A209 surface scheme.

`astex_a209_sfclyr` computes the surface heat/moisture fluxes via the same RICO 3-D bulk-aerodynamic drag law as
the CGILS/cloud_feedback scheme (drag coefficients scaled from their 20 m reference height to the model's lowest
level, ATEX-specification log-law), but with the ASTEX friction velocity ustar = 0.155 m/s (as set in rico).

Self-contained + differentiable; the prescribed time-dependent surface temperature `T_sfc` is passed in (the caller
interpolates `T_sfc_given` in time). The ASTEX-A209 *case* is blocked end-to-end, but this surface scheme mirrors
the oracle (cf. the gate-validated sibling cloud_feedback.py:cloud_feedback_sfclyr).
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import p0, kappa
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc

_C_H_20 = 0.001094     # RICO 3-D heat drag coefficient at 20 m
_C_Q_20 = 0.001133     # RICO 3-D moisture drag coefficient at 20 m
_Z0 = 0.00015          # ATEX roughness length [m]
_STD_FLUX_ALT = 20.0   # reference height for the drag coefficients [m]
_USTAR = 0.155         # fixed surface friction velocity [m/s] (astex_a209.F90:212, "set as it is set in rico")


def astex_a209_sfclyr(thlm_sfc, rtm_sfc, lowest_level, ubar, p_sfc, T_sfc, saturation_formula):
    """Surface heat/moisture fluxes for the ASTEX-A209 case (astex_a209.F90:astex_a209_sfclyr).

    Args (per column, broadcastable arrays): thlm_sfc [K], rtm_sfc [kg/kg], lowest_level [m], ubar [m/s],
    p_sfc [Pa], T_sfc (time-interpolated) [K], saturation_formula (SATURATION_* integer).
    Returns (wpthlp_sfc [m K/s], wprtp_sfc [m kg/(s kg)], ustar [m/s]).
    """
    exner_sfc = (p_sfc / p0) ** kappa
    log_ratio = jnp.log(_STD_FLUX_ALT / _Z0) / jnp.log(lowest_level / _Z0)
    scale = log_ratio ** 2
    Ch = _C_H_20 * scale
    Cq = _C_Q_20 * scale
    rsat = sat_mixrat_liq(p_sfc, T_sfc, saturation_formula)
    wpthlp_sfc = compute_wpthlp_sfc(Ch, ubar, thlm_sfc, T_sfc, exner_sfc)
    wprtp_sfc = compute_wprtp_sfc(Cq, ubar, rtm_sfc, rsat)
    ustar = jnp.broadcast_to(_USTAR, jnp.shape(p_sfc))
    return wpthlp_sfc, wprtp_sfc, ustar
