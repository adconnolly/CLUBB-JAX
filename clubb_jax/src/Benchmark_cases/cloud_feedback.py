"""JAX port of Benchmark_cases/cloud_feedback.F90 — the CGILS / cloud-feedback surface scheme.

`cloud_feedback_sfclyr` computes the surface heat/moisture fluxes for the 12 CGILS / cloud-feedback cases via a
bulk aerodynamic (drag-law) formula with RICO/ATEX drag coefficients scaled from their 20 m reference height to
the model's lowest level. ustar is held at a constant 0.3 m/s.

Self-contained + differentiable. The prescribed time-dependent surface temperature `T_sfc` is passed in (the
caller interpolates `T_sfc_given` in time), decoupling this flux physics from the time-dependent-input reader.
Validated in `tests/test_cloud_feedback_sfclyr.py` against a literal NumPy transcription + physical invariants.

NOTE: the cloud_feedback *cases* are not yet gateable — their time-dependent CGILS large-scale forcing reader
diverges in the JAX driver — but this surface scheme is a faithful, tested port of the oracle.
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import p0, kappa
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_wpthlp_sfc, compute_wprtp_sfc

_C_H_20 = 0.001094     # RICO 3-D heat drag coefficient at 20 m
_C_Q_20 = 0.001133     # RICO 3-D moisture drag coefficient at 20 m
_Z0 = 0.00015          # ATEX roughness length [m]
_STD_FLUX_ALT = 20.0   # reference height for the drag coefficients [m]
_USTAR = 0.3           # fixed surface friction velocity [m/s]


def cloud_feedback_sfclyr(thlm_sfc, rtm_sfc, lowest_level, ubar, p_sfc, T_sfc, saturation_formula):
    """Surface heat/moisture fluxes for the CGILS/cloud-feedback cases (cloud_feedback.F90:cloud_feedback_sfclyr).

    Args (per column, broadcastable arrays):
        thlm_sfc:           thlm at the lowest level         [K].
        rtm_sfc:            rtm at the lowest level          [kg/kg].
        lowest_level:       height of the lowest model level [m].
        ubar:               sqrt(u^2+v^2) at the surface     [m/s].
        p_sfc:              surface pressure                 [Pa].
        T_sfc:              (time-interpolated) surface temperature [K].
        saturation_formula: SATURATION_* integer for the saturation mixing ratio.
    Returns:
        (wpthlp_sfc, wprtp_sfc, ustar) — heat flux [m K/s], moisture flux [m kg/(s kg)], friction velocity [m/s].
    """
    exner_sfc = (p_sfc / p0) ** kappa
    # Drag coefficients scaled from 20 m to the lowest model level (log-law).
    log_ratio = jnp.log(_STD_FLUX_ALT / _Z0) / jnp.log(lowest_level / _Z0)
    scale = log_ratio ** 2
    Ch = _C_H_20 * scale
    Cq = _C_Q_20 * scale
    rsat = sat_mixrat_liq(p_sfc, T_sfc, saturation_formula)
    wpthlp_sfc = compute_wpthlp_sfc(Ch, ubar, thlm_sfc, T_sfc, exner_sfc)
    wprtp_sfc = compute_wprtp_sfc(Cq, ubar, rtm_sfc, rsat)
    ustar = jnp.broadcast_to(_USTAR, jnp.shape(p_sfc))
    return wpthlp_sfc, wprtp_sfc, ustar
