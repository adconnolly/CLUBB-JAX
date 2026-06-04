"""JAX port of update_xp2_mc (advance_xp2_xpyp_module.F90:6176).

Effects of rain evaporation on the second moments (rtp2, thlp2, wprtp, wpthlp, rtpthlp) when l_morr_xp2_mc is
on. Rain is assumed to fall through the moist (cold) portion of the PDF, treated as a double-delta with a
precipitation fraction; evaporation makes the moist component moister and the cold component colder. The moment
tendencies are computed on zt levels and interpolated to zm. Pure-jnp (top-down fill via lax.scan,
zt2zm interpolation) → differentiable.

l_morr_xp2_mc defaults to .false. and no gated case uses it; this is a completeness port validated against a
literal NumPy transcription (the formulas are explicit) plus the independently-validated zt2zm interpolation.
"""
import jax
import jax.numpy as jnp
from jax import lax

from clubb_jax.src.CLUBB_core.constants_clubb import cloud_frac_min, Cp, Lv
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax

jax.config.update("jax_enable_x64", True)


def _precip_frac_double_delta(cloud_frac):
    """Top-down fill: precip_frac = cloud_frac where cloudy, else inherit the value from the level above; the
    top level is 0 (ascending grid: index 0 = bottom, nzt-1 = top)."""
    cf_rev = cloud_frac[:, ::-1]          # index 0 = top
    ng = cloud_frac.shape[0]

    def step(carry, cf_lev):
        pf = jnp.where(cf_lev > cloud_frac_min, cf_lev, carry)
        return pf, pf

    _, rest = lax.scan(step, jnp.zeros(ng), cf_rev[:, 1:].T)   # levels 1..nzt-1 (top-first)
    pf_rev = jnp.concatenate([jnp.zeros((1, ng)), rest], axis=0)   # (nzt, ng), top-first
    return pf_rev.T[:, ::-1]              # (ng, nzt), bottom-first


def update_xp2_mc(gr, dt, cloud_frac, rcm, rvm, thlm, wm, exner, rrm_evap, pdf_params):
    """Rain-evaporation tendencies of the second moments on zm levels (advance_xp2_xpyp_module.F90:update_xp2_mc).

    Array inputs are (ngrdcol, nzt). pdf_params is a mapping (or object) providing mixt_frac, rt_1/rt_2,
    varnce_rt_1/2, thl_1/2, varnce_thl_1/2, w_1/2, varnce_w_1/2 (each (ngrdcol, nzt)). rrm_evap is the (negative)
    rain-evaporation rate. Returns (rtp2_mc, thlp2_mc, wprtp_mc, wpthlp_mc, rtpthlp_mc), each (ngrdcol, nzm)."""
    def _g(name):
        return jnp.asarray(pdf_params[name] if isinstance(pdf_params, dict) else getattr(pdf_params, name),
                           dtype=jnp.float64)

    cloud_frac = jnp.asarray(cloud_frac, dtype=jnp.float64)
    rcm = jnp.asarray(rcm, dtype=jnp.float64); rvm = jnp.asarray(rvm, dtype=jnp.float64)
    thlm = jnp.asarray(thlm, dtype=jnp.float64); wm = jnp.asarray(wm, dtype=jnp.float64)
    exner = jnp.asarray(exner, dtype=jnp.float64); rrm_evap = jnp.asarray(rrm_evap, dtype=jnp.float64)

    a = _g('mixt_frac')
    pf = _precip_frac_double_delta(cloud_frac)
    pf_const = jnp.where(pf > cloud_frac_min, (1.0 - pf) / jnp.where(pf > cloud_frac_min, pf, 1.0), 0.0)

    rt_tot = rcm + rvm
    temp_rtp2 = (a * ((_g('rt_1') - rt_tot) ** 2 + _g('varnce_rt_1'))
                 + (1.0 - a) * ((_g('rt_2') - rt_tot) ** 2 + _g('varnce_rt_2')))
    temp_thlp2 = (a * ((_g('thl_1') - thlm) ** 2 + _g('varnce_thl_1'))
                  + (1.0 - a) * ((_g('thl_2') - thlm) ** 2 + _g('varnce_thl_2')))
    temp_wp2 = (a * ((_g('w_1') - wm) ** 2 + _g('varnce_w_1'))
                + (1.0 - a) * ((_g('w_2') - wm) ** 2 + _g('varnce_w_2')))

    lvcpex = Lv / (Cp * exner)
    abse = jnp.abs(rrm_evap)

    rtp2_mc_zt = rrm_evap ** 2 * pf_const * dt + 2.0 * abse * jnp.sqrt(temp_rtp2 * pf_const)
    thlp2_mc_zt = (rrm_evap * lvcpex) ** 2 * pf_const * dt + 2.0 * abse * lvcpex * jnp.sqrt(temp_thlp2 * pf_const)
    wprtp_mc_zt = abse * jnp.sqrt(pf_const) * jnp.sqrt(temp_wp2)
    wpthlp_mc_zt = -lvcpex * abse * jnp.sqrt(pf_const) * jnp.sqrt(temp_wp2)
    rtpthlp_mc_zt = (-abse * jnp.sqrt(pf_const) * (lvcpex * jnp.sqrt(temp_rtp2) + jnp.sqrt(temp_thlp2))
                     - lvcpex * pf_const * rrm_evap ** 2 * dt)

    return (zt2zm_jax(rtp2_mc_zt, gr), zt2zm_jax(thlp2_mc_zt, gr), zt2zm_jax(wprtp_mc_zt, gr),
            zt2zm_jax(wpthlp_mc_zt, gr), zt2zm_jax(rtpthlp_mc_zt, gr))
