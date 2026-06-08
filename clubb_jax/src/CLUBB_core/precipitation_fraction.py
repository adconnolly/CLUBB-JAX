"""JAX port of precipitation_fraction.F90 — overall + per-component precip fraction.

precip_fraction determines the precipitation fraction f_p (fraction of the grid box that
is precipitating) overall and within each PDF component, at every vertical level. It is a
hydrometeor-PDF input the KK rate functions need (accr/evap weight by precip_frac_i).

Algorithm (oracle precipitation_fraction.F90):
  1. precip_frac_tol[col] = max(0.1 * max(maxval cloud_frac, maxval ice_supersat_frac),
                                cloud_frac_min)   (ice term only if any frozen hydrometeor).
  2. Overall f_p = greatest cloud_frac (or max(cloud_frac, ice_supersat_frac)) AT OR ABOVE
     a level — a downward cumulative max (suffix max on an ascending grid).
  3. Special checks: any hydromet >= tol but f_p < precip_frac_tol -> f_p = precip_frac_tol;
     all hydromet < tol -> f_p = 0.
  4. Component split (component_precip_frac_specify, precip_frac_calc_type=2 — the fixed
     default): upsilon = mixt_frac f_p(1)/f_p is the specified ratio (clubb_params
     iupsilon_precip_frac_rat). Only upsilon==1 and the general 0<upsilon<1 branches are
     reachable (the upsilon==0 branch is `0 < 0` dead code).
  5. max_hm_ip_comp_mean limiter: boost precip_frac_i so the in-precip component mean
     hm_i/precip_frac_i <= 0.0025 kg/kg for mixing-ratio hydrometeors.
  6. f_p = mixt_frac f_p(1) + (1-mixt_frac) f_p(2); clamp to [precip_frac_tol, 1] where
     hydromet is present.

Ascending-grid SCM assumption (k increases upward): "at or above" = higher index, so step 2
is a reverse cumulative max (jax.lax.cummax(..., reverse=True)). Verified bit-to-bit against
the f2py API (f2py_precip_fraction); see tests/test_precip_fraction.py.

All operations are jnp / lax (differentiable where the max/where branches are smooth).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from clubb_jax.src.CLUBB_core.constants_clubb import cloud_frac_min

jax.config.update("jax_enable_x64", True)

# precipitation_fraction.F90 parameter (cloud_frac_min imported from constants_clubb above)
_PRECIP_FRAC_TOL_COEF = 0.1
_MAX_HM_IP_COMP_MEAN = 0.0025
_EPS = jnp.finfo(jnp.float64).eps
_CONST_EPS = max(1.0e-10, np.finfo(np.float64).eps)   # constants_clubb.F90:eps


def precip_frac_assert_check(hydromet, hydromet_tol, mixt_frac, precip_frac,
                             precip_frac_1, precip_frac_2, precip_frac_tol):
    """Validity checks for the precipitation-fraction outputs (precipitation_fraction.F90:precip_frac_assert_check).

    At each level: where any hydrometeor >= its tol, precip_frac in [precip_frac_tol, 1] and precip_frac_1/2 in
    [0, 1] but not in (0, precip_frac_tol-eps); where all hydrometeors < tol, precip_frac/_1/_2 are all ~0 (|.|<=eps).
    Always: precip_frac == mixt_frac*precip_frac_1 + (1-mixt_frac)*precip_frac_2 within eps (eps=constants_clubb.eps,
    ~1e-10). The Fortran sets err_code=fatal on any violation; the JAX path never error-stops, so this returns True
    iff every check passes. Concrete-numpy check (not in any gradient path). hydromet is (..., hydromet_dim) per
    level / (nzt, hydromet_dim); hydromet_tol is (hydromet_dim,); the precip_frac arrays and mixt_frac are (nzt,);
    precip_frac_tol is the per-column scalar."""
    hm = np.asarray(hydromet, dtype=np.float64)
    pf = np.asarray(precip_frac, dtype=np.float64)
    pf1 = np.asarray(precip_frac_1, dtype=np.float64)
    pf2 = np.asarray(precip_frac_2, dtype=np.float64)
    mf = np.asarray(mixt_frac, dtype=np.float64)
    tol = np.asarray(hydromet_tol, dtype=np.float64)
    pft = float(precip_frac_tol)
    eps = _CONST_EPS

    has_hm = np.any(hm >= tol, axis=-1)          # (nzt,)
    # Always: overall == mixt_frac-weighted component sum, within eps.
    consistent = np.abs(pf - (mf * pf1 + (1.0 - mf) * pf2)) <= eps
    # Cloudy levels: in-range overall + components (and not sub-tol-but-positive).
    cloudy_ok = ((pf >= pft) & (pf <= 1.0)
                 & ~((pf1 > 0.0) & (pf1 < pft - eps)) & (pf1 >= 0.0) & (pf1 <= 1.0)
                 & ~((pf2 > 0.0) & (pf2 < pft - eps)) & (pf2 >= 0.0) & (pf2 <= 1.0))
    # Clear levels: everything ~0.
    clear_ok = (np.abs(pf) <= eps) & (np.abs(pf1) <= eps) & (np.abs(pf2) <= eps)
    return bool(np.all(consistent) and np.all(np.where(has_hm, cloudy_ok, clear_ok)))


def _specify_general(pf, mf, tol, upsilon):
    """component_precip_frac_specify, the 0<upsilon<1 branch. F90:979-1074. Element-wise.

    pf, mf are (ngrdcol,nzt); tol is broadcast (ngrdcol,1). Returns (pf1, pf2)."""
    omf = 1.0 - mf

    pf1 = upsilon * pf / mf
    pf1 = jnp.where(pf1 > 1.0, 1.0, jnp.where(pf1 < tol, tol, pf1))
    pf2 = (pf - mf * pf1) / omf

    # Shared "recompute pf1 from pf2, then double-check pf1" used in both pf2 sub-cases.
    def _recompute_from_pf2(pf2_set):
        p1 = (pf - omf * pf2_set) / mf
        # if p1>1: p1=1, p2=(pf-mf)/omf ; elif p1<tol: p1=tol, p2=p1*(((pf/p1)-mf)/omf)
        p1_lt = jnp.maximum(tol, 0.0)            # tol value for the <tol case
        p2_when_gt1 = (pf - mf) / omf
        p2_when_lttol = p1_lt * (((pf / p1_lt) - mf) / omf)
        p1_out = jnp.where(p1 > 1.0, 1.0, jnp.where(p1 < tol, p1_lt, p1))
        p2_out = jnp.where(p1 > 1.0, p2_when_gt1,
                           jnp.where(p1 < tol, p2_when_lttol, pf2_set))
        return p1_out, p2_out

    # Case pf2 > 1: set pf2=1, pf1=(pf-(1-mf))/mf, then double-check pf1.
    pf1_gt1 = (pf - omf) / mf
    p2_gt1_gt1 = (pf - mf) / omf
    p2_gt1_lttol = tol * (((pf / tol) - mf) / omf)
    pf1_after_gt1 = jnp.where(pf1_gt1 > 1.0, 1.0, jnp.where(pf1_gt1 < tol, tol, pf1_gt1))
    pf2_after_gt1 = jnp.where(pf1_gt1 > 1.0, p2_gt1_gt1,
                              jnp.where(pf1_gt1 < tol, p2_gt1_lttol, 1.0))

    # Case pf2 < tol: set pf2=tol, recompute pf1, double-check.
    pf1_lttol, pf2_lttol = _recompute_from_pf2(tol)

    pf1_out = jnp.where(pf2 > 1.0, pf1_after_gt1,
                        jnp.where(pf2 < tol, pf1_lttol, pf1))
    pf2_out = jnp.where(pf2 > 1.0, pf2_after_gt1,
                        jnp.where(pf2 < tol, pf2_lttol, pf2))
    return pf1_out, pf2_out


def _specify_upsilon_one(pf, mf, tol):
    """component_precip_frac_specify, the upsilon==1 branch. F90:847-899. Element-wise."""
    omf = 1.0 - mf
    # precip_frac <= mixt_frac: all precip in comp 1.
    pf1_a = pf / mf
    pf2_a = jnp.zeros_like(pf)
    # precip_frac > mixt_frac: pf1=1, pf2=(pf-mf)/omf with sub-cases.
    pf2_b = (pf - mf) / omf
    # pf2_b > 1 (and pf~1): pf2=1 ; elif pf2_b<tol: pf2=tol, recompute pf1, double-check.
    near_one = jnp.abs(pf - 1.0) < jnp.abs(pf + 1.0) / 2.0 * _EPS
    # pf2<tol recompute
    p1_r = (pf - omf * tol) / mf
    p2_r_gt1 = (pf - mf) / omf
    p2_r_lttol = tol * (((pf / tol) - mf) / omf)
    p1_r_out = jnp.where(p1_r > 1.0, 1.0, jnp.where(p1_r < tol, tol, p1_r))
    p2_r_out = jnp.where(p1_r > 1.0, p2_r_gt1, jnp.where(p1_r < tol, p2_r_lttol, tol))
    pf1_b = jnp.where((pf2_b > 1.0) & near_one, 1.0,
                      jnp.where(pf2_b < tol, p1_r_out, 1.0))
    pf2_b_out = jnp.where((pf2_b > 1.0) & near_one, 1.0,
                          jnp.where(pf2_b < tol, p2_r_out, pf2_b))
    le = pf <= mf
    return jnp.where(le, pf1_a, pf1_b), jnp.where(le, pf2_a, pf2_b_out)


def component_precip_frac_specify(precip_frac, mixt_frac, tol, upsilon_precip_frac_rat):
    """Per-component precip fraction via the specified-upsilon method (precip_frac_calc_type=2, the fixed default).
    precipitation_fraction.F90:component_precip_frac_specify. upsilon = mixt_frac*f_p(1)/f_p is specified
    (clubb_params iupsilon_precip_frac_rat); only the upsilon==1 and general 0<upsilon<1 branches are reachable
    (the upsilon==0 branch is dead code). Returns (precip_frac_1, precip_frac_2). The Fortran's max_hm_ip_comp_mean
    limiter is applied by the caller (precip_fraction), after this split, mirroring the Fortran boundary."""
    is_one = jnp.abs(upsilon_precip_frac_rat - 1.0) < jnp.abs(upsilon_precip_frac_rat + 1.0) / 2.0 * _EPS
    pf1_g, pf2_g = _specify_general(precip_frac, mixt_frac, tol, upsilon_precip_frac_rat)
    pf1_1, pf2_1 = _specify_upsilon_one(precip_frac, mixt_frac, tol)
    pf1 = jnp.where(is_one, pf1_1, pf1_g)
    pf2 = jnp.where(is_one, pf2_1, pf2_g)
    return pf1, pf2


def precip_fraction(hydromet, cloud_frac, cloud_frac_1, cloud_frac_2,
                    ice_supersat_frac, ice_supersat_frac_1, ice_supersat_frac_2,
                    mixt_frac, l_mix_rat_hm, l_frozen_hm, hydromet_tol,
                    upsilon_precip_frac_rat):
    """Overall + per-component precipitation fraction. precipitation_fraction.F90:26.

    hydromet: (ngrdcol, nzt, hydromet_dim). cloud/ice/mixt: (ngrdcol, nzt).
    l_mix_rat_hm, l_frozen_hm, hydromet_tol: (hydromet_dim,). Ascending grid.
    Returns (precip_frac, precip_frac_1, precip_frac_2, precip_frac_tol[ngrdcol])."""
    hydromet = jnp.asarray(hydromet, dtype=jnp.float64)
    cloud_frac = jnp.asarray(cloud_frac, dtype=jnp.float64)
    mixt_frac = jnp.asarray(mixt_frac, dtype=jnp.float64)
    isf = jnp.asarray(ice_supersat_frac, dtype=jnp.float64)
    l_frozen = jnp.asarray(l_frozen_hm, dtype=bool)
    l_mix = jnp.asarray(l_mix_rat_hm, dtype=bool)
    hm_tol = jnp.asarray(hydromet_tol, dtype=jnp.float64)
    any_frozen = bool(jnp.any(l_frozen))

    # 1. precip_frac_tol (per column).
    cf_max = jnp.max(cloud_frac, axis=1)
    if any_frozen:
        cf_max = jnp.maximum(cf_max, jnp.max(isf, axis=1))
    precip_frac_tol = jnp.maximum(_PRECIP_FRAC_TOL_COEF * cf_max, cloud_frac_min)  # (ngrdcol,)
    tol = precip_frac_tol[:, None]

    # 2. Overall precip_frac: greatest cloud_frac (or max with ice) AT OR ABOVE a level.
    base = jnp.maximum(cloud_frac, isf) if any_frozen else cloud_frac
    precip_frac = lax.cummax(base, axis=1, reverse=True)  # suffix max (ascending grid)

    # 3. Special checks (element-wise).
    has_hm = jnp.any(hydromet >= hm_tol, axis=2)          # (ngrdcol, nzt)
    precip_frac = jnp.where(~has_hm, 0.0,
                            jnp.where(precip_frac < tol, tol, precip_frac))

    # 4. Component split (component_precip_frac_specify; upsilon==1 or general).
    pf1, pf2 = component_precip_frac_specify(precip_frac, mixt_frac, tol, upsilon_precip_frac_rat)
    # No hydrometeors at the level -> both component fractions 0.
    pf1 = jnp.where(has_hm, pf1, 0.0)
    pf2 = jnp.where(has_hm, pf2, 0.0)

    # 5. max_hm_ip_comp_mean limiter (mixing-ratio hydrometeors), sequential over species.
    omf = 1.0 - mixt_frac
    n_hm = hydromet.shape[2]
    for ivar in range(n_hm):
        if not bool(l_mix[ivar]):
            continue
        hm = hydromet[:, :, ivar]
        present = hm >= hm_tol[ivar]
        c1 = present & (hm > mixt_frac * pf1 * _MAX_HM_IP_COMP_MEAN)
        pf1_new = jnp.maximum(jnp.minimum(hm / (mixt_frac * _MAX_HM_IP_COMP_MEAN), 1.0), tol)
        pf1 = jnp.where(c1, pf1_new, pf1)
        c2 = present & (hm > omf * pf2 * _MAX_HM_IP_COMP_MEAN)
        pf2_new = jnp.maximum(jnp.minimum(hm / (omf * _MAX_HM_IP_COMP_MEAN), 1.0), tol)
        pf2 = jnp.where(c2, pf2_new, pf2)

    # 6. Recompute overall and clamp where hydromet present.
    precip_frac = mixt_frac * pf1 + omf * pf2
    precip_frac = jnp.where(has_hm, jnp.minimum(jnp.maximum(precip_frac, tol), 1.0), precip_frac)

    return precip_frac, pf1, pf2, precip_frac_tol


def precip_frac_double_delta_jax(cloud_frac):
    """Top-down fill: precip_frac = cloud_frac where cloudy, else inherit the value from the level above; the
    top level is 0 (ascending grid: index 0 = bottom, nzt-1 = top). The double-delta-PDF precip-fraction
    used by the Morrison rain-evap update_xp2_mc — mirrors the "greatest cloud fraction at or above a grid
    level" form of precipitation_fraction.F90:precip_fraction. (Relocated here from advance_xp2_xpyp_module.py
    to its precip-fraction Fortran home, mirror-refactor iter 240.)
    """
    cf_rev = cloud_frac[:, ::-1]          # index 0 = top
    ng = cloud_frac.shape[0]

    def step(carry, cf_lev):
        pf = jnp.where(cf_lev > cloud_frac_min, cf_lev, carry)
        return pf, pf

    _, rest = lax.scan(step, jnp.zeros(ng), cf_rev[:, 1:].T)   # levels 1..nzt-1 (top-first)
    pf_rev = jnp.concatenate([jnp.zeros((1, ng)), rest], axis=0)   # (nzt, ng), top-first
    return pf_rev.T[:, ::-1]              # (ng, nzt), bottom-first
