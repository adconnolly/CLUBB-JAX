"""JAX port of pdf_closure_module.F90 — PDF moment-integral closures.

Mirrors clubb_release/src/CLUBB_core/pdf_closure_module.F90. The closure orchestration lives
here as `pdf_closure_driver` (the ADG1/zt-specialized driver) + `calc_pdf_higher_order_moments_jax`
+ `adg1_pdf_driver_zt_jax`; advance_clubb_core_module.py *calls* `pdf_closure_driver` rather than
inlining it. Alongside the driver this module holds the standalone analytic moment-integral
routines that close the higher-order moments from the two-component PDF parameters, plus the
cloudy-updraft diagnostic:

  calc_wp2xp_pdf / calc_wpxp2_pdf / calc_wp2xp2_pdf / calc_wp4_pdf / calc_wpxpyp_pdf
      — <w'^2 x'>, <w'x'^2>, <w'^2 x'^2>, <w'^4>, <w'x'y'> from the binormal/trinormal
        PDF integrals (the bare Fortran-named forms; both f2py-validated
        (tests/test_pdf_moment_integrals.py) AND called by the live calc_pdf_higher_order_moments_jax)
  calc_w_up_in_cloud — mean cloudy updraft/downdraft vertical velocity (aerosol activation)

All pure-jnp → differentiable.

References:
  src/CLUBB_core/pdf_closure_module.F90, calc_{wp2xp,wpxp2,wp2xp2,wp4,wpxpyp}_pdf,
  calc_w_up_in_cloud.
"""

import jax
import jax.numpy as jnp
import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt, _asarray
from clubb_jax.src.CLUBB_core.constants_clubb import (
    eps, ep, ep1, ep2, Lv, Rd, Cp, chi_tol, max_num_stdevs, max_mag_correlation,
    min_max_smth_mag, T_freeze_K, sqrt_2, sqrt_2pi, rt_tol, thl_tol, w_tol_sqd,
    ibeta, w_tol, zero_threshold, iiPDF_ADG1, ipdf_post_advance_fields,
)
from clubb_jax.src.CLUBB_core.advance_helper_module import smooth_max
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_ice, sat_mixrat_liq
from clubb_jax.src.CLUBB_core.adg1_adg2_3d_luhar_pdf import (
    calc_comp_corrs_binormal, ADG1_pdf_driver,
)
from clubb_jax.src.CLUBB_core.Skx_module import Skx_func, compute_gamma_Skw
from clubb_jax.src.CLUBB_core.sigma_sqd_w_module import compute_sigma_sqd_w
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax, zm2zt_jax


def calc_wp2xp2_pdf(wm, xm, w_1, w_2, x_1, x_2,
                         varnce_w_1, varnce_w_2,
                         varnce_x_1, varnce_x_2,
                         corr_w_x_1, corr_w_x_2,
                         mixt_frac):
    """pdf_closure_module.F90:calc_wp2xp2_pdf — <w'^2 x'^2> from bivariate PDF integral.

    Formula:
      wp2xp2 = mf * ((dw_1^2 * (dx_1^2 + varnce_x_1)
                      + 4*corr_w_x_1*sqrt(varnce_w_1*varnce_x_1)*dx_1*dw_1
                      + (dx_1^2 + (1+2*corr_w_x_1^2)*varnce_x_1)*varnce_w_1))
             + (1-mf) * (same for component 2)
    """
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm
    dw_2 = w_2 - wm
    dx_1 = x_1 - xm
    dx_2 = x_2 - xm

    term1 = (dw_1 ** 2 * (dx_1 ** 2 + varnce_x_1)
             + 4.0 * corr_w_x_1 * _safe_sqrt(varnce_w_1 * varnce_x_1) * dx_1 * dw_1
             + (dx_1 ** 2 + (1.0 + 2.0 * corr_w_x_1 ** 2) * varnce_x_1) * varnce_w_1)
    term2 = (dw_2 ** 2 * (dx_2 ** 2 + varnce_x_2)
             + 4.0 * corr_w_x_2 * _safe_sqrt(varnce_w_2 * varnce_x_2) * dx_2 * dw_2
             + (dx_2 ** 2 + (1.0 + 2.0 * corr_w_x_2 ** 2) * varnce_x_2) * varnce_w_2)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_w_up_in_cloud(mixt_frac, cloud_frac_1, cloud_frac_2,
                       w_1, w_2, varnce_w_1, varnce_w_2):
    """Mean cloudy updraft / downdraft vertical velocity from the binormal w-PDF
    (pdf_closure_module.F90:calc_w_up_in_cloud). For aerosol activation, this gives a w representative of
    cloudy updrafts (an alternative to sqrt(wp2)). Per PDF component, the truncated-Gaussian updraft integral is
      w_up = 1/2 w (1+erf(r)) + sigma/sqrt(2pi) exp(-r^2),  r = w/(sqrt(2) sigma),  updraft_frac = 1/2(1+erf(r)),
    with all-updraft / all-downdraft shortcuts when |w| > max_num_stdevs*sigma. The cloudy means weight the two
    components by mixt_frac*cloud_frac. Returns
    (w_up_in_cloud, w_down_in_cloud, cloudy_updraft_frac, cloudy_downdraft_frac). Pure-jnp → differentiable.

    All inputs are (ngrdcol, nz). varnce_w_* are variances (sigma^2)."""
    import jax.scipy.special as jsp
    from clubb_jax.src.CLUBB_core.constants_clubb import sqrt_2, sqrt_2pi, max_num_stdevs

    def _component(w, varnce):
        w = jnp.asarray(w, dtype=jnp.float64)
        stdev = jnp.sqrt(jnp.asarray(varnce, dtype=jnp.float64))
        all_up = w > max_num_stdevs * stdev
        all_down = w < -max_num_stdevs * stdev
        ratio = w / (sqrt_2 * jnp.maximum(eps, stdev))
        erf_r = jsp.erf(ratio)
        exp_neg = jnp.exp(-ratio ** 2)
        w_up_mid = 0.5 * w * (1.0 + erf_r) + (stdev / sqrt_2pi) * exp_neg
        uf_mid = 0.5 * (1.0 + erf_r)
        w_down_mid = 0.5 * w * (1.0 - erf_r) - (stdev / sqrt_2pi) * exp_neg
        w_up = jnp.where(all_up, w, jnp.where(all_down, 0.0, w_up_mid))
        uf = jnp.where(all_up, 1.0, jnp.where(all_down, 0.0, uf_mid))
        w_down = jnp.where(all_up, 0.0, jnp.where(all_down, w, w_down_mid))
        df = 1.0 - uf   # holds in all three branches (Fortran: 1, 0, 1-uf_mid)
        return w_up, uf, w_down, df

    a = jnp.asarray(mixt_frac, dtype=jnp.float64)
    cf1 = jnp.asarray(cloud_frac_1, dtype=jnp.float64); cf2 = jnp.asarray(cloud_frac_2, dtype=jnp.float64)
    w_up_1, uf_1, w_down_1, df_1 = _component(w_1, varnce_w_1)
    w_up_2, uf_2, w_down_2, df_2 = _component(w_2, varnce_w_2)

    cloudy_updraft_frac = a * cf1 * uf_1 + (1.0 - a) * cf2 * uf_2
    cloudy_downdraft_frac = a * cf1 * df_1 + (1.0 - a) * cf2 * df_2
    w_up_in_cloud = ((a * cf1 * w_up_1 + (1.0 - a) * cf2 * w_up_2)
                     / jnp.maximum(eps, cloudy_updraft_frac))
    w_down_in_cloud = ((a * cf1 * w_down_1 + (1.0 - a) * cf2 * w_down_2)
                       / jnp.maximum(eps, cloudy_downdraft_frac))
    return w_up_in_cloud, w_down_in_cloud, cloudy_updraft_frac, cloudy_downdraft_frac


def calc_wp4_pdf(wm, w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac):
    """<w'^4> integrated over the two-component-normal PDF of w (pdf_closure_module.F90:calc_wp4_pdf):
      <w'^4> = Σ_i weight_i (3 σ_w_i^4 + 6 (μ_w_i-<w>)^2 σ_w_i^2 + (μ_w_i-<w>)^4). Pure-jnp → differentiable."""
    wm = jnp.asarray(wm); a = jnp.asarray(mixt_frac)
    d1 = jnp.asarray(w_1) - wm; d2 = jnp.asarray(w_2) - wm
    v1 = jnp.asarray(varnce_w_1); v2 = jnp.asarray(varnce_w_2)
    return (a * (3.0 * v1 ** 2 + 6.0 * d1 ** 2 * v1 + d1 ** 4)
            + (1.0 - a) * (3.0 * v2 ** 2 + 6.0 * d2 ** 2 * v2 + d2 ** 4))


def calc_wp2xp_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """<w'^2 x'> integrated over the binormal PDF of (w, x) (pdf_closure_module.F90:calc_wp2xp_pdf):
      Σ_i weight_i [ ((μ_w_i-<w>)^2 + σ_w_i^2)(μ_x_i-<x>) + 2 corr_i σ_w_i σ_x_i (μ_w_i-<w>) ]."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    c1 = jnp.asarray(corr_w_x_1); c2 = jnp.asarray(corr_w_x_2)
    return (a * ((dw1 ** 2 + vw1) * dx1 + 2.0 * c1 * _safe_sqrt(vw1 * vx1) * dw1)
            + (1.0 - a) * ((dw2 ** 2 + vw2) * dx2 + 2.0 * c2 * _safe_sqrt(vw2 * vx2) * dw2))


def calc_wpxp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """<w'x'^2> integrated over the binormal PDF of (w, x) (pdf_closure_module.F90:calc_wpxp2_pdf):
      Σ_i weight_i [ (μ_w_i-<w>)((μ_x_i-<x>)^2 + σ_x_i^2) + 2 corr_i σ_w_i σ_x_i (μ_x_i-<x>) ]."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    c1 = jnp.asarray(corr_w_x_1); c2 = jnp.asarray(corr_w_x_2)
    return (a * (dw1 * (dx1 ** 2 + vx1) + 2.0 * c1 * _safe_sqrt(vw1 * vx1) * dx1)
            + (1.0 - a) * (dw2 * (dx2 ** 2 + vx2) + 2.0 * c2 * _safe_sqrt(vw2 * vx2) * dx2))


def calc_wpxpyp_pdf(wm, xm, ym, w_1, w_2, x_1, x_2, y_1, y_2,
                    varnce_w_1, varnce_w_2, varnce_x_1, varnce_x_2, varnce_y_1, varnce_y_2,
                    corr_w_x_1, corr_w_x_2, corr_w_y_1, corr_w_y_2, corr_x_y_1, corr_x_y_2, mixt_frac):
    """<w'x'y'> integrated over the trinormal PDF of (w, x, y) (pdf_closure_module.F90:calc_wpxpyp_pdf):
      Σ_i weight_i [ (μ_w-<w>)(μ_x-<x>)(μ_y-<y>) + corr_xy σ_x σ_y (μ_w-<w>)
                     + corr_wy σ_w σ_y (μ_x-<x>) + corr_wx σ_w σ_x (μ_y-<y>) ]_i."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); ym = jnp.asarray(ym); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    dy1 = jnp.asarray(y_1) - ym; dy2 = jnp.asarray(y_2) - ym
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    vy1 = jnp.asarray(varnce_y_1); vy2 = jnp.asarray(varnce_y_2)
    cwx1 = jnp.asarray(corr_w_x_1); cwx2 = jnp.asarray(corr_w_x_2)
    cwy1 = jnp.asarray(corr_w_y_1); cwy2 = jnp.asarray(corr_w_y_2)
    cxy1 = jnp.asarray(corr_x_y_1); cxy2 = jnp.asarray(corr_x_y_2)
    comp1 = (dw1 * dx1 * dy1 + cxy1 * _safe_sqrt(vx1 * vy1) * dw1
             + cwy1 * _safe_sqrt(vw1 * vy1) * dx1 + cwx1 * _safe_sqrt(vw1 * vx1) * dy1)
    comp2 = (dw2 * dx2 * dy2 + cxy2 * _safe_sqrt(vx2 * vy2) * dw2
             + cwy2 * _safe_sqrt(vw2 * vy2) * dx2 + cwx2 * _safe_sqrt(vw2 * vx2) * dy2)
    return a * comp1 + (1.0 - a) * comp2


def transform_pdf_chi_eta_component(tl, rsatl, rt, exner_in,
                                        varnce_rt, varnce_thl, corr_rt_thl):
    """pdf_closure_module.F90:transform_pdf_chi_eta_component (line 1699).

    Sommeria & Deardorff (1977) extended-liquid-water-temperature transform of a single
    PDF component from (rt, thl) to the (chi, eta) coordinate that diagnoses liquid water.
    Returns the chi/eta means, standard deviations, the rt/thl→chi sensitivity coefficients
    (crt, cthl) and the chi-eta correlation for one PDF component.

    Returns the Fortran out-arg order:
        (chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta, corr_chi_eta)
    """
    beta       = ep * Lv**2 / (Rd * Cp * tl**2)
    invrs      = 1.0 / (1.0 + beta * rsatl)
    chi        = (rt - rsatl) * invrs
    crt        = invrs
    cthl       = ((1.0 + beta * rt) * invrs**2
                  * (Cp / Lv) * beta * rsatl * exner_in)
    vrnc_rt_t  = crt**2 * varnce_rt
    vrnc_thl_t = cthl**2 * varnce_thl
    corr_t     = (2.0 * corr_rt_thl * crt * cthl
                  * jnp.sqrt(varnce_rt * varnce_thl))
    vrnc_chi   = vrnc_rt_t - corr_t + vrnc_thl_t
    vrnc_eta   = vrnc_rt_t + corr_t + vrnc_thl_t
    stdev_chi  = _safe_sqrt(vrnc_chi)
    stdev_eta  = _safe_sqrt(vrnc_eta)
    covar_chi_eta = vrnc_rt_t - vrnc_thl_t
    # smooth_corr_quotient (pdf_utilities.F90:1360)
    _denom_thresh = chi_tol**2
    _smth = min(min_max_smth_mag, _denom_thresh)
    denom = stdev_chi * stdev_eta
    tmp_d = smooth_max(jnp.abs(covar_chi_eta) / max_mag_correlation, denom, _smth)
    tmp_d = smooth_max(tmp_d, _denom_thresh, _smth)
    corr_chi_eta = covar_chi_eta / tmp_d
    return chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta, corr_chi_eta


def calc_liquid_cloud_frac_component(mean_chi, stdev_chi_in):
    """pdf_closure_module.F90:calc_liquid_cloud_frac_component (lines 2453-2479).

    Liquid cloud fraction and liquid water mixing ratio of one PDF component from the
    Gaussian CDF of chi (extended liquid water), with ±max_num_stdevs truncation to the
    clear / fully-cloudy limits.  Returns (cloud_frac, rc).
    """
    is_clear = (
        ((jnp.abs(mean_chi) <= eps) & (stdev_chi_in <= chi_tol))
        | (mean_chi < -max_num_stdevs * stdev_chi_in)
    )
    is_full  = mean_chi > max_num_stdevs * stdev_chi_in
    safe_s   = jnp.maximum(stdev_chi_in, 1.0e-100)
    zeta     = mean_chi / safe_s
    cf_mid   = 0.5 * (1.0 + jax.scipy.special.erf(zeta / sqrt_2))
    rc_mid   = (mean_chi * cf_mid
                + stdev_chi_in * jnp.exp(-0.5 * zeta**2) / sqrt_2pi)
    cf = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, cf_mid))
    rc = jnp.where(is_clear, 0.0, jnp.where(is_full, mean_chi, rc_mid))
    return cf, rc


def calc_pdf_liquid_cloud_frac_components_jax(*, adg1, exner, p_in_Pa,
                                              corr_rt_thl_1, corr_rt_thl_2, saturation_formula):
    """Per-component liquid cloud-fraction PDF closure (pdf_closure_module.F90). For each ADG1 component
    it forms the chi-eta transform (transform_pdf_chi_eta_component, F90:1699) and the liquid cloud fraction
    (calc_liquid_cloud_frac_component, F90:2453), then combines the two by mixture fraction (F90:1020-1024).

    The rt-thl component correlation is passed in (the pre-advance caller derives it via the binormal closure;
    the post-advance caller derives it from post-advance state). Returns a dict of every per-component
    intermediate (chi/crt/cthl/stdev_chi/stdev_eta/covar_ce/corr_ce, cf/rc, tl/rsatl, plus the ADG1
    means/variances and exner/p) and the combined rcm/cloud_frac — the post-advance path reuses all of these
    for its ice-supersat, xprcp, pdf_params and stats steps.
    """
    _mf    = adg1['mixt_frac']
    _rt1   = adg1['rt_1'];          _rt2   = adg1['rt_2']
    _thl1  = adg1['thl_1'];         _thl2  = adg1['thl_2']
    _vrt1  = adg1['varnce_rt_1'];   _vrt2  = adg1['varnce_rt_2']
    _vthl1 = adg1['varnce_thl_1'];  _vthl2 = adg1['varnce_thl_2']

    _exner_j = jnp.asarray(exner)
    _p_j     = jnp.asarray(p_in_Pa)
    _tl1     = _thl1 * _exner_j
    _tl2     = _thl2 * _exner_j
    _rsatl1  = sat_mixrat_liq(_p_j, _tl1, saturation_formula)
    _rsatl2  = sat_mixrat_liq(_p_j, _tl2, saturation_formula)

    (_chi1, _crt1, _cthl1, _schi1, _seta1, _covar_ce1, _corr_ce1) = \
        transform_pdf_chi_eta_component(
            _tl1, _rsatl1, _rt1, _exner_j, _vrt1, _vthl1, corr_rt_thl_1)
    (_chi2, _crt2, _cthl2, _schi2, _seta2, _covar_ce2, _corr_ce2) = \
        transform_pdf_chi_eta_component(
            _tl2, _rsatl2, _rt2, _exner_j, _vrt2, _vthl2, corr_rt_thl_2)

    _cf1, _rc1 = calc_liquid_cloud_frac_component(_chi1, _schi1)
    _cf2, _rc2 = calc_liquid_cloud_frac_component(_chi2, _schi2)

    cloud_frac = _mf * _cf1 + (1.0 - _mf) * _cf2
    rcm        = jnp.maximum(0.0, _mf * _rc1 + (1.0 - _mf) * _rc2)
    return {
        'mixt_frac': _mf, 'exner': _exner_j, 'p': _p_j,
        'rt_1': _rt1, 'rt_2': _rt2, 'thl_1': _thl1, 'thl_2': _thl2,
        'varnce_rt_1': _vrt1, 'varnce_rt_2': _vrt2,
        'varnce_thl_1': _vthl1, 'varnce_thl_2': _vthl2,
        'tl_1': _tl1, 'tl_2': _tl2, 'rsatl_1': _rsatl1, 'rsatl_2': _rsatl2,
        'chi_1': _chi1, 'chi_2': _chi2, 'crt_1': _crt1, 'crt_2': _crt2,
        'cthl_1': _cthl1, 'cthl_2': _cthl2,
        'stdev_chi_1': _schi1, 'stdev_chi_2': _schi2,
        'stdev_eta_1': _seta1, 'stdev_eta_2': _seta2,
        'covar_ce_1': _covar_ce1, 'covar_ce_2': _covar_ce2,
        'corr_ce_1': _corr_ce1, 'corr_ce_2': _corr_ce2,
        'cf_1': _cf1, 'cf_2': _cf2, 'rc_1': _rc1, 'rc_2': _rc2,
        'cloud_frac': cloud_frac, 'rcm': rcm,
    }


def calc_pdf_liquid_cloud_frac_jax(*, adg1, rtpthlp_zt, rtm, thlm, exner, p_in_Pa,
                                   saturation_formula):
    """Liquid cloud fraction and cloud water from the ADG1 PDF components (the cloud-fraction computation
    of pdf_closure) for the pre-advance path: derive the binormal rt-thl component correlation, then defer
    to calc_pdf_liquid_cloud_frac_components_jax. Returns (rcm, cloud_frac).
    """
    _corr_rtthl_1, _corr_rtthl_2 = calc_comp_corrs_binormal(
        jnp.asarray(rtpthlp_zt), jnp.asarray(rtm), jnp.asarray(thlm),
        adg1['rt_1'], adg1['rt_2'], adg1['thl_1'], adg1['thl_2'],
        adg1['varnce_rt_1'], adg1['varnce_rt_2'], adg1['varnce_thl_1'], adg1['varnce_thl_2'],
        adg1['mixt_frac'],
    )
    _comp = calc_pdf_liquid_cloud_frac_components_jax(
        adg1=adg1, exner=exner, p_in_Pa=p_in_Pa,
        corr_rt_thl_1=_corr_rtthl_1, corr_rt_thl_2=_corr_rtthl_2,
        saturation_formula=saturation_formula)
    return _comp['rcm'], _comp['cloud_frac']


def calc_xprcp_component(wm, rtm, thlm, um, vm, rcm,
                             w_i, rt_i, thl_i, u_i, v_i,
                             varnce_w_i, chi_i, stdev_chi_i, stdev_eta_i,
                             corr_w_chi_i, corr_chi_eta_i, crt_i, cthl_i,
                             rc_i, cloud_frac_i):
    """pdf_closure_module.F90:calc_xprcp_component (line 2652).

    Per-PDF-component contributions to the cloud-water covariances <w'rc'>, <w'^2 rc'>,
    <rt'rc'>, <thl'rc'>, <u'rc'>, <v'rc'> for one PDF component, on the zt grid.  Mirrors
    the ADG1 path (F90:3089-3104); the non-ADG1 corr_w_chi correction (F90:3110-3138, run
    only for iiPDF_type ∉ {ADG1, ADG2, new_hybrid}) is omitted — the gated config is ADG1,
    where corr_w_chi = 0, so `chi_i`/`corr_w_chi_i` are accepted for signature fidelity only.

    Returns: (wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp)
    """
    drc = rc_i - rcm
    wprcp  = (w_i - wm) * drc
    wp2rcp = ((w_i - wm) ** 2 + varnce_w_i) * drc
    rtprcp = ((rt_i - rtm) * drc
              + (corr_chi_eta_i * stdev_eta_i + stdev_chi_i)
                / (2.0 * crt_i) * stdev_chi_i * cloud_frac_i)
    # Guard against cthl=0 (rsatl=0 limit); cloud_frac=0 masks the result there.
    cthl_safe = jnp.where(cthl_i == 0.0, 1.0, cthl_i)
    thlprcp = ((thl_i - thlm) * drc
               + (corr_chi_eta_i * stdev_eta_i - stdev_chi_i)
                 / (2.0 * cthl_safe) * stdev_chi_i * cloud_frac_i)
    uprcp = (u_i - um) * drc
    vprcp = (v_i - vm) * drc
    return wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp


def calc_pdf_xprcp_fluxes_jax(*, adg1, comp, wm_zt, rtm, thlm, um, vm, rcm_zt, gr):
    """Mixed cloud-water turbulent fluxes from the ADG1 PDF (the x'rc' section of pdf_closure_driver).

    Calls calc_xprcp_component for each PDF component (w'rc', w'^2rc', rt'rc', thl'rc', u'rc', v'rc'), mixes
    them by mixture fraction, and regrids the zm-output fluxes to momentum levels, zeroing the upper boundary
    (pdf_closure_module.F90:4233-4261). corr_w_chi = 0 for ADG1. `comp` is the per-component dict from
    calc_pdf_liquid_cloud_frac_components_jax. Returns a dict of the zt-grid fluxes (consumed by the x'thv'
    assembly, which wants the native pdf-grid values) and the regridded zm-grid fluxes.
    """
    _mf   = comp['mixt_frac']
    _zero = jnp.zeros_like(_mf)
    _wm_zt = jnp.asarray(wm_zt)
    _rtm   = jnp.asarray(rtm)
    _thlm  = jnp.asarray(thlm)
    _um    = jnp.asarray(um)
    _vm    = jnp.asarray(vm)

    (_wprcp_c1, _wp2rcp_c1, _rtprcp_c1, _thlprcp_c1,
     _uprcp_c1, _vprcp_c1) = calc_xprcp_component(
        _wm_zt, _rtm, _thlm, _um, _vm, rcm_zt,
        adg1['w_1'], adg1['rt_1'], adg1['thl_1'], adg1['u_1'], adg1['v_1'],
        adg1['varnce_w_1'], comp['chi_1'], comp['stdev_chi_1'], comp['stdev_eta_1'],
        _zero, comp['corr_ce_1'], comp['crt_1'], comp['cthl_1'], comp['rc_1'], comp['cf_1'])
    (_wprcp_c2, _wp2rcp_c2, _rtprcp_c2, _thlprcp_c2,
     _uprcp_c2, _vprcp_c2) = calc_xprcp_component(
        _wm_zt, _rtm, _thlm, _um, _vm, rcm_zt,
        adg1['w_2'], adg1['rt_2'], adg1['thl_2'], adg1['u_2'], adg1['v_2'],
        adg1['varnce_w_2'], comp['chi_2'], comp['stdev_chi_2'], comp['stdev_eta_2'],
        _zero, comp['corr_ce_2'], comp['crt_2'], comp['cthl_2'], comp['rc_2'], comp['cf_2'])

    # Mix with mixt_frac (on zt grid)
    _wprcp_zt  = _mf * _wprcp_c1  + (1.0 - _mf) * _wprcp_c2
    _wp2rcp_zt = _mf * _wp2rcp_c1 + (1.0 - _mf) * _wp2rcp_c2
    _rtprcp_zt = _mf * _rtprcp_c1 + (1.0 - _mf) * _rtprcp_c2
    _thlprcp_zt= _mf * _thlprcp_c1+ (1.0 - _mf) * _thlprcp_c2
    _uprcp_zt  = _mf * _uprcp_c1  + (1.0 - _mf) * _uprcp_c2
    _vprcp_zt  = _mf * _vprcp_c1  + (1.0 - _mf) * _vprcp_c2

    # Convert zt→zm; zero at k_ub_zm
    _k_ub = gr.k_ub_zm
    _wprcp_zm   = zt2zm_jax(_wprcp_zt,   gr).at[:, _k_ub].set(0.0)
    _rtprcp_zm  = zt2zm_jax(_rtprcp_zt,  gr).at[:, _k_ub].set(0.0)
    _thlprcp_zm = zt2zm_jax(_thlprcp_zt, gr).at[:, _k_ub].set(0.0)
    _uprcp_zm   = zt2zm_jax(_uprcp_zt,   gr).at[:, _k_ub].set(0.0)
    _vprcp_zm   = zt2zm_jax(_vprcp_zt,   gr).at[:, _k_ub].set(0.0)

    return {
        'wprcp_zt': _wprcp_zt, 'wp2rcp_zt': _wp2rcp_zt, 'rtprcp_zt': _rtprcp_zt,
        'thlprcp_zt': _thlprcp_zt, 'uprcp_zt': _uprcp_zt, 'vprcp_zt': _vprcp_zt,
        'wprcp_zm': _wprcp_zm, 'rtprcp_zm': _rtprcp_zm, 'thlprcp_zm': _thlprcp_zm,
        'uprcp_zm': _uprcp_zm, 'vprcp_zm': _vprcp_zm,
    }


def calc_pdf_higher_order_moments_jax(adg1, wm_zt, rtm, thlm, um, vm,
                                      corr_rt_thl_1, corr_rt_thl_2, gr):
    """pdf_closure higher-order-moment section (pdf_closure_module.F90 body).

    Integrates the two-component ADG1 PDF for the higher-order velocity-scalar moments
    via the standalone PDF-integral routines (calc_wp2xp_pdf / calc_wpxp2_pdf /
    calc_wp2xp2_pdf / calc_wp4_pdf / calc_wpxpyp_pdf), reproducing the grid handling of
    the Fortran pdf_closure body + pdf_closure_driver regrid: wp2up2/wp2vp2/wp4 are formed
    on zt then converted to zm with the k_lb/k_ub zeroing (wp4 also uses zm_min=0).

    For ADG1 the velocity-scalar correlations corr_w_rt = corr_w_thl = corr_u_w = corr_v_w
    = 0; only corr_rt_thl (per component) is non-zero (used by wprtpthlp).

    Returns a dict with the Fortran moment names:
        wp2rtp, wp2thlp, wp2up, wpup2, wpvp2, wp2up2_zm, wp2vp2_zm, wp4_zm,
        wprtp2, wpthlp2, wprtpthlp
    """
    mf = adg1['mixt_frac']
    zero = jnp.zeros_like(mf)
    w1, w2 = adg1['w_1'], adg1['w_2']
    vw1, vw2 = adg1['varnce_w_1'], adg1['varnce_w_2']

    def _wp2xp(xm, x_1, x_2, vx_1, vx_2):
        return calc_wp2xp_pdf(wm=wm_zt, xm=xm, w_1=w1, w_2=w2, x_1=x_1, x_2=x_2,
                                  varnce_w_1=vw1, varnce_w_2=vw2, varnce_x_1=vx_1,
                                  varnce_x_2=vx_2, corr_w_x_1=zero, corr_w_x_2=zero,
                                  mixt_frac=mf)

    def _wpxp2(xm, x_1, x_2, vx_1, vx_2):
        return calc_wpxp2_pdf(wm=wm_zt, xm=xm, w_1=w1, w_2=w2, x_1=x_1, x_2=x_2,
                                  varnce_w_1=vw1, varnce_w_2=vw2, varnce_x_1=vx_1,
                                  varnce_x_2=vx_2, corr_w_x_1=zero, corr_w_x_2=zero,
                                  mixt_frac=mf)

    def _wp2xp2(xm, x_1, x_2, vx_1, vx_2):
        return calc_wp2xp2_pdf(wm=wm_zt, xm=xm, w_1=w1, w_2=w2, x_1=x_1, x_2=x_2,
                                   varnce_w_1=vw1, varnce_w_2=vw2, varnce_x_1=vx_1,
                                   varnce_x_2=vx_2, corr_w_x_1=zero, corr_w_x_2=zero,
                                   mixt_frac=mf)

    # <w'^2 x'>: wp2rtp, wp2thlp, wp2up
    wp2rtp  = _wp2xp(rtm,  adg1['rt_1'],  adg1['rt_2'],  adg1['varnce_rt_1'],  adg1['varnce_rt_2'])
    wp2thlp = _wp2xp(thlm, adg1['thl_1'], adg1['thl_2'], adg1['varnce_thl_1'], adg1['varnce_thl_2'])
    wp2up   = _wp2xp(um,   adg1['u_1'],   adg1['u_2'],   adg1['varnce_u_1'],   adg1['varnce_u_2'])

    # <w'x'^2>: wpup2, wpvp2
    wpup2 = _wpxp2(um, adg1['u_1'], adg1['u_2'], adg1['varnce_u_1'], adg1['varnce_u_2'])
    wpvp2 = _wpxp2(vm, adg1['v_1'], adg1['v_2'], adg1['varnce_v_1'], adg1['varnce_v_2'])

    # <w'^2 x'^2>: wp2up2, wp2vp2 — formed on zt, regridded to zm, k_ub zeroed
    wp2up2_zt = _wp2xp2(um, adg1['u_1'], adg1['u_2'], adg1['varnce_u_1'], adg1['varnce_u_2'])
    wp2vp2_zt = _wp2xp2(vm, adg1['v_1'], adg1['v_2'], adg1['varnce_v_1'], adg1['varnce_v_2'])
    k_ub = gr.k_ub_zm
    wp2up2_zm = zt2zm_jax(wp2up2_zt, gr).at[:, k_ub].set(0.0)
    wp2vp2_zm = zt2zm_jax(wp2vp2_zt, gr).at[:, k_ub].set(0.0)

    # <w'^4> — formed on zt, regridded (zm_min=0), k_lb + k_ub zeroed
    wp4_zt = calc_wp4_pdf(wm=wm_zt, w_1=w1, w_2=w2,
                              varnce_w_1=vw1, varnce_w_2=vw2, mixt_frac=mf)
    wp4_zm = zt2zm_jax(wp4_zt, gr, zm_min=0.0)
    wp4_zm = wp4_zm.at[:, gr.k_lb_zm].set(0.0).at[:, k_ub].set(0.0)

    # <w'x'^2>: wprtp2, wpthlp2
    wprtp2  = _wpxp2(rtm,  adg1['rt_1'],  adg1['rt_2'],  adg1['varnce_rt_1'],  adg1['varnce_rt_2'])
    wpthlp2 = _wpxp2(thlm, adg1['thl_1'], adg1['thl_2'], adg1['varnce_thl_1'], adg1['varnce_thl_2'])

    # <w'rt'thl'>: corr_w_rt = corr_w_thl = 0; corr_rt_thl per component
    wprtpthlp = calc_wpxpyp_pdf(
        wm=wm_zt, xm=rtm, ym=thlm, w_1=w1, w_2=w2,
        x_1=adg1['rt_1'], x_2=adg1['rt_2'], y_1=adg1['thl_1'], y_2=adg1['thl_2'],
        varnce_w_1=vw1, varnce_w_2=vw2,
        varnce_x_1=adg1['varnce_rt_1'], varnce_x_2=adg1['varnce_rt_2'],
        varnce_y_1=adg1['varnce_thl_1'], varnce_y_2=adg1['varnce_thl_2'],
        corr_w_x_1=zero, corr_w_x_2=zero, corr_w_y_1=zero, corr_w_y_2=zero,
        corr_x_y_1=corr_rt_thl_1, corr_x_y_2=corr_rt_thl_2, mixt_frac=mf)

    return {
        'wp2rtp': wp2rtp, 'wp2thlp': wp2thlp, 'wp2up': wp2up,
        'wpup2': wpup2, 'wpvp2': wpvp2,
        'wp2up2_zm': wp2up2_zm, 'wp2vp2_zm': wp2vp2_zm, 'wp4_zm': wp4_zm,
        'wprtp2': wprtp2, 'wpthlp2': wpthlp2, 'wprtpthlp': wprtpthlp,
    }


def calc_xpthvp_terms_jax(exner, thv_ds_zt, wprcp_zt, wp2rcp_zt, rtprcp_zt, thlprcp_zt,
                          wpthlp_zt, wprtp_zt, wp2thlp_zt, wp2rtp_zt,
                          rtpthlp_zt, rtp2_zt, thlp2_zt, gr):
    """pdf_closure_module.F90:1122-1158 — the <x'th_v'> buoyancy-flux assembly.

    Builds rc_coef and the four virtual-potential-temperature fluxes on the zt grid:
        rc_coef   = Lv/(exner*Cp) - ep2*thv_ds
        x'thv'    = x'thl' + ep1*thv_ds*x'rt' + rc_coef*x'rc'   (x = w, w^2, rt, thl)
    then converts the three zm-output fluxes (and rc_coef) from zt→zm and zeroes k_ub_zm
    (the pdf_closure_driver post-step regrid, F90:4233-4261).  wp2thvp stays on zt.

    Returns: (wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt, rc_coef_zm)
    """
    rc_coef_zt = Lv / (exner * Cp) - ep2 * thv_ds_zt
    wpthvp_zt   = wpthlp_zt  + ep1 * thv_ds_zt * wprtp_zt   + rc_coef_zt * wprcp_zt
    wp2thvp_zt  = wp2thlp_zt + ep1 * thv_ds_zt * wp2rtp_zt  + rc_coef_zt * wp2rcp_zt
    rtpthvp_zt  = rtpthlp_zt + ep1 * thv_ds_zt * rtp2_zt    + rc_coef_zt * rtprcp_zt
    thlpthvp_zt = thlp2_zt   + ep1 * thv_ds_zt * rtpthlp_zt + rc_coef_zt * thlprcp_zt
    k_ub = gr.k_ub_zm
    wpthvp_zm   = zt2zm_jax(wpthvp_zt,   gr).at[:, k_ub].set(0.0)
    rtpthvp_zm  = zt2zm_jax(rtpthvp_zt,  gr).at[:, k_ub].set(0.0)
    thlpthvp_zm = zt2zm_jax(thlpthvp_zt, gr).at[:, k_ub].set(0.0)
    rc_coef_zm  = zt2zm_jax(rc_coef_zt,  gr).at[:, k_ub].set(0.0)
    return wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt, rc_coef_zm


def calc_ice_cloud_frac_component(mean_chi, stdev_chi_in, crt, rsatl, tl,
                                      cf_liq, p_in_Pa):
    """pdf_closure_module.F90:calc_ice_cloud_frac_component (line 2490).

    Ice supersaturation fraction of one PDF component.  Above freezing it equals the
    liquid cloud-fraction component; below freezing it is the PDF fraction supersaturated
    w.r.t. ice (chi above chi_at_ice_sat = crt*(rsat_ice - rsatl)).
    """
    rsat_ice = sat_mixrat_ice(p_in_Pa, tl)
    delta    = mean_chi - crt * (rsat_ice - rsatl)   # chi - chi_at_ice_sat
    is_clear = (((jnp.abs(delta) <= eps) & (stdev_chi_in <= chi_tol))
                | (delta < -max_num_stdevs * stdev_chi_in))
    is_full  = delta > max_num_stdevs * stdev_chi_in
    safe_s   = jnp.maximum(stdev_chi_in, 1.0e-100)
    zeta     = delta / safe_s
    ssf_mid  = 0.5 * (1.0 + jax.scipy.special.erf(zeta / sqrt_2))
    ssf      = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, ssf_mid))
    # Above freezing: same as the liquid cloud-fraction component.
    return jnp.where(tl > T_freeze_K, cf_liq, ssf)


def calc_pdf_ice_supersat_frac_jax(*, comp):
    """Mixed ice-supersaturation fraction from the ADG1 PDF (pdf_closure_module.F90). Forms the per-component
    ice cloud fraction (calc_ice_cloud_frac_component) and combines by mixture fraction. `comp` is the
    per-component dict from calc_pdf_liquid_cloud_frac_components_jax. Returns (issf_1, issf_2, ice_supersat_frac).
    """
    _mf = comp['mixt_frac']
    _issf1 = calc_ice_cloud_frac_component(
        comp['chi_1'], comp['stdev_chi_1'], comp['crt_1'], comp['rsatl_1'],
        comp['tl_1'], comp['cf_1'], comp['p'])
    _issf2 = calc_ice_cloud_frac_component(
        comp['chi_2'], comp['stdev_chi_2'], comp['crt_2'], comp['rsatl_2'],
        comp['tl_2'], comp['cf_2'], comp['p'])
    _issf = _mf * _issf1 + (1.0 - _mf) * _issf2
    return _issf1, _issf2, _issf


def calc_pdf_skewness_diagnostics_jax(*, rtp2_zt, rtp3, thlp2_zt, thlp3, rtp2, thlp2,
                                      sigma_sqd_w_zm, wp3_zm, wp2, clubb_params, gr):
    """Diagnostic skewnesses of the post-advance PDF (pdf_closure_module.F90:4448-4465). Sk_rt and Sk_thl on
    the zt and zm grids (via Skx_func), and Skw_velocity = (1/(1-sigma_sqd_w)) * wp3_zm / max(wp2, w_tol_sqd).
    Returns (Skrt_zt, Skthl_zt, Skrt_zm, Skthl_zm, Skw_velocity).
    """
    _cp = jnp.asarray(clubb_params)
    _Skrt_zt  = Skx_func(jnp.asarray(rtp2_zt),  jnp.asarray(rtp3),  rt_tol,  _cp)
    _Skthl_zt = Skx_func(jnp.asarray(thlp2_zt), jnp.asarray(thlp3), thl_tol, _cp)
    _rtp3_zm  = zt2zm_jax(jnp.asarray(rtp3),  gr)
    _thlp3_zm = zt2zm_jax(jnp.asarray(thlp3), gr)
    _Skrt_zm  = Skx_func(jnp.asarray(rtp2),  _rtp3_zm,  rt_tol,  _cp)
    _Skthl_zm = Skx_func(jnp.asarray(thlp2), _thlp3_zm, thl_tol, _cp)
    _ssw_zm = jnp.asarray(sigma_sqd_w_zm)
    _Skw_vel = (1.0 / (1.0 - _ssw_zm)) * (jnp.asarray(wp3_zm)
                                          / jnp.maximum(jnp.asarray(wp2), w_tol_sqd))
    return _Skrt_zt, _Skthl_zt, _Skrt_zm, _Skthl_zm, _Skw_vel


def calc_pdf_chi_mean_var_jax(*, comp):
    """Grid-mean extended-liquid-water chi and its variance chip2 from the PDF components (pdf_closure).
    chi = mixt_frac-weighted component mean; chip2 = between-component + within-component (stdev_chi²) variance.
    `comp` is the per-component dict from calc_pdf_liquid_cloud_frac_components_jax. Returns (chi, chip2).
    """
    _mf = comp['mixt_frac']
    _chi1, _chi2 = comp['chi_1'], comp['chi_2']
    _schi1, _schi2 = comp['stdev_chi_1'], comp['stdev_chi_2']
    _chi = _mf * _chi1 + (1.0 - _mf) * _chi2
    _chip2 = (_mf * (_chi1 - _chi)**2
              + (1.0 - _mf) * (_chi2 - _chi)**2
              + _mf * _schi1**2
              + (1.0 - _mf) * _schi2**2)
    return _chi, _chip2



def pdf_closure_driver(
        clubb_params,
        exner,
        flags,
        gr,
        l_gamma_skw,
        l_sample,
        mixt_frac_max_mag,
        p_in_Pa,
        rtm,
        rtp2,
        rtp3,
        rtpthlp,
        stats_writer,
        thlm,
        thlp2,
        thlp3,
        thv_ds_zt,
        um,
        up2,
        upwp,
        vm,
        vp2,
        vpwp,
        wm_zt,
        wp2,
        wp2_zt,
        wp3,
        wprtp,
        wpthlp,
        pdf_params,
        rtpthvp,
        thlpthvp,
        wp2thvp,
        wpthvp,
):
    """Post-advance PDF closure — the JAX driver mirroring pdf_closure_module.F90:pdf_closure_driver
    (the ADG1 invocation + component/cloud-frac/ice-supersat/cloud-water-flux/higher-order-moment/
    skewness pieces via the extracted pdf_closure_module routines + pdf_params/stats plumbing). Returns
    the 21 PDF-derived fields/moments the caller feeds to later blocks/stats; carries the ADG1 result
    across timesteps via the module global _prev_adg1 (Block P override). The 5 input-and-output fields
    pdf_params/rtpthvp/thlpthvp/wp2thvp/wpthvp are passed in (read) and returned (rewritten).
    Pure (no module-global writes): the ADG1 result is returned as the last output (`_adg1`, None for
    non-ADG1) so the caller can update the cross-timestep `_prev_adg1` carry (mirror-refactor iter 142).
    Un-inlined from the advance_clubb_core Block U (iter 141)."""
    _adg1 = None  # ADG1 result; returned to the caller for the _prev_adg1 cross-timestep carry (Block P override)
    if flags.ipdf_call_placement == ipdf_post_advance_fields:
        l_samp_stats = True
    else:
        l_samp_stats = False  # already sampled in pre-advance call

    # All PDF-derived state updates below use the JAX-computed ADG1 values.

    # Post-advance path: invoke the ADG1 PDF driver on the post-advance state (the ADG1 call of
    # pdf_closure_module:pdf_closure_driver).
    if flags.iiPDF_type == iiPDF_ADG1:
        # Recompute zt-level fields using post-advance state (mirrors Fortran internals)
        _Skw_zt_adg = Skx_func(
            jnp.asarray(wp2_zt), jnp.asarray(wp3), w_tol, jnp.asarray(clubb_params))
        _rtp2_zt_adg = jnp.maximum(
            zm2zt_jax(jnp.asarray(rtp2), gr), rt_tol ** 2)
        _thlp2_zt_adg = jnp.maximum(
            zm2zt_jax(jnp.asarray(thlp2), gr), thl_tol ** 2)

        # Fortran pdf_closure_driver recomputes sigma_sqd_w from post-advance state
        # before calling ADG1_pdf_driver.  Replicate that here so _adg1 gets
        # the correct post-advance sigma_sqd_w (not the pre-advance value from iter18/39).
        _wp3_zm_post = _asarray(zt2zm_jax(jnp.asarray(wp3), gr))
        _Skw_zm_post = _asarray(Skx_func(
            jnp.asarray(wp2), jnp.asarray(_wp3_zm_post), w_tol, jnp.asarray(clubb_params)))
        # compute_gamma_Skw (Skx_module.F90; Fortran pdf_closure_driver `use Skx_module`)
        _gamma_post = _asarray(compute_gamma_Skw(
            _Skw_zm_post, clubb_params, l_gamma_skw))
        _ssw_post = _asarray(compute_sigma_sqd_w(
            jnp.asarray(_gamma_post),
            jnp.asarray(wp2), jnp.asarray(thlp2), jnp.asarray(rtp2),
            jnp.asarray(up2), jnp.asarray(vp2),
            jnp.asarray(wpthlp), jnp.asarray(wprtp),
            jnp.asarray(upwp), jnp.asarray(vpwp),
            flags.l_predict_upwp_vpwp, gr,
        ))
        _adg1, _wprtp_zt_adg, _wpthlp_zt_adg = adg1_pdf_driver_zt_jax(
            wm_zt=wm_zt, rtm=rtm, thlm=thlm, um=um, vm=vm, wp2_zt=wp2_zt,
            rtp2_zt=_rtp2_zt_adg, thlp2_zt=_thlp2_zt_adg, Skw_zt=_Skw_zt_adg,
            sigma_sqd_w_zt=jnp.maximum(zm2zt_jax(jnp.asarray(_ssw_post), gr), zero_threshold),
            up2=up2, vp2=vp2, wprtp=wprtp, wpthlp=wpthlp, upwp=upwp, vpwp=vpwp,
            clubb_params=clubb_params, gr=gr, mixt_frac_max_mag=mixt_frac_max_mag)

        # The ADG1 result `_adg1` is returned to the caller, which updates the cross-timestep _prev_adg1
        # carry there (under a tracer-guard) — see the caller after this function returns.

        if l_sample and l_samp_stats and stats_writer is not None:
            stats_writer.update("mixt_frac",    _asarray(_adg1['mixt_frac'],    dtype=np.float64))
            stats_writer.update("w_1",          _asarray(_adg1['w_1'],          dtype=np.float64))
            stats_writer.update("w_2",          _asarray(_adg1['w_2'],          dtype=np.float64))
            stats_writer.update("varnce_w_1",   _asarray(_adg1['varnce_w_1'],   dtype=np.float64))
            stats_writer.update("varnce_w_2",   _asarray(_adg1['varnce_w_2'],   dtype=np.float64))
            stats_writer.update("rt_1",         _asarray(_adg1['rt_1'],         dtype=np.float64))
            stats_writer.update("rt_2",         _asarray(_adg1['rt_2'],         dtype=np.float64))
            stats_writer.update("varnce_rt_1",  _asarray(_adg1['varnce_rt_1'],  dtype=np.float64))
            stats_writer.update("varnce_rt_2",  _asarray(_adg1['varnce_rt_2'],  dtype=np.float64))
            stats_writer.update("thl_1",        _asarray(_adg1['thl_1'],        dtype=np.float64))
            stats_writer.update("thl_2",        _asarray(_adg1['thl_2'],        dtype=np.float64))
            stats_writer.update("varnce_thl_1", _asarray(_adg1['varnce_thl_1'], dtype=np.float64))
            stats_writer.update("varnce_thl_2", _asarray(_adg1['varnce_thl_2'], dtype=np.float64))

        # calc_comp_corrs_binormal for rt-thl (always for ADG1)
        _rtpthlp_zt_adg = zm2zt_jax(jnp.asarray(rtpthlp), gr)
        (_corr_rt_thl_1,
         _corr_rt_thl_2) = calc_comp_corrs_binormal(
            xpyp=_rtpthlp_zt_adg,
            xm=jnp.asarray(rtm),
            ym=jnp.asarray(thlm),
            mu_x_1=_adg1['rt_1'],
            mu_x_2=_adg1['rt_2'],
            mu_y_1=_adg1['thl_1'],
            mu_y_2=_adg1['thl_2'],
            sigma_x_1_sqd=_adg1['varnce_rt_1'],
            sigma_x_2_sqd=_adg1['varnce_rt_2'],
            sigma_y_1_sqd=_adg1['varnce_thl_1'],
            sigma_y_2_sqd=_adg1['varnce_thl_2'],
            mixt_frac=_adg1['mixt_frac'],
        )

        # ============================================================ #
        # Block U: JAX rcm/cloud_frac from post-advance PDF           #
        # Mirrors Fortran pdf_closure_driver → transform_pdf_chi_eta   #
        # + calc_liquid_cloud_frac_component for the post-advance path. #
        # rcm is returned and becomes the NEXT timestep's Block I/K    #
        # input, replacing what Fortran's Block U oracle used to give.  #
        # ============================================================ #
        # Per-component liquid cloud-fraction PDF closure (pdf_closure_module.F90), shared with the
        # pre-advance path. Names below preserve the downstream `_*_padv` locals consumed by the
        # ice-supersat / xprcp / pdf_params / stats steps.
        _comp_padv = calc_pdf_liquid_cloud_frac_components_jax(
            adg1=_adg1, exner=exner, p_in_Pa=p_in_Pa,
            corr_rt_thl_1=_corr_rt_thl_1, corr_rt_thl_2=_corr_rt_thl_2,
            saturation_formula=flags.saturation_formula)
        _mfpadv    = _comp_padv['mixt_frac']
        _rt1_padv  = _comp_padv['rt_1'];   _rt2_padv  = _comp_padv['rt_2']
        _thl1_padv = _comp_padv['thl_1'];  _thl2_padv = _comp_padv['thl_2']
        _vrt1_padv = _comp_padv['varnce_rt_1'];  _vrt2_padv = _comp_padv['varnce_rt_2']
        _vthl1_padv = _comp_padv['varnce_thl_1']; _vthl2_padv = _comp_padv['varnce_thl_2']
        _exnerpadv = _comp_padv['exner']; _ppadv = _comp_padv['p']
        _tl1_padv  = _comp_padv['tl_1'];  _tl2_padv  = _comp_padv['tl_2']
        _rsatl1_padv = _comp_padv['rsatl_1']; _rsatl2_padv = _comp_padv['rsatl_2']
        (_chi1_padv, _crt1_padv, _cthl1_padv, _schi1_padv,
         _seta1_padv, _covar_ce1_padv_u, _corr_ce1_padv) = (
            _comp_padv['chi_1'], _comp_padv['crt_1'], _comp_padv['cthl_1'],
            _comp_padv['stdev_chi_1'], _comp_padv['stdev_eta_1'],
            _comp_padv['covar_ce_1'], _comp_padv['corr_ce_1'])
        (_chi2_padv, _crt2_padv, _cthl2_padv, _schi2_padv,
         _seta2_padv, _covar_ce2_padv_u, _corr_ce2_padv) = (
            _comp_padv['chi_2'], _comp_padv['crt_2'], _comp_padv['cthl_2'],
            _comp_padv['stdev_chi_2'], _comp_padv['stdev_eta_2'],
            _comp_padv['covar_ce_2'], _comp_padv['corr_ce_2'])
        _cf1_padv, _rc1_padv = _comp_padv['cf_1'], _comp_padv['rc_1']
        _cf2_padv, _rc2_padv = _comp_padv['cf_2'], _comp_padv['rc_2']

        # rcm/cloud_frac for the NEXT timestep's Block I/K
        _cloud_frac_padv = _comp_padv['cloud_frac']
        _rcm_padv        = _comp_padv['rcm']
        rcm        = _asarray(_rcm_padv,        dtype=np.float64)
        cloud_frac = _asarray(_cloud_frac_padv, dtype=np.float64)

        # Update ice_supersat_frac from Block U ADG1 output (l_calc_ice_supersat_frac
        # is hardcoded .true. in pdf_closure_module.F90:928). Faithful port of
        # calc_ice_cloud_frac_component (pdf_closure_module.F90:2490): for levels above
        # freezing it equals the liquid cloud_frac; for below-freezing levels it is the
        # PDF fraction supersaturated w.r.t. ICE (chi above chi_at_ice_sat). The old
        # warm-only shortcut (=cloud_frac) gave 0 at cold, ice-supersaturated layers
        # (e.g. ekman's 10 km top, T~203 K), corrupting the splat Brunt-Vaisala term.
        _issf1_padv, _issf2_padv, _issf_padv = calc_pdf_ice_supersat_frac_jax(comp=_comp_padv)
        ice_supersat_frac = _asarray(_issf_padv, dtype=np.float64)

        # propagate the zt-level PDF component moments into the returned pdf_params so the
        # KK microphysics (kk_microphys_step) can read them. The JAX otherwise computes them as Block-U
        # locals that flow only to stats; pdf_params was zero-initialized (it is a fallback for
        # non-ADG1/non-ARM, :1815), and the 15 non-microphysics cases never read these fields — so
        # populating them is safe (verified: ARM/bomex still bit-faithful).
        pdf_params = pdf_params._replace(
            chi_1=_asarray(_chi1_padv, np.float64), chi_2=_asarray(_chi2_padv, np.float64),
            stdev_chi_1=_asarray(_schi1_padv, np.float64), stdev_chi_2=_asarray(_schi2_padv, np.float64),
            cloud_frac_1=_asarray(_cf1_padv, np.float64), cloud_frac_2=_asarray(_cf2_padv, np.float64),
            rc_1=_asarray(_rc1_padv, np.float64), rc_2=_asarray(_rc2_padv, np.float64),
            mixt_frac=_asarray(_mfpadv, np.float64),
            thl_1=_asarray(_thl1_padv, np.float64), thl_2=_asarray(_thl2_padv, np.float64),
            ice_supersat_frac_1=_asarray(_issf1_padv, np.float64),
            ice_supersat_frac_2=_asarray(_issf2_padv, np.float64),
            # the additional component moments the KK second-moment covariance driver
            # (KK_upscaled_covar_driver) consumes — w/eta/rt means+stdevs, the chi-eta transform
            # coefficients, and corr_chi_eta. corr_w_chi/corr_w_eta stay 0 (ADG1, pdf_closure:1037).
            w_1=_asarray(_adg1['w_1'], np.float64), w_2=_asarray(_adg1['w_2'], np.float64),
            varnce_w_1=_asarray(_adg1['varnce_w_1'], np.float64),
            varnce_w_2=_asarray(_adg1['varnce_w_2'], np.float64),
            rt_1=_asarray(_rt1_padv, np.float64), rt_2=_asarray(_rt2_padv, np.float64),
            stdev_eta_1=_asarray(_seta1_padv, np.float64), stdev_eta_2=_asarray(_seta2_padv, np.float64),
            crt_1=_asarray(_crt1_padv, np.float64), crt_2=_asarray(_crt2_padv, np.float64),
            cthl_1=_asarray(_cthl1_padv, np.float64), cthl_2=_asarray(_cthl2_padv, np.float64),
            corr_chi_eta_1=_asarray(_corr_ce1_padv, np.float64),
            corr_chi_eta_2=_asarray(_corr_ce2_padv, np.float64))

        # ============================================================ #
        # Block U: JAX cloud water flux variables                     #
        # Mirrors Fortran calc_xprcp_component (pdf_closure_module.F90 #
        # lines 3089-3104). For ADG1 no corr_w_chi correction (lines  #
        # 3112-3138 only run for non-ADG1 PDF types).                  #
        # After mixing: convert zt→zm, zero k_ub_zm (lines 4233-4261).#
        # ============================================================ #
        # Mixed cloud-water turbulent fluxes (pdf_closure_module: calc_pdf_xprcp_fluxes_jax — the
        # x'rc' section of pdf_closure_driver). Names below preserve the downstream `_*_padv` locals.
        _xprcp = calc_pdf_xprcp_fluxes_jax(
            adg1=_adg1, comp=_comp_padv, wm_zt=wm_zt, rtm=rtm, thlm=thlm, um=um, vm=vm,
            rcm_zt=_rcm_padv, gr=gr)
        _wprcp_zt_padv   = _xprcp['wprcp_zt']
        _rtprcp_zt_padv  = _xprcp['rtprcp_zt']
        _thlprcp_zt_padv = _xprcp['thlprcp_zt']
        _wprcp_zm_padv   = _xprcp['wprcp_zm']
        _rtprcp_zm_padv  = _xprcp['rtprcp_zm']
        _thlprcp_zm_padv = _xprcp['thlprcp_zm']
        _uprcp_zm_padv   = _xprcp['uprcp_zm']
        _vprcp_zm_padv   = _xprcp['vprcp_zm']

        wprcp_out = _asarray(_wprcp_zm_padv,    dtype=np.float64)
        _wp2rcp   = _asarray(_xprcp['wp2rcp_zt'], dtype=np.float64)  # stays on zt
        _rtprcp   = _asarray(_rtprcp_zm_padv,   dtype=np.float64)
        thlprcp   = _asarray(_thlprcp_zm_padv,  dtype=np.float64)
        uprcp     = _asarray(_uprcp_zm_padv,    dtype=np.float64)
        vprcp     = _asarray(_vprcp_zm_padv,    dtype=np.float64)

        # pdf_closure higher-order-moment section (pdf_closure_module.F90 body),
        # now a module routine (calc_pdf_higher_order_moments_jax, mirror-refactor
        # iter 37). For ADG1 corr_w_rt=corr_w_thl=corr_u_w=corr_v_w=0; only the
        # per-component corr_rt_thl is non-zero (feeds wprtpthlp).
        _hom = calc_pdf_higher_order_moments_jax(
            _adg1, jnp.asarray(wm_zt), jnp.asarray(rtm), jnp.asarray(thlm),
            jnp.asarray(um), jnp.asarray(vm),
            _corr_rt_thl_1, _corr_rt_thl_2, gr)

        # <x'thv'> buoyancy-flux assembly (pdf_closure_module.F90:1122-1158) +
        # the pdf_closure_driver zt→zm regrid (F90:4233-4261), now a module routine
        # (calc_xpthvp_terms_jax, mirror-refactor iter v2). The native zt-grid rc-flux
        # moments from calc_xprcp_component (NOT a zt→zm→zt round-trip of the zm output)
        # are fed in: Fortran computes wprcp and wpthvp=...+rc_coef*wprcp on the same pdf
        # grid in one pass; the round-trip would smooth sharp cloud-top wprcp gradients
        # (~5e-4 error for thick-cloud DYCOMS/BOMEX).
        (_wpthvp_zm_dc, _wp2thvp_zt_dc, _rtpthvp_zm_dc, _thlpthvp_zm_dc,
         _rc_coef_dc, _rc_coef_zm_dc) = calc_xpthvp_terms_jax(
            jnp.asarray(exner), jnp.asarray(thv_ds_zt),
            _wprcp_zt_padv, jnp.asarray(_wp2rcp), _rtprcp_zt_padv, _thlprcp_zt_padv,
            _wpthlp_zt_adg, _wprtp_zt_adg, _hom['wp2thlp'], _hom['wp2rtp'],
            zm2zt_jax(jnp.asarray(rtpthlp), gr), zm2zt_jax(jnp.asarray(rtp2), gr),
            zm2zt_jax(jnp.asarray(thlp2), gr), gr)

        # Override pdf_closure_driver state with the JAX-computed values
        # (replaces the Fortran pdf_closure_driver for the ADG1 path).
        sigma_sqd_w = _ssw_post
        wpthvp    = _asarray(_wpthvp_zm_dc)
        wp2thvp   = _asarray(_wp2thvp_zt_dc)
        rtpthvp   = _asarray(_rtpthvp_zm_dc)
        thlpthvp  = _asarray(_thlpthvp_zm_dc)
        # rc_coef_zm carried (post-advance placement) to the NEXT step's
        # diagnose_upxp/upthvp cloud term (rc_coef_zm*uprcp).
        rc_coef_zm = _asarray(_rc_coef_zm_dc, dtype=np.float64)
        wpup2     = _asarray(_hom['wpup2'])
        wpvp2     = _asarray(_hom['wpvp2'])
        wp2up2    = _asarray(_hom['wp2up2_zm'])
        wp2vp2    = _asarray(_hom['wp2vp2_zm'])
        wp4       = _asarray(_hom['wp4_zm'])
        wp2rtp    = _asarray(_hom['wp2rtp'])
        wp2thlp   = _asarray(_hom['wp2thlp'])
        wp2up     = _asarray(_hom['wp2up'])
        wprtp2    = _asarray(_hom['wprtp2'])
        wpthlp2   = _asarray(_hom['wpthlp2'])
        wprtpthlp = _asarray(_hom['wprtpthlp'])
        # Update carry variables: Fortran's advance_clubb_core local wpthlp2/wprtp2/wprtpthlp
        # persist on the stack between calls for ipdf_post_advance_fields. JAX explicitly
        # carries them so next timestep's advance_xp2_xpyp sees the post-advance values.
        _wprtp2    = _asarray(_hom['wprtp2'])
        _wpthlp2   = _asarray(_hom['wpthlp2'])
        _wprtpthlp = _asarray(_hom['wprtpthlp'])

        if l_sample and l_samp_stats and stats_writer is not None:
            # pdf_closure_module.F90 stats for Block U ADG1 variables
            # Skw and sigma_sqd_w: written from inside Fortran pdf_closure using post-advance
            # state (Fortran pdf_closure_module.F90:4446, 4447, 4454, 4512).
            # _Skw_zm_post and _Skw_zt_adg are post-advance Skw; _ssw_post is post-advance sigma_sqd_w.
            stats_writer.update("Skw_zm", _asarray(_Skw_zm_post, dtype=np.float64))
            stats_writer.update("Skw_zt", _asarray(_Skw_zt_adg, dtype=np.float64))
            stats_writer.update("sigma_sqd_w", _ssw_post)
            stats_writer.update("gamma_Skw_fnc", _gamma_post)
            stats_writer.update("corr_rt_thl_1", _asarray(_corr_rt_thl_1, dtype=np.float64))
            stats_writer.update("corr_rt_thl_2", _asarray(_corr_rt_thl_2, dtype=np.float64))
            stats_writer.update("wp2rtp",    _asarray(_hom['wp2rtp'],    dtype=np.float64))
            stats_writer.update("wp2thlp",   _asarray(_hom['wp2thlp'],   dtype=np.float64))
            stats_writer.update("wp2up",     _asarray(_hom['wp2up'],     dtype=np.float64))
            stats_writer.update("wpup2",     _asarray(_hom['wpup2'],     dtype=np.float64))
            stats_writer.update("wpvp2",     _asarray(_hom['wpvp2'],     dtype=np.float64))
            stats_writer.update("wp2up2",    _asarray(_hom['wp2up2_zm'], dtype=np.float64))
            stats_writer.update("wp2vp2",    _asarray(_hom['wp2vp2_zm'], dtype=np.float64))
            stats_writer.update("wp4",       _asarray(_hom['wp4_zm'],    dtype=np.float64))
            stats_writer.update("wprtp2",    _asarray(_hom['wprtp2'],    dtype=np.float64))
            stats_writer.update("wpthlp2",   _asarray(_hom['wpthlp2'],   dtype=np.float64))
            stats_writer.update("wprtpthlp", _asarray(_hom['wprtpthlp'], dtype=np.float64))
            stats_writer.update("wpthvp",    _asarray(wpthvp,         dtype=np.float64))
            stats_writer.update("wp2thvp",   _asarray(wp2thvp,        dtype=np.float64))
            stats_writer.update("rtpthvp",   _asarray(rtpthvp,        dtype=np.float64))
            stats_writer.update("thlpthvp",  _asarray(thlpthvp,       dtype=np.float64))
            # PDF cloud-water fluxes (computed in the Block-U cloud-water-flux section; Fortran writes these).
            # Outputting them lets the diagnostic compare verify the cloud-flux
            # physics directly instead of reading an unwritten 0.
            stats_writer.update("wprcp",     _asarray(_wprcp_zm_padv,   dtype=np.float64))
            stats_writer.update("rtprcp",    _asarray(_rtprcp_zm_padv,  dtype=np.float64))
            stats_writer.update("thlprcp",   _asarray(_thlprcp_zm_padv, dtype=np.float64))
            stats_writer.update("uprcp",     _asarray(_uprcp_zm_padv,   dtype=np.float64))
            stats_writer.update("vprcp",     _asarray(_vprcp_zm_padv,   dtype=np.float64))

            # rc_coef and rc_coef_zm (Fortran pdf_closure_module.F90 line 4503-4519).
            # rc_coef_zm reuses the value already returned by calc_xpthvp_terms_jax (the regrid is
            # identical: zt2zm(rc_coef_zt) with the upper boundary zeroed) instead of recomputing it.
            stats_writer.update("rc_coef", _asarray(_rc_coef_dc, dtype=np.float64))
            stats_writer.update("rc_coef_zm", _asarray(_rc_coef_zm_dc, dtype=np.float64))

            # Sk_rt/Sk_thl (zt and zm) + Skw_velocity (pdf_closure_module.F90:4448-4465)
            (_Skrt_zt_dg, _Skthl_zt_dg, _Skrt_zm_dg, _Skthl_zm_dg, _Skw_vel_dg) = (
                calc_pdf_skewness_diagnostics_jax(
                    rtp2_zt=_rtp2_zt_adg, rtp3=rtp3, thlp2_zt=_thlp2_zt_adg, thlp3=thlp3,
                    rtp2=rtp2, thlp2=thlp2, sigma_sqd_w_zm=sigma_sqd_w, wp3_zm=_wp3_zm_post,
                    wp2=wp2, clubb_params=clubb_params, gr=gr))
            stats_writer.update("Skrt_zt",  _asarray(_Skrt_zt_dg,  dtype=np.float64))
            stats_writer.update("Skthl_zt", _asarray(_Skthl_zt_dg, dtype=np.float64))
            stats_writer.update("Skrt_zm",  _asarray(_Skrt_zm_dg,  dtype=np.float64))
            stats_writer.update("Skthl_zm", _asarray(_Skthl_zm_dg, dtype=np.float64))
            stats_writer.update("Skw_velocity", _asarray(_Skw_vel_dg, dtype=np.float64))

            # rsatl_1/2, chi_1/2, crt/cthl, stdev_chi/eta, covar/corr_chi_eta, chi, chip2.
            # Reuse the chi-eta transform already computed for the post-advance cloud fraction
            # above — the inputs (tl/rsatl/rt/exner/varnce/corr_rt_thl) are identical, so this just
            # surfaces those same values as stats (Fortran pdf_closure computes the transform once).
            stats_writer.update("rsatl_1", _asarray(_rsatl1_padv, dtype=np.float64))
            stats_writer.update("rsatl_2", _asarray(_rsatl2_padv, dtype=np.float64))

            (_chi1_dg, _crt1_dg, _cthl1_dg, _stdev_chi1_dg, _stdev_eta1_dg,
             _covar_ce1_dg, _corr_ce1_dg) = (
                _chi1_padv, _crt1_padv, _cthl1_padv, _schi1_padv, _seta1_padv,
                _covar_ce1_padv_u, _corr_ce1_padv)
            (_chi2_dg, _crt2_dg, _cthl2_dg, _stdev_chi2_dg, _stdev_eta2_dg,
             _covar_ce2_dg, _corr_ce2_dg) = (
                _chi2_padv, _crt2_padv, _cthl2_padv, _schi2_padv, _seta2_padv,
                _covar_ce2_padv_u, _corr_ce2_padv)

            stats_writer.update("chi_1",          _asarray(_chi1_dg,       dtype=np.float64))
            stats_writer.update("chi_2",          _asarray(_chi2_dg,       dtype=np.float64))
            stats_writer.update("crt_1",          _asarray(_crt1_dg,       dtype=np.float64))
            stats_writer.update("crt_2",          _asarray(_crt2_dg,       dtype=np.float64))
            stats_writer.update("cthl_1",         _asarray(_cthl1_dg,      dtype=np.float64))
            stats_writer.update("cthl_2",         _asarray(_cthl2_dg,      dtype=np.float64))
            stats_writer.update("stdev_chi_1",    _asarray(_stdev_chi1_dg, dtype=np.float64))
            stats_writer.update("stdev_chi_2",    _asarray(_stdev_chi2_dg, dtype=np.float64))
            stats_writer.update("stdev_eta_1",    _asarray(_stdev_eta1_dg, dtype=np.float64))
            stats_writer.update("stdev_eta_2",    _asarray(_stdev_eta2_dg, dtype=np.float64))
            stats_writer.update("covar_chi_eta_1",_asarray(_covar_ce1_dg,  dtype=np.float64))
            stats_writer.update("covar_chi_eta_2",_asarray(_covar_ce2_dg,  dtype=np.float64))
            stats_writer.update("corr_chi_eta_1", _asarray(_corr_ce1_dg,   dtype=np.float64))
            stats_writer.update("corr_chi_eta_2", _asarray(_corr_ce2_dg,   dtype=np.float64))

            # Grid-mean chi + its variance chip2 (pdf_closure_module.calc_pdf_chi_mean_var_jax)
            _chi_dg, _chip2_dg = calc_pdf_chi_mean_var_jax(comp=_comp_padv)
            stats_writer.update("chi", _asarray(_chi_dg, dtype=np.float64))
            stats_writer.update("chip2", _asarray(_chip2_dg, dtype=np.float64))
    return (cloud_frac, ice_supersat_frac, pdf_params, rc_coef_zm, rcm, rtpthvp, thlprcp, thlpthvp, uprcp, vprcp, wp2rtp, wp2thlp, wp2thvp, wp2up, wp2up2, wp2vp2, wp4, wprcp_out, wpthvp, wpup2, wpvp2, _adg1)

__all__ = [
    "calc_wp2xp_pdf", "calc_wpxp2_pdf", "calc_wp2xp2_pdf",
    "calc_wp4_pdf", "calc_wpxpyp_pdf",
    "calc_w_up_in_cloud",
    "transform_pdf_chi_eta_component",
    "calc_liquid_cloud_frac_component",
    "calc_ice_cloud_frac_component",
    "calc_xprcp_component",
    "calc_xpthvp_terms_jax",
    "adg1_pdf_driver_zt_jax",
    "pdf_closure_driver",
]


def adg1_pdf_driver_zt_jax(*, wm_zt, rtm, thlm, um, vm, wp2_zt, rtp2_zt, thlp2_zt,
                           Skw_zt, sigma_sqd_w_zt, up2, vp2, wprtp, wpthlp, upwp, vpwp,
                           clubb_params, gr, mixt_frac_max_mag):
    """Shared zt-regrid + ADG1_pdf_driver call for the pre- and post-advance PDF-closure paths.

    Mirrors the ADG1 invocation inside pdf_closure_module.F90:pdf_closure_driver — regrids the zm-level
    moments to zt, then calls ADG1_pdf_driver. The two call sites (Block I_pre pre-advance, Block U
    post-advance) differ only in how rtp2_zt/thlp2_zt/Skw_zt/sigma_sqd_w_zt are derived, which are passed in.
    Returns the ADG1 result dict plus the regridded wprtp_zt/wpthlp_zt (the post-advance path reuses them for
    its xpthvp-terms assembly).
    """
    _up2_zt = jnp.maximum(zm2zt_jax(jnp.asarray(up2), gr), w_tol_sqd)
    _vp2_zt = jnp.maximum(zm2zt_jax(jnp.asarray(vp2), gr), w_tol_sqd)
    _wprtp_zt = zm2zt_jax(jnp.asarray(wprtp), gr)
    _wpthlp_zt = zm2zt_jax(jnp.asarray(wpthlp), gr)
    _upwp_zt = zm2zt_jax(jnp.asarray(upwp), gr)
    _vpwp_zt = zm2zt_jax(jnp.asarray(vpwp), gr)
    _sqrt_wp2_zt = jnp.sqrt(jnp.asarray(wp2_zt))
    _beta = jnp.asarray(clubb_params[:, ibeta - 1])
    _adg1 = ADG1_pdf_driver(
        wm=jnp.asarray(wm_zt),
        rtm=jnp.asarray(rtm),
        thlm=jnp.asarray(thlm),
        um=jnp.asarray(um),
        vm=jnp.asarray(vm),
        wp2=jnp.asarray(wp2_zt),
        rtp2=jnp.asarray(rtp2_zt),
        thlp2=jnp.asarray(thlp2_zt),
        up2=_up2_zt,
        vp2=_vp2_zt,
        Skw=jnp.asarray(Skw_zt),
        wprtp=_wprtp_zt,
        wpthlp=_wpthlp_zt,
        upwp=_upwp_zt,
        vpwp=_vpwp_zt,
        sqrt_wp2=_sqrt_wp2_zt,
        sigma_sqd_w=jnp.asarray(sigma_sqd_w_zt),
        beta=_beta,
        mixt_frac_max_mag=mixt_frac_max_mag,
    )
    return _adg1, _wprtp_zt, _wpthlp_zt
